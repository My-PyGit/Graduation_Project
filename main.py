from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt,  QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from UI.MainWindow import Ui_MainWindow
import numpy as np
import sys
import torch
import argparse
from config import build_model_config
from models.detectors import build_model
from utils.misc import load_weight, compute_flops
from utils.datasets import LoadImages
import cv2
from utils.box_ops import rescale_bboxes
from utils.vis_tools import visualize
import time
import json
import os
from utils.CustomMessageBox import MessageBox

#YOLO线程
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray) #发送检测后信息（图像）
    send_raw = pyqtSignal(np.ndarray) #发送检测前信息（图像）
    send_statistic = pyqtSignal(dict) #发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str) #发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_percent = pyqtSignal(int) #发送信号：设置进度条
    # send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.args = self.parse_args() #参数列表
        self.source = '0'  # 视频源/图片源/摄像头源
        self.jump_out = False  # 跳出循环
        self.is_continue = True  # 继续/暂停
        self.percent_length = 100  # 进度条
        self.rate_check = True  # 是否启用延时
        self.rate = 100  # 延时HZ
        # self.conf_thres = 0.5  # 置信度 用于NMS
        # self.iou_thres = 0.45  # iou 用于NMS

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
        # Basic setting
        parser.add_argument('-size', '--img_size', default=640, type=int,
                            help='the max size of input image')
        parser.add_argument('--cuda', action='store_true', default=False,
                            help='use cuda.')
        parser.add_argument('--save_folder', default='det_results/', type=str,
                            help='Dir to save results')
        parser.add_argument('-ws', '--window_scale', default=1.0, type=float,
                            help='resize window of cv2 for visualization.')
        parser.add_argument('--resave', action='store_true', default=False,
                            help='resave checkpoints without optimizer state dict.')

        # Model setting
        parser.add_argument('-m', '--model', default='yolov5_n', type=str,
                            help='build yolo')
        parser.add_argument('--weight', default='weights/voc/yolov5_n/yolov5_n_best_tiny_150epoch.pth',
                            type=str, help='Trained state_dict file path to open')
        parser.add_argument('-ct', '--conf_thresh', default=0.3, type=float,
                            help='confidence threshold')
        parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                            help='NMS threshold')
        parser.add_argument('--topk', default=100, type=int,
                            help='topk candidates dets of each level before NMS')
        parser.add_argument("--no_decode", action="store_true", default=False,
                            help="not decode in inference or yes")
        parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                            help='fuse Conv & BN')
        parser.add_argument('--no_multi_labels', action='store_true', default=False,
                            help='Perform post-process with multi-labels trick.')
        parser.add_argument('--nms_class_agnostic', action='store_true', default=False,
                            help='Perform NMS operations regardless of category.')

        # Data setting
        parser.add_argument('-d', '--dataset', default='voc',
                            help='coco, voc.')
        parser.add_argument('--min_box_size', default=8.0, type=float,
                            help='min size of target bounding box.')
        parser.add_argument('--mosaic', default=None, type=float,
                            help='mosaic augmentation.')
        parser.add_argument('--mixup', default=None, type=float,
                            help='mixup augmentation.')
        parser.add_argument('--load_cache', action='store_true', default=False,
                            help='load data into memory.')

        return parser.parse_args()

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            update=False,  # update all models
            # project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            num_classes = 2,
            names = ['insulotor','defect']
            ):
        try:
            if self.args.cuda:
                print('use cuda')
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                print('use cpu')
            #load model
            model_cfg = build_model_config(self.args)
            np.random.seed(0)
            class_colors = [(np.random.randint(255),
                            np.random.randint(255),
                            np.random.randint(255)) for _ in range(num_classes)]
            # build model
            model = build_model(self.args, model_cfg, device, num_classes, False)

            # load trained weight
            model = load_weight(model, self.args.weight, self.args.fuse_conv_bn)
            model.to(device).eval()


            # model stride
            stride = int(max(model.stride))
            # imgsz = check_img_size(imgsz, s=stride)  # check image size


            #构建dataset
            dataset = LoadImages(args=self.args,path=self.source, img_size=imgsz, stride=stride,device=device)

            count = 0 #用于进度条显示
            dataset = iter(dataset)

            #开始检测
            while True:
                if self.jump_out: #按停止按钮手动停止
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('stopped')
                    break
                if self.is_continue: #暂停开关
                    path, x, im0s, self.vid_cap, orig_w, orig_h, ratio = next(dataset) #获取数据 x:transform后的图片，im0s
                    im0s_copy = im0s.copy()
                    count += 1 #一帧
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)  # get(cv2.CAP_PROP_FRAME_COUNT)：获取视频总帧数
                    else:  # 否则无法读取摄像头帧，意味着已经结束，则设置进度条为100
                        percent = self.percent_length
                    self.send_percent.emit(percent)  # 设置进度条
                    statistic_dic = {name: 0 for name in names}

                    #前向推理
                    outputs = model(x)
                    scores = outputs['scores']
                    labels = outputs['labels']
                    bboxes = outputs['bboxes']
                    bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

                    img_processed = visualize(image=im0s,
                                              bboxes=bboxes,
                                              scores=scores,
                                              labels=labels,
                                              class_colors=class_colors,
                                              class_names=names,
                                              class_indexs=[0,1])
                    # 控制视频发送频率
                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    # print(type(im0s))
                    self.send_img.emit(img_processed)
                    self.send_raw.emit(im0s_copy if isinstance(im0s_copy, np.ndarray) else im0s_copy[0])
                    self.send_statistic.emit(statistic_dic)

                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('Detection ended')
                        # 正常跳出循环
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # yolov5线程
        self.det_thread = DetThread()

        self.det_thread.source = '0'  # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.input_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x,self.output_video))  # 在label中显示图像 lambda传参时使用 x是emit发送的变量，使用lambda表达式是为了传递self.out_video参数。
        # self.det_thread.send_statistic.connect(self.show_statistic)  # 实时统计结果
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))  #
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))  # 进度条

        self.fileButton.clicked.connect(self.open_file)

        self.runButton.clicked.connect(self.run_or_continue)  # 开始或继续按钮
        self.stopButton.clicked.connect(self.stop)  # 停止按钮

    def statistic_msg(self, msg): #最下面状态栏信息
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg): #在状态栏显示开始，暂停...等信息
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def open_file(self):  #保持上一次打开的文件
        config_file = './config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Select a video or picture', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('The loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Paused')

    # 退出检测循环
    def stop(self):
        # self.out_video.clear()
        self.det_thread.jump_out = True

    @staticmethod
    def show_image(img_src, label): #显示图像
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Warning', 'Do you really want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.det_thread.jump_out = True
            event.accept()
            MessageBox(
                title='   Prompt   ', text='        Closing.......             ', time=1000, auto=True).exec_()
            sys.exit(0)
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
