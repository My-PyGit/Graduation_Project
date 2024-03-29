from pathlib import Path
import glob
import os
import cv2
import numpy as np
from dataset.build import build_transform
from config import build_dataset_config, build_model_config, build_trans_config

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class LoadImages:  # for inference
    def __init__(self, args, path, img_size=640, stride=32, device = 'cpu'):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p): #如果给定路径是目录
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p): #如果给定路径是文件
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos #把一次选中的视频和图片的路径汇总
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv #如果为False。则读取图片，如果为True,则读取视频
        self.mode = 'image' #模式初始化为图片
        self.device = device
        self.args = args
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self): #该方法用于返回一个迭代对象self，该对象self实现了__next__方法
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf: #读取完成，停止迭代
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read() #ret_val：bool表示是否读取成功
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            # img0 = cv2.imread(path)  # BGR
            img0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)

            assert img0 is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        orig_h, orig_w, _ = img0.shape
        # prepare
        x, _, ratio = self.transform(self.args,img0)
        x = x.unsqueeze(0).to(self.device)


        return path, x, img0, self.cap, orig_w, orig_h, ratio#img0:直接读取的图像 img:经过处理的图像

    def transform(self,args,image):
        model_cfg = build_model_config(args)
        trans_cfg = build_trans_config(model_cfg['trans_type'])
        val_transform, trans_cfg = build_transform(args, trans_cfg, model_cfg['max_stride'], is_train=False)
        return val_transform(image)

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files