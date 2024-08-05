# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
    Dataloaders and dataset utils
    这个文件主要是进行数据增强操作。
"""

"""
    ·注释来源于各位大佬的视频+博客，收集不易，祝你早日出sci！
    ·秉持开源精神！取之于大佬，用之于各位！
    ·@Dragon AI 
"""

import glob # python自己带的一个文件操作相关模块 查找符合自己目的的文件(如模糊匹配)
import hashlib  # 哈希模块 提供了多种安全方便的hash方法
import json # json文件操作模块
import math # 数学公式模块
import os   # 与操作系统进行交互的模块 包含文件路径操作和解析
import random
import shutil    # 文件夹、压缩包处理模块
import time # 时间模块 更底层
from itertools import repeat     # 复制模块
from multiprocessing.pool import Pool, ThreadPool    # 多线程模块 线程池
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread     # 多线程操作模块
from zipfile import ZipFile

import cv2   # opencv模块
import numpy as np   # numpy矩阵操作模块
import torch
import torch.nn.functional as F # PyTorch函数接口 封装了很多卷积、池化等函数
import yaml  # yaml文件操作模块
from PIL import ExifTags, Image, ImageOps    # 图片、相机操作模块
from torch.utils.data import DataLoader, Dataset, dataloader, distributed   # 自定义数据集模块
from tqdm import tqdm   # 进度条模块

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes

# ---------------------------------------- 1、相机设置 ----------------------------------------
# 相机相关设置，当使用相机采样时才会使用
# 专门为数码相机的照片而设定  可以记录数码照片的属性信息和拍摄数据
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(paths):
    # 返回文件列表的hash值
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # 获取数码相机的图片宽高信息  并且判断是否需要旋转（数码相机可以多角度拍摄）
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    """
        这个函数会在train.py中被调用，用于生成Trainloader, dataset，testloader：

        自定义dataloader函数:    调用LoadImagesAndLabels获取数据集dataset(包括数据增强) + 调用分布式采样器DistributedSampler
                                    + 自定义InfiniteDataLoader 进行永久持续的采样数据 + 获取dataloader。

        关键核心是LoadImagesAndLabels()，这个文件的后面所有代码都是围绕这个模块进行的。

        :param path: 图片数据加载路径 train/test  如: ../datasets/VOC/images/train2007
        :param imgsz: train/test图片尺寸（数据增强后大小） 640
        :param batch_size: batch size 大小 8/16/32
        :param stride: 模型最大stride=32   [32 16 8]
        :param single_cls: 数据集是否是单类别 默认False
        :param hyp: 超参列表dict 网络训练时的一些超参数，包括学习率等，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
        :param augment: 是否要进行数据增强  True
        :param cache: 是否cache_images False
        :param pad: 设置矩形训练的shape时进行的填充 默认0.0
        :param rect: 是否开启矩形train/test  默认训练集关闭 验证集开启
        :param rank:  多卡训练时的进程编号 rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式  默认-1
        :param workers: dataloader的numworks 加载数据时的cpu进程数
        :param image_weights: 训练时是否根据图片样本真实框分布权重来选择图片  默认False
        :param quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
        :param prefix: 显示信息   一个标志，多为train/val，处理标签时保存cache文件会用到
        """
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # 主进程实现数据的预读取并缓存，然后其它子进程则从缓存中读取数据并进行一系列运算。
    # 为了完成数据的正常同步, yolov5基于torch.distributed.barrier()函数实现了上下文管理器
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        # 载入文件数据(增强数据集)
        # Todo LoadImagesAndLabels入口
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache, # 缓存：使图片与便签进行缓存，使得第二次训练时更快读取
                                      single_cls=single_cls, # 单类别才使用
                                      stride=int(stride), # 从输入到输出降采样的比例
                                      pad=pad,  # 图片大小不同时，边角填充
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # 分布式采样器DistributedSampler
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # 使用InfiniteDataLoader和_RepeatSampler来对DataLoader进行封装, 代替原先的DataLoader, 能够永久持续的采样数据
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  # num_workers=nw,
                  num_workers=0,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """
        当image_weights=False时（不根据图片样本真实框分布权重来选择图片）就会调用这两个函数进行自定义DataLoader，进行持续性采样。
        在上面的create_dataloader模块中被调用。

        https://github.com/ultralytics/yolov5/pull/876
        使用InfiniteDataLoader和_RepeatSampler来对DataLoader进行封装, 代替原先的DataLoader, 能够永久持续的采样数据
        Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 调用_RepeatSampler进行持续采样
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever
    这部分是进行持续采样
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    """
        用到很少 load web网页中的数据
    """
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    """
        load 文件夹中视频流
        multiple IP or RTSP cameras
        定义迭代器 用于detect.py
    """
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'     # 初始化mode为images
        self.img_size = img_size
        self.stride = stride      # 最大下采样步长

        # 如果sources为一个保存了多个视频流的文件  获取每一个视频流，保存为一个列表
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            # 反之，只有一个视频流文件就直接保存
            sources = [sources]

        n = len(sources)    # 视频流个数
        # 初始化图片 fps 总帧数 线程数
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto

        # 遍历每一个视频流
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            # 打印当前视频index/总视频数/视频流地址
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            # s='0'打开本地摄像头，否则打开视频流地址
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            # 获取视频的宽和长
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            # 帧数
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            # 获取视频的帧率
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            # 读取当前画面
            _, self.imgs[i] = cap.read()  # guarantee first frame
            # 创建多线程读取视频流，daemon表示主线程结束时子线程也结束
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        # 获取进行resize+pad之后的shape，letterbox函数默认(参数auto=True)是按照矩形推理进行填充
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            # 每4帧读取一次
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack  将读取的图片拼接到一起
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    """
        这个文件是根据数据集中所有图片的路径找到数据集中所有labels对应的路径。
        用在LoadImagesAndLabels模块的__init__函数中。

        :params img_paths: {list: 50}  整个数据集的图片相对路径  例如: '..\\datasets\\VOC\\images\\train2007\\000012.jpg'
                                                        =>   '..\\datasets\\VOC\\labels\\train2007\\000012.jpg'
    """
    # Define label paths as a function of image paths
    # 因为python是跨平台的,在Windows上,文件的路径分隔符是'\',在Linux上是'/'
    # 为了让代码在不同的平台上都能运行，那么路径应该写'\'还是'/'呢？ os.sep根据你所处的平台, 自动采用相应的分隔符号
    # sa: '\\images\\'    sb: '\\labels\\'
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    # 把img_paths中所以图片路径中的images替换为labels
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    """
        这个部分是数据载入（数据增强）部分，也就是自定义数据集部分，继承自Dataset，需要重写__init__,__getitem()__等抽象方法，
        另外目标检测一般还需要重写collate_fn函数。所以，理解这三个函数是理解数据增强（数据载入）的重中之重。

    """
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        """
                初始化过程并没有什么实质性的操作,更多是一个定义参数的过程（self参数）,以便在__getitem()__中进行数据增强操作,
                所以这部分代码只需要抓住self中的各个变量的含义就算差不多了

                __init__()主要步骤
                        1、赋值一些基础的self变量 用于后面在__getitem__中调用
                        2、得到path路径下的所有图片的路径self.img_files
                        3、根据imgs路径找到labels的路径self.label_files
                        4、cache label
                        5、Read cache 生成self.labels、self.shapes、self.img_files、self.label_files、self.batch、self.n、self.indices等变量
                        6、为Rectangular Training作准备: 生成self.batch_shapes
                        7、是否需要cache image(一般不需要，太大了)

                self.img_files: {list: N} 存放着整个数据集图片的相对路径
                self.label_files: {list: N} 存放着整个数据集图片的相对路径
                cache label -> verify_image_label
                self.labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
                             否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
                self.shapes: 所有图片的shape
                self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
                               否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
                self.batch: 记载着每张图片属于哪个batch
                self.n: 数据集中所有图片的数量
                self.indices: 记载着所有图片的index
                self.rect=True时self.batch_shapes记载每个batch的shape(同一个batch的图片shape相同)
        """
        # 1、赋值一些基础的self变量 用于后面在__getitem__中调用
        self.img_size = img_size    # 经过数据增强后的数据图片的大小
        self.augment = augment  # 是否启动数据增强 一般训练时打开 验证时关闭
        self.hyp = hyp  # 超参列表
        # 图片按权重采样  True就可以根据类别频率(频率高的权重小,反正大)来进行采样  默认False: 不作类别区分
        self.image_weights = image_weights
        self.rect = False if image_weights else rect # 是否启动矩形训练 一般训练时关闭 验证时打开 可以加速
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # mosaic增强的边界值  [-320, -320]
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride # 最大下采样率 32
        self.path = path  # 图片路径
        self.albumentations = Albumentations() if augment else None

        # 2、得到path路径下的所有图片的路径self.img_files，这里需要自己debug一下 不会太难
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]: # 主要用来处理window与linux的区别
                # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
                # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
                p = Path(p)  # os-agnostic
                # 如果路径path为包含图片的文件夹路径
                if p.is_dir():  # dir
                    # glob.glab: 返回所有匹配的文件路径列表  递归获取p路径下所有文件
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                # 如果路径path为包含图片路径的txt文件
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()  # 获取图片路径，更换相对路径
                        # 获取数据集路径的上级父目录  os.sep为路径里的分隔符（不同路径的分隔符不同，os.sep可以根据系统自适应）
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            # 破折号替换为os.sep，os.path.splitext(x)将文件名与扩展名分开并返回一个列表
            # 筛选f中所有的图片文件
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # 3、根据imgs路径找到labels的路径self.label_files
        self.label_files = img2label_paths(self.img_files)  # labels

        # 4、cache label 下次运行这个脚本的时候直接从cache中取label，而不是去文件中取label 速度更快。
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            # 如果有cache文件，直接加载  exists=True: 是否已从cache文件中读出了nf, nm, ne, nc, n等信息
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            # 如果图片版本信息或者文件列表的hash值对不上号 说明本地数据集图片和label可能发生了变化 就重新cache label文件
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except Exception:
            # 否则调用cache_labels缓存标签及标签相关信息
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # 打印cache的结果 nf nm ne nc n = 找到的标签数量，漏掉的标签数量，空的标签数量，损坏的标签数量，总的标签数量
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        # 如果已经从cache文件读出了nf nm ne nc n等信息，直接显示标签信息  msgs信息等
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        # 数据集没有标签信息 就发出警告并显示标签label下载地址help_url
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # 5、Read cache  从cache中读出最新变量赋给self  方便给forward中使用
        # cache中的键值对最初有: cache[img_file]=[l, shape, segments] cache[hash] cache[results] cache[msg] cache[version]
        # 先从cache中去除cache文件中其他无关键值如:'hash', 'version', 'msgs'等都删除
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items

        # pop掉results、hash、version、msgs后只剩下cache[img_file]=[l, shape, segments]
        #   cache.values(): 取cache中所有值 对应所有l, shape, segments
        # labels: 如果数据集所有图片中没有一个多边形label  labels存储的label就都是原始label(都是正常的矩形label)
        #         否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
        # shapes: 所有图片的shape
        # self.segments: 如果数据集所有图片中没有一个多边形label  self.segments=None
        #                否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
        # zip 是因为cache中所有labels、shapes、segments信息都是按每张img分开存储的, zip是将所有图片对应的信息叠在一起
        labels, shapes, self.segments = zip(*cache.values())    # segments: 都是[]
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)    # image shapes to float64
        self.img_files = list(cache.keys())  # 更新所有图片的img_files信息 update img_files from cache result
        self.label_files = img2label_paths(cache.keys())  # 更新所有图片的label_files信息(因为img_files信息可能发生了变化)
        n = len(shapes)  # 当前数据集图片
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # 得到batch序列
        nb = bi[-1] + 1  # 得到batch的数量
        self.batch = bi  # batch index of image
        self.n = n  # number of images
        self.indices = range(n) # 所有图片的index

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # 6、为Rectangular Training作准备：即减少大小不同图片处理时，对于多余的黑边做到最小，实现降低计算量
        # 这里主要是注意shapes的生成 这一步很重要 因为如果采样矩形训练那么整个batch的形状要一样 就要计算这个符合整个batch的shape
        # 而且还要对数据集按照高宽比进行排序 这样才能保证同一个batch的图片的形状差不多相同 再选择一个共同的shape代价也比较小
        if self.rect:
            #  所有训练图片的shape
            s = self.shapes  # wh
            # 计算高宽比
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()    # 根据高宽比排序
            self.img_files = [self.img_files[i] for i in irect] # 获取排序后的img_files
            self.label_files = [self.label_files[i] for i in irect]  # 获取排序后的label_files
            self.labels = [self.labels[i] for i in irect]   # 获取排序后的labels
            self.shapes = s[irect]   # 获取排序后的wh
            ar = ar[irect]   # 获取排序后的wh

            # 计算每个batch采用的统一尺度 Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                # 同一个batch的图片提取出来
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()   # 获取第i个batch中，最小和最大高宽比
                if maxi < 1:
                    # [H,W]如果高/宽小于1(w > h)，宽大于高，矮胖型，(img_size*maxi,img_size)（保证原图像尺度不变进行缩放）
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    # [H,W]如果高/宽大于1(w < h)，宽小于高，瘦高型，(img_size,img_size *（1/mini）)（保证原图像尺度不变进行缩放）
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            # 要求每个batch_shapes的高宽都是32的整数倍，所以要先除以32，取整再乘以32（不过img_size如果是32倍数这里就没必要了）
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # 7、是否需要cache image 一般是False 因为RAM会不足  cache label还可以 但是cache image就太大了 所以一般不用
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(self.load_image, range(n))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:  # 'ram'
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """
                用在__init__函数中  cache数据集label

                这个函数用于加载文件路径中的label信息生成cache文件。
                cache文件中包括的信息有：im_file, l, shape, segments, hash, results, msgs, version等，具体看代码注释。

               加载label信息生成cache文件   Cache dataset labels, check images and read shapes

               :params path: cache文件保存地址
               :params prefix: 日志头部信息(彩打高亮部分)
               :return x: cache中保存的字典
                      包括的信息有: x[im_file] = [l, shape, segments]
                                 一张图片一个label相对应的保存到x, 最终x会保存所有图片的相对路径、gt框的信息、形状shape、所有的多边形gt信息
                                     im_file: 当前这张图片的path相对路径
                                     l: 当前这张图片的所有gt框的label信息(不包含segment多边形标签) [gt_num, cls+xywh(normalized)]
                                     shape: 当前这张图片的形状 shape
                                     segments: 当前这张图片所有gt的label信息(包含segment多边形标签) [gt_num, xy1...]
                                  hash: 当前图片和label文件的hash值  1
                                  results: 找到的label个数nf, 丢失label个数nm, 空label个数ne, 破损label个数nc, 总img/label个数len(self.img_files)
                                  msgs: 所有数据集的msgs信息
                                  version: 当前cache version
               """
        x = {}  # 初始化最终cache中保存的字典dict
        # 初始化number missing, found, empty, corrupt, messages
        # 初始化整个数据集: 漏掉的标签(label)总数量, 找到的标签(label)总数量, 空的标签(label)总数量, 错误标签(label)总数量, 所有错误信息
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..." # 日志
        # 多进程调用verify_image_label函数
        with Pool(NUM_THREADS) as pool:
            # 定义pbar进度条
            # pool.imap_unordered: 对大量数据遍历多进程计算 返回一个迭代器
            # 把self.img_files, self.label_files, repeat(prefix) list中的值作为参数依次送入(一次送一个)verify_image_label函数
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            # im_file: 当前这张图片的path相对路径
            # l: [gt_num, cls+xywh(normalized)]
            #    如果这张图片没有一个segment多边形标签 l就存储原label(全部是正常矩形标签)
            #    如果这张图片有一个segment多边形标签  l就存储经过segments2boxes处理好的标签(正常矩形标签不处理 多边形标签转化为矩形标签)
            # shape: 当前这张图片的形状 shape
            # segments: 如果这张图片没有一个segment多边形标签 存储None
            #           如果这张图片有一个segment多边形标签 就把这张图片的所有label存储到segments中(若干个正常gt 若干个多边形标签) [gt_num, xy1...]
            # nm_f(nm): number missing 当前这张图片的label是否丢失         丢失=1    存在=0
            # nf_f(nf): number found 当前这张图片的label是否存在           存在=1    丢失=0
            # ne_f(ne): number empty 当前这张图片的label是否是空的         空的=1    没空=0
            # nc_f(nc): number corrupt 当前这张图片的label文件是否是破损的  破损的=1  没破损=0
            # msg: 返回的msg信息  label文件完好=‘’  label文件破损=warning信息
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f  # 累加总number missing label
                nf += nf_f  # 累加总number found label
                ne += ne_f  # 累加总number empty label
                nc += nc_f  # 累加总number corrupt label
                if im_file:
                    x[im_file] = [lb, shape, segments]  # 信息存入字典 key=im_file  value=[l, shape, segments]
                if msg:
                    msgs.append(msg)    # 将msg加入总msg
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()    # 关闭进度条

        # 日志打印所有msg信息
        if msgs:
            LOGGER.info('\n'.join(msgs))
        # 一张label都没找到 日志打印help_url下载地址
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files) # 将当前图片和label文件的hash值存入最终字典dist
        x['results'] = nf, nm, ne, nc, len(self.img_files)  # 将nf, nm, ne, nc, len(self.img_files)存入最终字典dist
        x['msgs'] = msgs  # 将所有数据集的msgs信息存入最终字典dist
        x['version'] = self.cache_version  # 将当前cache version存入最终字典dist
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        # 这个函数是求数据集图片的数量。
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        """
               这部分是数据增强函数，一般一次性执行batch_size次。
               训练 数据增强: mosaic(random_perspective) + hsv + 上下左右翻转
               测试 数据增强: letterbox
               :return torch.from_numpy(img): 这个index的图片数据(增强后) [3, 640, 640]
               :return labels_out: 这个index图片的gt label [6, 6] = [gt_num, 0+class+xywh(normalized)]
               :return self.img_files[index]: 这个index图片的路径地址
               :return shapes: 这个batch的图片的shapes 测试时(矩形训练)才有  验证时为None   for COCO mAP rescaling
               """
        # 这里可以通过三种形式获取要进行数据增强的图片index  linear, shuffled, or image_weights
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp  # 超参 包含众多数据增强超参
        mosaic = self.mosaic and random.random() < hyp['mosaic']    # 是否使用马赛克处理
        # mosaic增强 对图像进行4张图拼接训练  一般训练时运行
        # mosaic + MixUp
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)   # 将四张图片拼接在一张马赛克图像中
            # img, labels = load_mosaic9(self, index)
            shapes = None

            # MixUp augmentation
            # mixup数据增强
            if random.random() < hyp['mixup']: # hyp['mixup']=0 默认为0则关闭 默认为1则100%打开
                # *load_mosaic(self, random.randint(0, self.n - 1)) 随机从数据集中任选一张图片和本张图片进行mixup数据增强
                # img:   两张图片融合之后的图片 numpy (640, 640, 3)
                # labels: 两张图片融合之后的标签label [M+N, cls+x1y1x2y2]
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
                # 测试代码 测试MixUp效果
                # cv2.imshow("MixUp", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(img.shape)   # (640, 640, 3)

                # 否则: 载入图片 + Letterbox  (val)
        else:
            # Load image
            # 载入图片  载入图片后还会进行一次resize  将当前图片的最长边缩放到指定的大小(512), 较小边同比例缩放
            # load image img=(343, 512, 3)=(h, w, c)  (h0, w0)=(335, 500)  numpy  index=4
            # img: resize后的图片   (h0, w0): 原始图片的hw  (h, w): resize后的图片的hw
            # 这一步是将(335, 500, 3) resize-> (343, 512, 3)
            img, (h0, w0), (h, w) = self.load_image(index)

            # 测试代码 测试load_image效果
            # cv2.imshow("load_image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            # Letterbox
            # letterbox之前确定这张当前图片letterbox之后的shape  如果不用self.rect矩形训练shape就是self.img_size
            # 如果使用self.rect矩形训练shape就是当前batch的shape 因为矩形训练的话我们整个batch的shape必须统一(在__init__函数第6节内容)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # letterbox 这一步将第一步缩放得到的图片再缩放到当前batch所需要的尺度 (343, 512, 3) pad-> (384, 512, 3)
            # (矩形推理需要一个batch的所有图片的shape必须相同，而这个shape在init函数中保持在self.batch_shapes中)
            # 这里没有缩放操作，所以这里的ratio永远都是(1.0, 1.0)  pad=(0.0, 20.5)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # 图片letterbox之后label的坐标也要相应变化  根据pad调整label坐标 并将归一化的xywh -> 未归一化的xyxy
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            # 测试代码 测试letterbox效果
            # cv2.imshow("letterbox", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)   # (640, 640, 3)

            if self.augment:
                # 不做mosaic的话就要做random_perspective增强 因为mosaic函数内部执行了random_perspective增强
                # random_perspective增强: 随机对图片进行旋转，平移，缩放，裁剪，透视变换
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # 色域空间增强Augment colorspace：H色调、S饱和度、V亮度
            # 通过一些随机值改变hsv，实现数据增强
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # 随机上下翻转 flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # 随机左右翻转 flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        # 6个值的tensor 初始化标签框对应的图片序号, 配合下面的collate_fn使用
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def load_image(self, i):
        """
            被LoadImagesAndLabels模块的__getitem__函数和load_mosaic模块中载入对应index的图片时调用

            这个函数是根据图片index，从self或者从对应图片路径中载入对应index的图片 并将原图中hw中较大者扩展到self.img_size, 较小者同比例扩展。
        """

        """
            loads 1 image from dataset, returns img, original hw, resized hw
            
            :params self: 一般是导入LoadImagesAndLabels中的self
            :param i: 当前图片的index
            :return: img: resize后的图片
                    (h0, w0): hw_original  原图的hw
                    img.shape[:2]: hw_resized resize后的图片hw(hw中较大者扩展到self.img_size, 较小者同比例扩展)
            """

        # 按index从self.imgs中载入当前图片, 但是由于缓存的内容一般会不够, 所以我们一般不会用self.imgs(cache)保存所有的图片
        im = self.imgs[i]
        # 图片是空的话, 就从对应文件路径读出这张图片
        if im is None:  # not cached 一般都不会使用cache缓存到self.imgs中
            npy = self.img_npy[i]
            if npy and npy.exists():  # load npy
                im = np.load(npy)
            else:  # read image
                f = self.img_files[i]
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'

            h0, w0 = im.shape[:2]  # orig hw
            # img_size 设置的是预处理后输出的图片尺寸   r=缩放比例
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                # cv2.INTER_AREA: 基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法, 它可以产生更少的波纹
                # cv2.INTER_LINEAR: 双线性插值,默认情况下使用该方式进行插值   根据ratio选择不同的插值方式
                # 将原图中hw中较大者扩展到self.img_size, 较小者同比例扩展
                im = cv2.resize(im,
                                (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized

    def load_mosaic(self, index):

        """
            这个模块就是很有名的mosaic增强模块，几乎训练的时候都会用它，可以显著的提高小样本的mAP。
            代码是数据增强里面最难的, 也是最有价值的，mosaic是非常非常有用的数据增强trick, 一定要熟练掌握。
        """

        """
            用在LoadImagesAndLabels模块的__getitem__函数 进行mosaic数据增强
            将四张图片拼接在一张马赛克图像中  loads images in a 4-mosaic
            :param index: 需要获取的图像索引
            :return: img4: mosaic和随机透视变换后的一张图片  numpy(640, 640, 3)
                     labels4: img4对应的target  [M, cls+x1y1x2y2]
        """
        # labels4: 用于存放拼接图像（4张图拼成一张）的label信息(不包含segments多边形)
        # segments4: 用于存放拼接图像（4张图拼成一张）的label信息(包含segments多边形)
        labels4, segments4 = [], []
        s = self.img_size # 一般的图片大小
        # 随机初始化拼接图像的中心点坐标  [0, s*2]之间随机取2个数作为拼接图像的中心坐标
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        # 从dataset中随机寻找额外的三张图像进行拼接 [14, 26, 2, 16] 再随机选三张图片的index
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        # 遍历四张图像进行拼接 4张不同大小的图像 => 1张[1472, 1472, 3]的图像
        for i, index in enumerate(indices):
            # load image   每次拿一张图片 并将这张图片resize到self.size(h,w)
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left  原图[375, 500, 3] load_image->[552, 736, 3]   hwc
                # 创建马赛克图像 [1472, 1472, 3]=[h, w, c]
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)   w=736  h = 552  马赛克图像：(x1a,y1a)左上角 (x2a,y2a)右下角
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)  图像：(x1b,y1b)左上角 (x2b,y2b)右下角
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # 将截取的图像区域填充到马赛克图像的相应位置   img4[h, w, c]
            # 将图像img的【(x1b,y1b)左上角 (x2b,y2b)右下角】区域截取出来填充到马赛克图像的【(x1a,y1a)左上角 (x2a,y2a)右下角】区域
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # 计算pad(当前图像边界与马赛克边界的距离，越界的情况padw/padh为负值)  用于后面的label映射
            padw = x1a - x1b    # 当前图像与马赛克图像在w维度上相差多少
            padh = y1a - y1b    # 当前图像与马赛克图像在h维度上相差多少

            # labels: 获取对应拼接图像的所有正常label信息(如果有segments多边形会被转化为矩形label)
            # segments: 获取对应拼接图像的所有不正常label信息(包含segments多边形也包含正常gt)
            # 在新图中更新坐标值
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # normalized xywh normalized to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels) # 更新labels4
            segments4.extend(segments) # 更新segments4

        # Concat/clip labels4 把labels4（[(2, 5), (1, 5), (3, 5), (1, 5)] => (7, 5)）压缩到一起
        labels4 = np.concatenate(labels4, 0)
        # 防止越界  label[:, 1:]中的所有元素的值（位置信息）必须在[0, 2*s]之间,小于0就令其等于0,大于2*s就等于2*s   out: 返回
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # 测试代码  测试前面的mosaic效果
        # cv2.imshow("mosaic", img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img4.shape)   # (1280, 1280, 3)

        # 随机偏移标签中心，生成新的标签与原标签结合 replicate
        # img4, labels4 = replicate(img4, labels4)
        #
        # # 测试代码  测试replicate效果
        # cv2.imshow("replicate", img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img4.shape)   # (1280, 1280, 3)

        # Augment
        # random_perspective Augment  随机透视变换 [1280, 1280, 3] => [640, 640, 3]
        # 对mosaic整合后的图片进行随机旋转、平移、缩放、裁剪，透视变换，并resize为输入大小img_size
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=self.hyp['degrees'], # 旋转
                                           translate=self.hyp['translate'], # 平移
                                           scale=self.hyp['scale'], # 缩放
                                           shear=self.hyp['shear'], # 错切/非垂直投影
                                           perspective=self.hyp['perspective'], # 透视变换
                                           border=self.mosaic_border)  # border to remove
        # 测试代码 测试mosaic + random_perspective随机仿射变换效果
        # cv2.imshow("random_perspective", img4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img4.shape)   # (640, 640, 3)

        return img4, labels4

    def load_mosaic9(self, index):
        """
            作者的实验模块，将九张图片拼接在一张马赛克图像中。
        """
        """
            用在LoadImagesAndLabels模块的__getitem__函数 替换mosaic数据增强
            将九张图片拼接在一张马赛克图像中  loads images in a 9-mosaic
            :param self:
            :param index: 需要获取的图像索引
            :return: img9: mosaic和仿射增强后的一张图片
                     labels9: img9对应的target
            """
        # labels9: 用于存放拼接图像（9张图拼成一张）的label信息(不包含segments多边形)
        # segments9: 用于存放拼接图像（9张图拼成一张）的label信息(包含segments多边形)
        labels9, segments9 = [], []
        s = self.img_size # 一般的图片大小(也是最终输出的图片大小)
        # 从dataset中随机寻找额外的三张图像进行拼接 [14, 26, 2, 16] 再随机选三张图片的index
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image  每次拿一张图片 并将这张图片resize到self.size(h,w)
            img, _, (h, w) = self.load_image(index)

            # 这里和上面load_mosaic函数的操作类似 就是将取出的img图片嵌到img9中(不是真的嵌入 而是找到对应的位置)
            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # 和上面load_mosaic函数的操作类似 找到mosaic9增强后的labels9和segments9
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # 生成对应的img9图片(将对应位置的图片嵌入img9中)
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment 同样进行 随机透视变换
        img9, labels9 = random_perspective(img9, labels9, segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        # 注意：这个函数一般是当调用了batch_size次getitem函数后才会调用一次这个函数，对batch_size张图片和对应的label进行打包。
        # 强烈建议这里大家debug试试这里return的数据是不是我说的这样定义的。
        """
            很多人以为写完 init 和 getitem 函数数据增强就做完了，
            我们在分类任务中的确写完这两个函数就可以了，因为系统中是给我们写好了一个collate_fn函数的，

            但是在目标检测中我们却需要重写collate_fn函数，下面我会仔细的讲解这样做的原因（代码中注释）。

            这个函数会在create_dataloader中生成dataloader时调用：
        """

        """这个函数会在create_dataloader中生成dataloader时调用：
               整理函数  将image和label整合到一起
               :return torch.stack(img, 0): 如[16, 3, 640, 640] 整个batch的图片
               :return torch.cat(label, 0): 如[15, 6] [num_target, img_index+class_index+xywh(normalized)] 整个batch的label
               :return path: 整个batch所有图片的路径
               :return shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
               pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包 通过重写此函数实现标签与图片对应的划分，一个batch中哪些标签属于哪一张图片,形如
                   [[0, 6, 0.5, 0.5, 0.26, 0.35],
                    [0, 6, 0.5, 0.5, 0.26, 0.35],
                    [1, 6, 0.5, 0.5, 0.26, 0.35],
                    [2, 6, 0.5, 0.5, 0.26, 0.35],]
                  前两行标签属于第一张图片, 第三行属于第二张。。。
       """

        # img: 一个tuple 由batch_size个tensor组成 整个batch中每个tensor表示一张图片
        # label: 一个tuple 由batch_size个tensor组成 每个tensor存放一张图片的所有的target信息
        #        label[6, object_num] 6中的第一个数代表一个batch中的第几张图
        # path: 一个tuple 由4个str组成, 每个str对应一张图片的地址信息

        img, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()

        # 返回的
        #      img=[batch_size, 3, 736, 736]
        #      torch.stack(img, 0): 将batch_size个[3, 736, 736]的矩阵拼成一个[batch_size, 3, 736, 736]
        #      label=[target_sums, 6]  6：表示当前target属于哪一张图+class+x+y+w+h
        #      torch.cat(label, 0): 将[n1,6]、[n2,6]、[n3,6]...拼接成[n1+n2+n3+..., 6]
        # 这里之所以拼接的方式不同是因为img拼接的时候它的每个部分的形状是相同的，都是[3, 736, 736]
        # 但label的每个部分的形状是不一定相同的，每张图的目标个数是不一定相同的（label肯定也希望用stack,更方便,但是不能那样拼）
        # 如果每张图的目标个数是相同的，那我们就可能不需要重写collate_fn函数了
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """
                同样在create_dataloader中生成dataloader时调用：
                这里是yolo-v5作者实验性的一个代码 quad-collate function
                        当train.py的opt参数quad=True 则调用collate_fn4代替collate_fn
                作用:  如之前用collate_fn可以返回图片[16, 3, 640, 640]，但是经过collate_fn4则返回图片[4, 3, 1280, 1280]
                      将4张mosaic图片[1, 3, 640, 640]合成一张大的mosaic图片[1, 3, 1280, 1280]
                      将一个batch的图片每四张处理, 0.5的概率将四张图片拼接到一张大图上训练, 0.5概率直接将某张图片上采样两倍训练
                """

        # img: 整个batch的图片 [16, 3, 640, 640]
        # label: 整个batch的label标签 [num_target, img_index+class_index+xywh(normalized)]
        # path: 整个batch所有图片的路径
        # shapes: (h0, w0), ((h / h0, w / w0), pad)    for COCO mAP rescaling
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4 # collate_fn4处理后这个batch中图片的个数
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n] # 初始化

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4  # 采样 [0, 4, 8, 16]
            if random.random() < 0.5:
                # 随机数小于0.5就直接将某张图片上采样两倍训练
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                lb = label[i]
            else:
                # 随机数大于0.5就将四张图片(mosaic后的)拼接到一张大图上训练
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(lb)

        # 后面返回的部分和collate_fn就差不多了 原因和解释都写在上一个函数了 自己debug看一下吧
        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------

# 两个文件操作
def create_folder(path='./new'):
    """
        create_folder函数用于创建一个新的文件夹。会用在下面的flatten_recursive函数中。
        用在flatten_recursive函数中
        创建文件夹  Create folder
        """
    # 如果path存在文件夹，则移除
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    # 再从新新建这个文件夹
    os.makedirs(path)  # make new output folder


def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    """
        这个模块是将一个文件路径中的所有文件复制到另一个文件夹中 即将image文件和label文件放到一个新文件夹中。
    """
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(str(path) + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        # shutil.copyfile: 复制文件到另一个文件夹中
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.datasets import *; extract_boxes()

    """
        自行使用 生成分类数据集
        这个模块是将目标检测数据集转化为分类数据集
        集体做法: 把目标检测数据集中的每一个gt拆解开 分类别存储到对应的文件当中。
    """

    """
        Convert detection dataset into classification dataset, with one directory per class
        使用: from utils.datasets import *; extract_boxes()
        :params path: 数据集地址
    """
    path = Path(path)  # images dir 数据集文件目录 默认'..\datasets\coco128'
    # remove existing path / 'classifier' 文件夹
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*')) # 递归遍历path文件下的'*.*'文件
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS: # 必须得是图片文件
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2] # 得到这张图片h w

            # labels 根据这张图片的路径找到这张图片的label路径
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):  # 遍历每一个gt
                    c = int(x[0])  # class
                    # 生成新'file_name path\classifier\class_index\image_name'
                    # 如: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45\train2017_000000000009_0.jpg'
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    # f.parent: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45'
                    if not f.parent.is_dir():
                        # 每一个类别的第一张照片存进去之前 先创建对应类的文件夹
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]   # box   normalized to 正常大小
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    # 防止b出界 clip boxes outside of image
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
        这个模块是进行自动划分数据集。当使用自己数据集时，可以用这个模块进行自行划分数据集。
    """

    """
        自动将数据集划分为train/val/test并保存 path/autosplit_*.txt files
        Usage: from utils.datasets import *; autosplit()
        :params path: 数据集image位置
        :params weights: 划分权重 默认分别是(0.9, 0.1, 0.0) 对应(train, val, test)
        :params annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    # 获取images中所有的图片 image files only
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # 随机数种子

    # assign each image to a split 根据(train, val, test)权重划分原始图片数据集
    # indices: [n]   0, 1, 2   分别表示数据集中每一张图片属于哪个数据集 分别对应着(train, val, test)
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    """
        用在cache_labels函数中
        这个函数用于检查每一张图片和每一张label文件是否完好。
        图片文件: 检查内容、格式、大小、完整性
        label文件: 检查每个gt必须是矩形(每行都得是5个数class+xywh) + 标签是否全部>=0 + 标签坐标xywh是否归一化 + 标签中是否有重复的坐标
    """

    """
        :params im_file: 数据集中一张图片的path相对路径
        :params lb_file: 数据集中一张图片的label相对路径
        :params prefix: 日志头部信息(彩打高亮部分)
        
        :return im_file: 当前这张图片的path相对路径
        :return l: [gt_num, cls+xywh(normalized)]
                   如果这张图片没有一个segment多边形标签 l就存储原label(全部是正常矩形标签)
                   如果这张图片有一个segment多边形标签  l就存储经过segments2boxes处理好的标签(正常矩形标签不处理 多边形标签转化为矩形标签)
        :return shape: 当前这张图片的形状 shape
        :return segments: 如果这张图片没有一个segment多边形标签 存储None
                          如果这张图片有一个segment多边形标签 就把这张图片的所有label存储到segments中(若干个正常gt 若干个多边形标签) [gt_num, xy1...]
        :return nm: number missing 当前这张图片的label是否丢失         丢失=1    存在=0
        :return nf: number found 当前这张图片的label是否存在           存在=1    丢失=0
        :return ne: number empty 当前这张图片的label是否是空的         空的=1    没空=0
        :return nc: number corrupt 当前这张图片的label文件是否是破损的  破损的=1  没破损=0
        :return msg: 返回的msg信息  label文件完好=‘’  label文件破损=warning信息
        """
    im_file, lb_file, prefix = args
    segments = []  # 存放这张图所有gt框的信息(包含segments多边形: label某一列数大于8)
    nm, nf, ne, nc, msg = 0, 0, 0, 0, '',  # number (missing, found, empty, corrupt), message, segments

    try:
        # 检查这张图片(内容、格式、大小、完整性) verify images
        im = Image.open(im_file) # 打开图片文件
        im.verify()  # PIL verify 检查图片内容和格式是否正常
        shape = exif_size(im)   # 当前图片的大小 image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'  # 图片大小必须大于9个pixels
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}' # 图片格式必须在img_format中
        if im.format.lower() in ('jpg', 'jpeg'): # 检查jpg格式文件
            with open(im_file, 'rb') as f:
                # f.seek: -2 偏移量 向文件头方向中移动的字节数 2 相对位置 从文件尾开始偏移
                f.seek(-2, 2)
                # f.read(): 读取图片文件  指令:\xff\xd9检测整张图片是否完整,如果不完整就返回corrupted JPEG
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file): # 如果这个label路径存在
            nf = 1  # label found
            with open(lb_file) as f: # 读取label文件
                # 读取当前label文件的每一行: 每一行都是当前图片的一个gt
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # any() 函数用于判断给定的可迭代参数 是否全部为False,则返回 False; 如果有一个为 True,则返回True
                # 如果当前图片的label文件某一列数大于8, 则认为label是存在segment的polygon点(多边形)  就不是矩阵 则将label信息存入segment中
                if any([len(x) > 8 for x in lb]):  # is segment
                    # 当前图片中所有gt框的类别
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    # 获得这张图中所有gt框的label信息(包含segment多边形标签)
                    # 因为segment标签可以是不同长度，所以这里segments是一个列表 [gt_num, xy1...(normalized)]
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    # 获得这张图中所有gt框的label信息(不包含segment多边形标签)
                    # segments(多边形) -> bbox(正方形), 得到新标签  [gt_num, cls+xywh(normalized)]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                # 判断标签是否有五列
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                # 判断标签是否全部>=0
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # 判断标签坐标x y w h是否归一化
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                # 判断标签中是否有重复的坐标
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty  l.shape[0] == 0则为空的标签，ne=1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing  不存在标签文件，则nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """
        这个模块是统计数据集的信息返回状态字典。包含: 每个类别的图片数量 + 每个类别的实例数量。
        olov5数据集没有用  自行使用
    """

    """
        Return dataset statistics dictionary with images and instances counts per split per class
        Usage: from utils.datasets import *; dataset_stats('coco128.yaml', verbose=True)
        
        :params path: 数据集信息  data.yaml
        :params autodownload: Attempt to download dataset if not found locally
        :params verbose: print可视化打印
        
        :return stats: 统计的数据集信息 详细介绍看后面
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test': # 分train、val、test统计数据集信息
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset

        # 统计数据集每个图片label中每个类别gt框的个数
        # x: {list: img_num} 每个list[class_num]  每个图片的label中每个类别gt框的个数
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  #   # list to numpy矩阵  [img_num, class_num] shape(128x80)
        # 分别统计train、val、test三个数据集的数据信息
        # 包括: 'image_stats': 字典dict  图片数量total  没有标签的文件个数unlabelled  数据集每个类别的gt个数[80]
        # 'instance_stats': 字典dict  数据集中所有图片的所有gt个数total   数据集中每个类别的gt个数[80]
        # 'labels': 字典dict  key=数据集中每张图片的文件名  value=每张图片对应的label信息 [n, cls+xywh]
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        # print可视化
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
