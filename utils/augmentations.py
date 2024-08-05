# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

"""
    ·注释来源于各位大佬的视频+博客，收集不易，祝你早日出sci！
    ·秉持开源精神！取之于大佬，用之于各位！
     @Dragon AI
"""

import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box
from utils.metrics import bbox_ioa


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """
        这个函数是关于图片的色域增强模块，图片并不发生移动，所有不需要改变label，只需要 img 增强即可。
        注意:hsv增强是随机生成各个色域参数的，所以每次增强的效果都是不同的：
    """
    """
        用在LoadImagesAndLabels模块的__getitem__函数
        hsv色域增强  处理图像hsv，不对label进行任何处理
        :param img: 待处理图片  BGR [736, 736]
        :param hgain: h通道色域参数 用于生成新的h通道
        :param sgain: h通道色域参数 用于生成新的s通道
        :param vgain: h通道色域参数 用于生成新的v通道
        :return: 返回hsv增强后的图片 img
    """
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数  用于生成新的hsv通道
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))  # 图像的通道拆分 h s v
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 生成新的h通道
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)    # 生成新的s通道
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)    # 生成新的s通道

        # 图像的通道合并 img_hsv=h+s+v  随机调整hsv之后重新组合hsv通道
        # cv2.LUT(hue, lut_hue)   通道色域变换 输入变换前通道hue 和变换后通道lut_hue
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # no return needed  dst:输出图像
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
     在val时这里主要是做了三件事
     load_image将图片从文件中加载出来，并resize到相应的尺寸（最长边等于我们需要的尺寸，最短边等比例缩放）；
         letterbox将之前resize后的图片再pad到我们所需要的放到dataloader中（collate_fn函数）的尺寸（矩形训练要求同一个batch中的图片的尺寸必须保持一致）；
          将label从相对原图尺寸（原文件中图片尺寸）缩放到相对letterbox pad后的图片尺寸。因为前两部分的图片尺寸发生了变化，同样的我们的label也需要发生相应的变化。
    """

    """
          用在LoadImagesAndLabels模块的__getitem__函数  只在val时才会使用
             将图片缩放调整到指定大小
             Resize and pad image while meeting stride-multiple constraints
             https://github.com/ultralytics/yolov3/issues/232
             :param img: 原图 hwc
             :param new_shape: 缩放后的最长边大小
             :param color: pad的颜色
             :param auto: True 保证缩放后的图片保持原图的比例 即 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放（不会失真）
                          False 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放,最后将较短边两边pad操作缩放到最长边大小（不会失真）
             :param scale_fill: True 简单粗暴的将原图resize到指定的大小 相当于就是resize 没有pad操作（失真）
             :param scale_up: True  对于小于new_shape的原图进行缩放,大于的不变
                              False 对于大于new_shape的原图进行缩放,小于的不变
             :return: img: letterbox后的图片 HWC
                      ratio: wh ratios
                      (dw, dh): w和h的pad
    """
    shape = im.shape[:2]  # 第一层resize后图片大小[h, w] = [343, 512]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 只进行下采样 因为上采样会让图片模糊
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # wh(512, 343) 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle  保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill: # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2  # divide padding into 2 sides  将padding分到上下，左右两侧  dw=0
    dh /= 2  # dh=20.5

    if shape[::-1] != new_unpad:  # resize  将原图resize到new_unpad（长边相同，比例相同的新图）
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) # 计算上下两侧的padding  # top=20 bottom=21
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    # 计算左右两侧的padding  # left=0 right=0
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    """
        这个函数是进行随机透视变换，对mosaic整合后的图片进行随机旋转、缩放、平移、裁剪，透视变换，并resize为输入大小img_size。
    """

    """
        这个函数会用于load_mosaic中用在mosaic操作之后
        随机透视变换  对mosaic整合后的图片进行随机旋转、缩放、平移、裁剪，透视变换，并resize为输入大小img_size
        
        :params img: mosaic整合后的图片img4 [2*img_size, 2*img_size]
        如果mosaic后的图片没有一个多边形标签就使用targets, segments为空  如果有一个多边形标签就使用segments, targets不为空
        :params targets: mosaic整合后图片的所有正常label标签labels4(不正常的会通过segments2boxes将多边形标签转化为正常标签) [N, cls+xyxy]
        :params segments: mosaic整合后图片的所有不正常label信息(包含segments多边形也包含正常gt)  [m, x1y1....]
        :params degrees: 旋转和缩放矩阵参数
        :params translate: 平移矩阵参数
        :params scale: 缩放矩阵参数
        :params shear: 剪切矩阵参数
        :params perspective: 透视变换参数
        :params border: 用于确定最后输出的图片大小 一般等于[-img_size, -img_size] 那么最后输出的图片大小为 [img_size, img_size]
        
        :return img: 通过透视变换/仿射变换后的img [img_size, img_size]
        :return targets: 通过透视变换/仿射变换后的img对应的标签 [n, cls+x1y1x2y2]  (通过筛选后的)
    """

    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    # 设定输出图片的 H W
    # border=-s // 2  所以最后图片的大小直接减半 [img_size, img_size, 3]
    height = im.shape[0] + border[0] * 2 # 最终输出图像的H
    width = im.shape[1] + border[1] * 2  # 最终输出图像的W

    # ============================ 开始变换 =============================
    # 需要注意的是，其实opencv是实现了仿射变换的, 不过我们要先生成仿射变换矩阵M
    # Center 设置中心平移矩阵
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective  设置透视变换矩阵
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale  设置旋转矩阵+缩放矩阵
    R = np.eye(3)   # 初始化R = [[1,0,0], [0,1,0], [0,0,1]]    (3, 3)
    # a: 随机生成旋转角度 范围在(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    # s: 随机生成旋转后图像的缩放比例 范围在(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    # cv2.getRotationMatrix2D: 二维旋转缩放函数
    # 参数 angle:旋转角度  center: 旋转中心(默认就是图像的中心)  scale: 旋转后图像的缩放比例
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear  设置剪切矩阵：错切/非垂直投影
    S = np.eye(3)   # 初始化T = [[1,0,0], [0,1,0], [0,0,1]]
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation 平移
    T = np.eye(3)   # 初始化T = [[1,0,0], [0,1,0], [0,0,1]]    (3, 3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix  @ 表示矩阵乘法  生成仿射变换矩阵M
    # 将所有变换矩阵连乘得到最终的变换矩阵
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # 将仿射变换矩阵M作用在图片上
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            # 透视变换函数  实现旋转平移缩放变换后的平行线不再平行
            # 参数和下面warpAffine类似
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # 仿射变换函数  实现旋转平移缩放变换后的平行线依旧平行
            # image changed  img  [1472, 1472, 3] => [736, 736, 3]
            # cv2.warpAffine: opencv实现的仿射变换函数
            # 参数： img: 需要变化的图像   M: 变换矩阵  dsize: 输出图像的大小  flags: 插值方法的组合（int 类型！）
            #       borderValue: （重点！）边界填充值  默认情况下，它为0。
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    # 同样需要调整标签信息
    n = len(targets)
    if n:
        # 判断是否可以使用segment标签: 只有segments不为空时即数据集中有多边形gt也有正常gt时才能使用segment标签 use_segments=True
        #                          否则如果只有正常gt时segments为空 use_segments=False
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        # 如果使用的是segments标签(标签中含有多边形gt)
        if use_segments:  # warp segments
            # 先对segment标签进行重采样
            # 比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)
            # [n, x1y2...x99y100] 扩增坐标-> [n, 500, 2]
            # 由于有旋转，透视变换等操作，所以需要对多边形所有角点都进行变换
            segments = resample_segments(segments)  # upsample
            # 其中 segment.shape = [n, 2], 表示物体轮廓各个坐标点
            for i, segment in enumerate(segments):  # segment: [500, 2]  多边形的500个点坐标xy
                xy = np.ones((len(segment), 3)) # [1, 1+1+1]
                xy[:, :2] = segment  # [500, 2]
                # 对该标签多边形的所有顶点坐标进行透视/仿射变换
                xy = xy @ M.T  # transform 应用旋转矩阵
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # 根据segment的坐标，取xy坐标的最大最小值，得到边框的坐标  clip
                new[i] = segment2box(xy, width, height)
        # 不使用segments标签 使用正常的矩形的标签targets
        else:
            # warp boxes 如果是box坐标, 这里targets每行为[x1,y1,x2,y2],n为行数,表示目标边框个数：
            # 直接对box透视/仿射变换
            # 由于有旋转，透视变换等操作，所以需要对四个角点都进行变换
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform 应用旋转矩阵， # transform 每个角点的坐标
            # 如果透视变换参数perspective不为0， 就需要做rescale，透视变换参数为0, 则无需做rescale。
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip 将坐标clip到[0, width],[0,height]区间内
            # clip  去除太小的target(target大部分跑到图外去了)
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates 进一步过滤,留下那些w,h>2,宽高比<20,变换后面积比之前比>0.1的那些xy
        # 长和宽必须大于wh_thr个像素 裁剪过小的框(面积小于裁剪前的area_thr)  长宽比范围在(1/ar_thr, ar_thr)之间的限制
        # 筛选结果 [n] 全是True或False   使用比如: box1[i]即可得到i中所有等于True的矩形框 False的矩形框全部删除
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        # 得到所有满足条件的targets
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
        这个函数是进行mixup数据增强：按比例融合两张图片。
    """
    """
        用在LoadImagesAndLabels模块中的__getitem__函数进行mixup增强
       mixup数据增强, 按比例融合两张图片  Applies MixUp augmentation
       论文: https://arxiv.org/pdf/1710.09412.pdf
       :params im:图片1  numpy (640, 640, 3)
       :params labels:[N, 5]=[N, cls+x1y1x2y2]
       :params im2:图片2  (640, 640, 3)
       :params labels2:[M, 5]=[M, cls+x1y1x2y2]
       :return img: 两张图片mixup增强后的图片 (640, 640, 3)
       :return labels: 两张图片mixup增强后的label标签 [M+N, cls+x1y1x2y2]
   """
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    # 随机从beta分布中获取比例,range[0, 1]
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    # 按照比例融合两张图片
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    # 将两张图片标签拼接到一起
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
