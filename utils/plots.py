# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
    # Plotting utils 这个脚本都是一些画图工具
"""

"""
    ·注释来源于各位大佬的视频+博客，收集不易，祝你早日出sci！
    ·秉持开源精神！取之于大佬，用之于各位！
    ·@Dragon AI 
"""

import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general import (CONFIG_DIR, FONT, LOGGER, Timeout, check_font, check_requirements, clip_coords,
                           increment_path, is_ascii, is_chinese, try_except, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness

# 设置一些基本的配置  Settings
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11}) # 自定义matplotlib图上字体font大小size=11

# 在PyCharm 页面中控制绘图显示与否
# 如果这句话放在import matplotlib.pyplot as plt之前就算加上plt.show()也不会再屏幕上绘图 放在之后其实没什么用
matplotlib.use('Agg')  # for writing to files only


class Colors:
    """
        函数功能：这是一个颜色类，用于选择相应的颜色，比如画框线的颜色，字体颜色等等。
    """
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 将hex列表中所有hex格式(十六进制)的颜色转换rgb格式的颜色
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        # 颜色个数
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        # 根据输入的index 选择对应的rgb颜色
        c = self.palette[int(i) % self.n]
        # 返回选择的颜色 默认是rgb
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 初始化Colors对象 下面调用colors的时候会调用__call__函数


def check_pil_font(font=FONT, size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # download if missing
        check_font(font)
        try:
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # known issue https://github.com/ultralytics/yolov5/issues/5374


class Annotator:
    if RANK in (-1, 0):
        check_pil_font()  # download TTF if necessary

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                bbox = self.font.getbbox(label)  # 获取文本的边界框
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # 计算文本的宽度和高度
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle((box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1), fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        bbox = self.font.getbbox(text)  # 获取文本的边界框
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # 计算文本的宽度和高度
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/100轮+YOLOV6.1原型结果')):
    """
        功能函数：用来可视化feature map的，可视化feature map(模型任意层都可以用)，而且可以实现可视化网络中任意一层的feature map。
        被调用：在yolo.py的Model类中的forward_once函数中
    """

    """
        :params x: Features map   [bs, channels, height, width]
        :params module_type: Module type
        :params stage: Module stage within model
        :params n: Maximum number of feature maps to plot
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            # torch.chunk: 与torch.cat()原理相反 将tensor x按dim（行或列）分割成channels个tensor块, 返回的是一个元组
            # 将第2个维度(channels)将x分成channels份  每张图有三个block batch张图  blocks=len(blocks)=3*batch
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # 总共可视化的feature map数量
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            LOGGER.info(f'Saving {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


def hist2d(x, y, n=100):
    """
        函数功能：使用numpy工具画出2d直方图。大多数都是调用工具包封装好的2d直方图方法。
        被调用：在plot_evolution函数和plot_test_txt函数中使用。
    """
    """用在plot_evolution
        使用numpy画出2d直方图
        2d histogram used in labels.png and evolve.png
        """
    # xedges: 返回在start=x.min()和stop=x.max()之间返回均匀间隔的n个数据
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    # np.histogram2d: 2d直方图  x: x轴坐标  y: y轴坐标  (xedges, yedges): bins  x, y轴的长条形数目
    # 返回hist: 直方图对象   xedges: x轴对象  yedges: y轴对象
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    # np.clip: 截取函数 令目标内所有数据都属于一个范围 [0, hist.shape[0] - 1] 小于0的等于0 大于同理
    # np.digitize 用于分区
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)    # x轴坐标
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)    # y轴坐标
    return np.log(hist[xidx, yidx])




# ================================================对检测到的目标格式进行处理（output_to_target）然后再将其画框显示在原图上（plot_images）==============================

def output_to_target(output):
    """
        函数功能：这个函数是用于将经过nms后的output [num_obj，x1y1x2y2+conf+cls] -> [num_obj，batch_id+class+xywh+conf]转变predict的格式
                    在画图操作plot_images之前以便在plot_images中进行绘图 + 显示label。
        被调用：用在test.py中进行绘制前3个batch的预测框predictions 因为只有predictions需要修改格式 target是不需要修改格式的
    """
    """
        :params output: list{tensor(8)}分别对应着当前batch的8(batch_size)张图片做完nms后的结果
                        list中每个tensor[n, 6]  n表示当前图片检测到的目标个数  6=x1y1x2y2+conf+cls

        :return np.array(targets): [num_targets, batch_id+class+xywh+conf]  其中num_targets为当前batch中所有检测到目标框的个数
    """
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):   # 对每张图片分别做处理
        for *box, conf, cls in o.cpu().numpy(): # 对每张图片的每个检测到的目标框进行convert格式
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    """
        函数功能：在output_to_target函数之后，绘制一个batch的所有图片的框框（真实框或预测框），
                                    是将一个batch的图片都放在一个大图mosaic上面，放不下删除。
        被调用： 用在test.py、train.py中，针对的也不再是一张图片一个框，而是整个batch中的所有框。
        
    """
    """
        :params images: 当前batch的所有图片  Tensor [batch_size, 3, h, w]  且图片都是归一化后的
        :params targets:  直接来自target: Tensor[num_target, img_index+class+xywh]  [num_target, 6]
                          来自output_to_target: Tensor[num_pred, batch_id+class+xywh+conf] [num_pred, 7]
        :params paths: tuple  当前batch中所有图片的地址
                       如: '..\\datasets\\coco128\\images\\train2017\\000000000315.jpg'
        :params fname: 最终保存的文件路径 + 名字  runs\train\exp8\train_batch2.jpg
        :params names: 传入的类名 从class index可以相应的key值  但是默认是None 只显示class index不显示类名
        :params max_size: 图片的最大尺寸640  如果images有图片的大小(w/h)大于640则需要resize 如果都是小于640则不需要resize
        :params max_subplots: 最大子图个数 16
        :params mosaic: 一张大图  最多可以显示max_subplots张图片  将总多的图片(包括各自的label框框)一起贴在一起显示
                        mosaic每张图片的左上方还会显示当前图片的名字  最好以fname为名保存起来
    """
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()   # tensor -> numpy array
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 反归一化 将归一化后的图片还原  un-normalise
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # # 子图总数  正方形  limit plot images  4 limit plot images
    ns = np.ceil(bs ** 0.5)  # ns=每行/每列最大子图个数  子图总数=ns*ns ceil向上取整  2

    # Build Image
    # np.full 返回一个指定形状、类型和数值的数组
    # shape: (int(ns * h), int(ns * w), 3) (1024, 1024, 3)  填充的值: 255   dtype 填充类型: np.uint8
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    # 对batch内每张图片
    for i, im in enumerate(images):
        # 如果图片要超过max_subplots我们就不管了
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        # (block_x, block_y) 相当于是左上角的左边
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    # Check if we should resize
    # 如果images有图片的大小(w/h)大于640则需要resize 如果都是小于640则不需要resize
    scale = max_size / ns / max(h, w)

    if scale < 1:# 如果scale_factor < 1说明h/w超过max_size 需要resize回来
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
             # 求出属于这张img的target
            ti = targets[targets[:, 0] == i]  # image targets
            # 将这张图片的所有target的xywh -> xyxy
            boxes = xywh2xyxy(ti[:, 2:6]).T
            # 得到这张图片所有target的类别classes
            classes = ti[:, 1].astype('int')
            # 如果image_targets.shape[1] == 6则说明没有置信度信息(此时target实际上是真实框)
            # 如果长度为7则第7个信息就是置信度信息(此时target为预测框信息)
            labels = ti.shape[1] == 6  # labels if no conf column
            # 得到当前这张图的所有target的置信度信息(pred) 如果没有就为空(真实label)
            # check for confidence presence (label vs pred)
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:   # boxes.shape[1]不为空说明这张图有target目标
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    # 因为图片是反归一化的 所以这里boxes也反归一化
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  
                    # 如果scale_factor < 1 说明resize过, 那么boxes也要相应变化
                    # absolute coords need scale if image scales
                    boxes *= scale
            # 上面得到的boxes信息是相对img这张图片的标签信息 因为我们最终是要将img贴到mosaic上 所以还要变换label->mosaic
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            
             # 将当前的图片img的所有标签boxes画到mosaic上
            for j, box in enumerate(boxes.T.tolist()):
                # 遍历每个box
                cls = classes[j]    # 得到这个box的class index
                color = colors(cls) # 得到这个box框线的颜色
                cls = names[cls] if names else cls  # 如果传入类名就显示类名 如果没传入类名就显示class index

                # 如果labels不为空说明是在显示真实target 不需要conf置信度 直接显示label即可
                # 如果conf[j] > 0.25 首先说明是在显示pred 且这个box的conf必须大于0.25 相当于又是一轮nms筛选 显示label + conf
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)  # save

# ================================================================================================================================================

def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    """
        函数功能：用来画出在训练过程中每个epoch的学习率变化情况
        被调用：用在train.py中学习率设置后实现可视化
    """
    """
        :params optimizer: 优化器
        :params scheduler: 策略调整器
        :params epochs: x
        :params save_dir: lr图片 保存地址
    """
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = [] # 存放每个epoch的学习率

    # 从optimizer中取学习率 一个epoch取一个 共取epochs个 每取一次需要使用scheduler.step更新下一个epoch的学习率
    for _ in range(epochs):
        scheduler.step()    # 更新下一个epoch的学习率
        # ptimizer.param_groups[0]['lr']: 取下一个epoch的学习率lr
        y.append(optimizer.param_groups[0]['lr'])

    plt.plot(y, '.-', label='LR') # 没有传入x 默认会传入 0..epochs-1
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200) # 保存
    plt.close()


def plot_val_txt():  # from utils.plots import *; plot_val()
    """
        函数功能：利用test.py中生成的test.txt文件（或者其他的*.txt文件），画出其xy直方图和xy双直方图。
    """
    # Plot val.txt histograms
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    """
        函数功能：利用targets.txt（真实框的xywh）画出其直方图。
    """
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel() # 将多维数组降位一维
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')
        ax[i].legend()  # 显示上行label图例
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_val_study(file='', dir='', x=None):  # from utils.plots import *; plot_val_study()
    # Plot file=study.txt generated by val.py (or plot all study*.txt in dir)
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False  # plot additional results
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt' for x in ['yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(save_dir.glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    f = save_dir / 'study.png'
    print(f'Saving {f}...')
    plt.savefig(f, dpi=300)


@try_except  # known issue https://github.com/ultralytics/yolov5/issues/5395
@Timeout(30)  # known issue https://github.com/ultralytics/yolov5/issues/5611
def plot_labels(labels, names=(), save_dir=Path('')):
    """
        函数功能：根据从datasets中取到的labels，分析其类别分布，画出labels相关直方图信息。最终会生成labels_correlogram.jpg和labels.jpg两张图片。
                    labels_correlogram.jpg：包含所有标签的 x，y，w，h的多变量联合分布直方图：查看两个或两个以上变量之间两两相互关系的可视化形式（里面包含x、y、w、h两两之间的分布直方图）。
                    labels.jpg：包含ax[0]画出classes的各个类的分布直方图，ax[1]画出所有的真实框；ax[2]画出xy直方图；ax[3]画出wh直方图。
        被调用：通常会用在train.py的载入数据datasets和labels后，统计分析labels相关分布信息。
    """
    """
       :params labels: 数据集的全部真实框标签  (num_targets, class+xywh)  (929, 5)
       :params names: 数据集的所有类别名
       :params save_dir: runs\train\exp21
   """
    # plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    # c: classes (929)    b: boxes  xywh (4, 929)   .transpose() 将(4, 929) -> (929, 4)
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # 类别总数 number of classes  80
    # pd.DataFrame: 创建DataFrame, 类似于一种excel, 表头是['x', 'y', 'width', 'height']  表格数据: b中数据按行依次存储
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # 1、画出labels的 xywh 各自联合分布直方图  labels_correlogram.jpg
    # seaborn correlogram  seaborn.pairplot  多变量联合分布图: 查看两个或两个以上变量之间两两相互关系的可视化形式
    # data: 联合分布数据x   diag_kind:表示联合分布图中对角线图的类型   kind:表示联合分布图中非对角线图的类型
    # corner: True 表示只显示左下侧 因为左下和右上是重复的   plot_kws,diag_kws: 可以接受字典的参数，对图形进行微调
    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)   # 保存labels_correlogram.jpg
    plt.close()

    # 2、画出classes的各个类的分布直方图ax[0], 画出所有的真实框ax[1], 画出xy直方图ax[2], 画出wh直方图ax[3] labels.jpg
    # matplotlib labels
    matplotlib.use('svg')  # faster
    # 将整个figure分成2*2四个区域
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    # 第一个区域ax[1]画出classes的分布直方图
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    try:  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    except Exception:
        pass
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30: # 小于30个类别就把所有的类别名作为横坐标
        ax[0].set_xticks(range(len(names))) # 设置刻度
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)  # 旋转90度 设置每个刻度标签
    else:
        ax[0].set_xlabel('classes') # 如果类别数大于30个, 可能就放不下去了, 所以只显示x轴label
    # 第三个区域ax[2]画出xy直方图     第四个区域ax[3]画出wh直方图
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    # 第二个区域ax[1]画出所有的真实框
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000 # xyxy
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)   # 初始化一个窗口
    for cls, *box in labels[:1000]: # 把所有的框画在img窗口中
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')   # 不要xy轴

    # 去掉上下左右坐标系(去掉上下左右边框)
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    print(f'Best results from row {j} of {evolve_csv}:')
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    f = evolve_csv.with_suffix('.png')  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')

# ==================================================================这两个函数都是用来对result.txt中的各项指标进行可视化=================================================================

def plot_results(file='path/to/results.csv', dir=''):
    """
        函数功能：这个函数是将训练后的结果results.csv中相关的训练指标标画在折线图上（共10个折线图）
                results.csv中一行的元素分别有：当前epoch/总epochs-1 、当前的显存容量mem、box回归损失、obj置信度损失、cls分类损失、训练总损失、真实目标数量num_target、图片尺寸img_shape、Precision、Recall、map@0.5、map@0.5:0.95、测试box回归损失、测试obj置信度损、测试cls分类损失。
                results.csv中画出的指标有：训练回归损失Box、训练置信度损失Objectness、训练分类损失Classification、Precision、Recall、验证回归损失 val Box、验证置信度损失val Objectness、验证分类损失val Classification、mAP@0.5、mAP@0.5:0.95。
    """

    """
        :params save_dir: 'runs\train\exp22'
    """
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
     # 建造一个figure 分割成2行5列, 由10个小subplots组成
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()  # 将多维数组降为一维
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'

    # 读取files文件数据进行可视化
    for fi, f in enumerate(files):
        try:
            # files 原始一行: epoch/epochs - 1, memory, Box, Objectness, Classification, sum_loss, targets.shape[0], img_shape, Precision, Recall, map@0.5, map@0.5:0.95, Val Box, Val Objectness, Val Classification
            # 只使用[2, 3, 4, 8, 9, 12, 13, 14, 10, 11]列 (10, 1) 分布对应 =>
            # [Box, Objectness, Classification, Precision, Recall, Val Box, Val Objectness, Val Classification, map@0.5, map@0.5:0.95]
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)   # 画子图
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)   # 保存results.png
    plt.close()

def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    """
        函数功能：为了防止在训练时有些指标非常的抖动，导致画出来很难看。当data值抖动太大, 就取data的平滑曲线。
    """
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter

# =====================================================================================================================================================================




def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    """没用到
        Plot iDetection '*.txt' per-image logs
    """
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print(f'Warning: Plotting error for {f}; {e}')
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
        函数功能：将预测到的目标从原图中扣出来，剪切好并保存，会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面。
        主动调用：xyxy2xywh、xywh2xyxy、clip_coords、increment_path等函数。
        被调用：在detect.py文件中  由opt的save-crop参数控制执不执行
    """

    """
        :params xyxy: 预测到的目标框信息 list 4个tensor x1 y1 x2 y2 左上角 + 右下角
        :params im: 原图片 需要裁剪的框从这个原图上裁剪  nparray  (1080, 810, 3)
        :params file: runs\detect\exp\crops\dog\bus.jpg
        :params gain: 1.02 xyxy缩放因子
        :params pad: xyxy pad一点点边界框 裁剪出来会更好看
        :params square: 是否需要将xyxy放缩成正方形
        :params BGR: 保存的图片是BGR还是RGB
        :params save: 是否要保存剪切的目标框
    """
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:  # 一般不需要rectangle to square
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
        # box wh * gain + pad  box*gain再加点pad 裁剪出来框更好看
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    # 将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    clip_coords(xyxy, im.shape)
    # crop: 剪切的目标框hw
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        # 保存剪切的目标框
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        cv2.imwrite(str(increment_path(file).with_suffix('.jpg')), crop)
    return crop
