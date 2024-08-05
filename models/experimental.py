# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
    Experimental modules
    这个模块大多是在近年来的一些奇思妙想的论文中提出来的，yolov5的作者将这些模块与yolov5相结合，进行尝试。
"""

"""
    ·注释来源于各位大佬的视频+博客，收集不易，祝你早日出sci！
    ·秉持开源精神！取之于大佬，用之于各位！
    ·@Dragon AI 
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download


class CrossConv(nn.Module):
    """
        Ghost Convolution 幻象卷积  轻量化网络卷积模块
        论文: https://arxiv.org/abs/1911.11907
        源码: https://github.com/huawei-noah/ghostnet
        常见的几种使用方式: https://github.com/ultralytics/yolov5/issues/2905
    """
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    """
       加权特征融合: 学习不同输入特征的重要性，对不同输入特征有区分的融合  Weighted sum of 2 or more layers

       思想: 传统的特征融合往往只是简单的feature map叠加/相加 (sum them up), 比如使用concat或者shortcut连接, 而不对同时加进来的
            feature map进行区分。然而,不同的输入feature map具有不同的分辨率, 它们对融合输入feature map的贡献也是不同的, 因此简单
            的对他们进行相加或叠加处理并不是最佳的操作, 所以这里我们提出了一种简单而高效的加权特融合的机制。

       与论文的不同点：这里的权重参数是可以学习的。
       from: https://arxiv.org/abs/1911.09070
   """
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight   # 是否使用加权权重融合
        self.iter = range(n - 1)  # 加权 iter
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True) # 初始化可学习权重

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2   # 得到每一个layer的可学习权重
            for i in self.iter:
                y = y + x[i + 1] * w[i] # 加权特征融合
        else:
            for i in self.iter:
                y = y + x[i + 1]    # 特征融合
        return y


class MixConv2d(nn.Module):
    """
       Mixed Depthwise Conv 混合深度卷积 就是使用不同大小的卷积核对深度卷积的不同channel分组处理
       可以看作是分组深度卷积 + Inception结构的多种卷积核混用
       论文: https://arxiv.org/abs/1907.09595.
       源码: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet.
   """
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """
                :params c1: 输入feature map的通道数
                :params c2: 输出的feature map的通道数（这个函数的关键点就是对c2进行分组）
                :params k: 混合的卷积核大小 其实论文里是[3, 5, 7...]用的比较多的
                :params s: 步长 stride
                :params equal_ch: 通道划分方式 有均等划分和指数划分两种方式  默认是均等划分
        """
        super().__init__()
        n = len(k)  # number of convolutions

        if equal_ch:
            # 均等划分通道
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:
            # 指数划分通道
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # 这里和原论文略有出入，这里加了一个shortcut操作
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class Ensemble(nn.ModuleList):
    """
        模型集成  Ensemble of models
            集成建模是通过使用许多不同的建模算法或使用不同的训练数据集创建多个不同模型来预测结果的过程。
            只要基础模型是多样且独立的，使用集成方法时模型的预测误差就会减小，即在做出预测时寻求群体的智慧。
            即使集成模型在模型中具有多个基础模型（求多个模型的平均值或最大值），它仍作为单个模型运行和执行（最终还是以一个综合模型的取整进行预测）。
        目的: 减少模型的泛化误差
        https://github.com/ultralytics/yolov5/issues/318
        来源: https://www.sciencedirect.com/topics/computer-science/ensemble-modeling
    """
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        # 集成模型为多个模型时, 在每一层forward运算时, 都要运行多个模型在该层的结果append进y中
        for module in self:
            y.append(module(x, augment, profile, visualize)[0]) # 添加module
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # # 将两个模型结果concat 后面做nms(等于翻了一倍的pred) nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    """
        这个函数用于加载模型权重文件并构建模型（可以构造普通模型或者集成模型）。
                用在val.py、detect.py、train.py等文件中  一般用在测试、验证阶段
        Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        :params weights: 模型的权重文件地址 默认weights/yolov5s.pt
                         可以是[a]也可以是list格式[a, b]  如果是list格式将调用上面的模型集成函数 多模型运算 提高最终模型的泛化误差
        :params map_location: attempt_download函数参数  表示模型运行设备device
        :params inplace: pytorch 1.7.0 compatibility设置
    """
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble() # 模型集成函数

    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model->fuse融合->验证模式
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates(关于版本兼容的设置)
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1: # 单个模型 正常返回
        return model[-1]  # return model
    else:   # 多个模型 使用模型集成 并对模型先进行一些必要的设置
        print(f'Ensemble created with {weights}\n')
        # 给每个模型一个name属性
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        # 给每个模型分配stride属性
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble 返回集成模型
