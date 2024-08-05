"""

    这个文件是yolov5的训练脚本。
    抓住 数据 + 模型 + 学习率 + 优化器 + 训练这五步即可。
    Train a YOLOv5 model on a custom dataset.
"""



"""
    Usage:
        $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
        $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse # 解析命令行参数模块
import math # 数学公式模块
import random   # 生成随机数模块
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time # 时间模块 更底层
from copy import deepcopy   # 深度拷贝模块
from datetime import datetime
from pathlib import Path

import numpy as np    # numpy数组操作模块
import torch
import torch.distributed as dist    # 分布式训练模块
import torch.nn as nn    # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
import yaml # 操作yaml文件模块
from torch.cuda import amp  # PyTorch amp自动混合精度训练模块
from torch.nn.parallel import DistributedDataParallel as DDP    # 多卡训练模块
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm   # 进度条模块

import os   # 与操作系统进行交互的模块 包含文件路径操作和解析
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 将'F:/yolo_v5/yolov5-U'加入系统的环境变量  该脚本结束后失效
FILE = Path(__file__).resolve() # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\train.py'
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

# pytorch 分布式训练初始化 https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))   # 这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))    # 这个 Worker 是全局第几个 Worker
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))     # 总共有几个 Worker


# path/to/hyp.yaml or hyp dictionary
def train(hyp,opt,device,callbacks):
    """
    :params hyp: data/hyps/hyp.scratch.yaml   hyp dictionary
    :params opt: main中opt参数
    :params device: 当前设备
    """
    # ----------------------------------------------- 初始化参数和配置信息 ----------------------------------------------
    # 初始化pt参数 + 路径信息 + 超参设置保存 + 保存opt + 加载数据配置信息 + 打印日志信息(logger + wandb) + 其他参数(plots、cuda、nc、names、is_coco)
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # # 保存权重的路径 如runs/train/exp18/weights
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last = w / 'last.pt'    # runs/train/exp18/weights/last.pt
    best = w / 'best.pt'    # runs/train/exp18/weights/best.pt

    # Hyperparameters超参数
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 加载hyp超参信息
    # 日志输出超参信息 hyperparameters: ...
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    if not evolve:
        # 保存hyp
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        # 保存opt
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None

    # 数据集参数
    train_path, val_path = data_dict['train'], data_dict['val']

    # nc: number of classes  数据集有多少种类别
    nc = 1 if single_cls else int(data_dict['nc'])
    # names: 数据集所有类别的名字
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # ============================================== 1、model =================================================
    # 载入模型(预训练/不预训练) + 检查数据集 + 设置数据集路径参数(train_path、test_path) + 冻结权重层

    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        # 使用预训练
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(LOCAL_RANK):
            # 这里下载是去google云盘下载, 一般会下载失败,所以建议自行去github中下载再放到weights下
            weights = attempt_download(weights)  # download if not found locally
        # 加载模型及参数
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

        # 这里加载模型有两种方式，一种是通过opt.cfg 另一种是通过ckpt['model'].yaml
        # 区别在于是否使用resume 如果使用resume会将opt.cfg设为空，按照ckpt['model'].yaml来创建模型
        # 这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        # 原因: 保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor会自己覆盖自己设定的anchor
        # 详情参考: https://github.com/ultralytics/yolov5/issues/459
        # 所以下面设置intersect_dicts()就是忽略exclude
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 筛选字典中的键值对  把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 不使用预训练
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # 冻结权重层
    # 这里只是给了冻结权重层的一个例子, 但是作者并不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 当然也会更慢
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False


    # Image size
    # gs: 获取模型最大stride=32   [32 16 8]
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 获取训练图片辨率 imgsz=640 
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # ============================================== 2、优化器 =======================================================================
    # 参数设置(nbs、accumulate、hyp[‘weight_decay’]) + 分组优化(pg0、pg1、pg2) + 选择优化器 + 为三个优化器选择优化方式 + 删除变量
    # nbs 标称的batch_size,模拟的batch_size 比如默认的话上面设置的opt.batch_size=16 -> nbs=64
    # 也就是模型梯度累计 64/16=4(accumulate) 次之后就更新一次模型 等于变相的扩大了batch_size
    # Optimizer
    nbs = 64  # nominal batch size
    # 假设batch_size为16，则意思是迭代64/16=4次batch以后才更新一组参数：变相的增加batch_size，相当于迭代了64个才更新一次参数的更新。
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置超参: 权重衰减参数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}") # 日志

    # 将模型参数分为三组(weights、biases、bn)来进行分组优化
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)   # biases
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight) # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight) # apply decay

    # 选择优化器 并设置g0(bn参数)的优化方式
    if opt.optimizer == 'Adam':
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 设置g1(weights)的优化方式
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # 设置g2(biases)的优化方式
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # 打印log日志 优化信息
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    # 删除三个变量 优化代码
    del g0, g1, g2

    # ============================================== 3、学习率 =================================================
    # 线性学习率 + one cycle学习率 + 实例化 scheduler + 画出学习率变化曲线
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.cos_lr:
        # 使用one cycle 学习率  https://arxiv.org/pdf/1803.09820.pdf
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        # 使用线性学习率
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # 实例化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 画出学习率变化曲线 plot_lr_scheduler(optimizer, scheduler, epochs)

   
    # ---------------------------------------------- 训练前最后准备 ------------------------------------------------------
    # EMA + 使用预训练 + 参数设置(gs、nl、imgsz、imgsz_test) + DP + DDP + SyncBatchNorm

    # EMA
    # 单卡训练: 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # 使用预训练
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch: # 假设你设定的epoch为100次，但是此时加载的模型已经训练150次，则接着再训练100次。如果小于则训练剩余的部分。
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # 是否使用DP mode
    # 如果rank=-1且gpu数量>1，则使用DataParallel单机多卡模式，效果并不好（分布不平均）
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm  是否使用跨卡BN
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # =================================================== 4、数据加载 ===========================================================================================
    # 加载训练集dataloader、dataset + 参数(mlc、nb) + 加载验证集testloader + 如果不使用断点续训，设置labels相关参数(labels、c) ，plots可视化数据集labels信息，检查anchors(k-means + 遗传进化算法)，model半精度
    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect, rank=LOCAL_RANK, workers=workers,
                                              image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    # 获取标签中最大类别值，与类别数作比较，如果小于类别数则表示有问题
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    # TestLoader
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache,
                                       rect=True, rank=-1, workers=workers * 2, pad=0.5,
                                       prefix=colorstr('val: '))[0]
        # 如果不使用断点续训
        if not resume:
            # 统计dataset的label信息
            # [6301, 5] 数据集中有6301个target  [:, class+x+y+w+h]  nparray
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes，将labels从nparray转为tensor格式
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Check
            # 计算默认锚框anchor与数据集标签框的高宽比
            # 标签的高h宽w与anchor的高h_a宽h_b的比值 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
            # 如果bpr小于98%，则根据k-mean算法聚类新的锚框Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end') # ？？？？

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    # ============================================== 5、训练 ===============================================
    """
        设置/初始化一些训练要用的参数(hyp[‘box’]、hyp[‘cls’]、hyp[‘obj’]、hyp[‘label_smoothing’]、model.nc、model.hyp、model.gr、从训练样本标签
                得到类别权重model.class_weights、model.names、热身迭代的次数iterationsnw、last_opt_step、初始化maps和results、学习率衰减所进行到的轮次scheduler.last_epoch + 
                设置amp混合精度训练scaler + 初始化损失函数compute_loss + 打印日志信息) 
    """
    # 设置/初始化一些训练要用的参数（超参数）
    # Model parameters

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # 分类损失系数，scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # 从训练样本标签得到类别权重（和类别中的目标数即类别频率成反比）
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names # 获取类别名

    # Start training
    t0 = time.time()
    # 获取热身迭代的次数iterations
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化maps(每个类别的map)和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练    GradScaler + autocast
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    # 打印日志信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    # 开始训练-----------------------------------------------------------------------------------------------------
    """
         开始训练(注意五点：图片采样策略 + Warmup热身训练 + multi_scale多尺度训练 + amp混合精度训练 + accumulate 梯度更新策略) + 
                打印训练相关信息(包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等 + 
                Plot 前三次迭代的barch的标签框再图片中画出来并保存 + wandb ) + 
                validation(调整学习率、scheduler.step() 、emp val.run()得到results, maps相关信息、
                        将测试结果results写入result.txt中、wandb_logger、Update best mAP 以加权mAP fitness为衡量标准、Save model)
    """
    # start training
    for epoch in range(start_epoch, epochs):  # epoch
        model.train()

        # Update image weights (optional, single-GPU only)
        # 如果为True 进行图片采样策略(按数据集各类别权重采样)
        if opt.image_weights:
            # 从训练(gt)标签获得每个类的权重  标签频率高的类权重低
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # 得到每一张图片对应的采样权重[128]
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            # random.choices: 从range(dataset.n)序列中按照weights(参考每张图片采样权重)进行采样, 一次取一个数字，采样次数为k
            # 最终得到所有图片的采样顺序(参考每张图片采样权重) list [128]
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
            train_loader.sampler.set_epoch(epoch)
        # 进度条，方便展示信息
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            # 创建进度条
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        # 梯度清零
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            # 热身训练（前nw次迭代）热身训练迭代的次数iteration范围[1:nw]  选取较小的accumulate，学习率以及momentum,慢慢的训练
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # bias的学习率从0.1下降到基准学习率lr*lf(epoch) 其他的参数学习率增加到lr*lf(epoch)
                    # lf为上面设置的余弦退火的衰减函数
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 多尺度训练   从[imgsz*0.5, imgsz*1.5+gs]间随机选取一个尺寸(32的倍数)作为当前batch的尺寸送入模型开始训练
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward  混合精度训练 开启autocast的上下文
            with amp.autocast(enabled=cuda):
                # pred: [8, 3, 68, 68, 25] [8, 3, 34, 34, 25] [8, 3, 17, 17, 25]
                # [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，置信度损失和框的回归损失
                # loss为总损失值  loss_items为一个元组，包含分类损失、置信度IOU损失、框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 采用DDP训练 平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.

            # Backward：反向传播 将梯度放大防止梯度的underflow（amp混合精度训练）
            scaler.scale(loss).backward()

            # Optimize
            # 模型反向传播accumulate次（iterations）后，再根据累计的梯度更新一次参数
            if ni - last_opt_step >= accumulate:
                # scaler.step()首先把梯度的值unscale回来
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)  # optimizer.step 参数更新
                # 准备着，看是否要增大scaler
                scaler.update()
                # 梯度清零
                optimizer.zero_grad()
                if ema:
                    # 当前epoch训练结束  更新ema
                    ema.update(model)
                last_opt_step = ni

            # Log
            # 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler  一个epoch训练结束后都要调整学习率（学习率衰减）
        # group中三个学习率（pg0、pg1、pg2）每个都要调整
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # validation
        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 将model中的属性赋值给ema
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # notest: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                # 测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                # results: [1] Precision 所有类别的平均precision(最大f1时)
                #          [1] Recall 所有类别的平均recall
                #          [1] map@0.5 所有类别的平均mAP@0.5
                #          [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                #          [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                # maps: [80] 所有类别的mAP@0.5:0.95
                results, maps, _ = val.run(data_dict,# 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
                                           batch_size=batch_size // WORLD_SIZE * 2, # bs
                                           imgsz=imgsz, # test img size
                                           model=ema.ema,   # ema model
                                           single_cls=single_cls,   # 是否是单类数据集
                                           dataloader=val_loader,   # test dataloader
                                           save_dir=save_dir,   # 保存地址 runs/train/expn
                                           plots=False, # 是否可视化
                                           callbacks=callbacks,
                                           compute_loss=compute_loss) # 损失函数(train)

            # Update best mAP 这里的best mAP其实是[P, R, mAP@.5, mAP@.5-.95]的一个加权值
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            # 保存带checkpoint的模型用于inference或resuming training
            # 保存模型, 还保存了epoch, results, optimizer等信息
            # optimizer将不会在最后一轮完成后保存
            # model保存的是EMA的模型
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    # =================================================================== 打印一些信息================================================================== 
    # 打印一些信息(日志: 打印训练时间、plots可视化训练结果results1.png、confusion_matrix.png 以及(‘F1’, ‘PR’, ‘P’, ‘R’)曲线变化 、日志信息) + coco评价(只在coco数据集才会运行) + 释放显存 return results
    if RANK in [-1, 0]:
        # 日志: 打印训练时间
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=is_coco,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    # 释放显存
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
        函数功能：设置opt参数
    """
    parser = argparse.ArgumentParser()

    # --------------------------------------------------- 常用参数 ---------------------------------------------
    # weights: 权重文件
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    # cfg: 网络模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # data: 实现数据集配置文件 包括path、train、val、test、nc、names、download等
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # hyp: 训练时的超参文件
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # epochs: 训练轮次
    parser.add_argument('--epochs', type=int, default=300)
    # batch-size: 训练批次大小
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # imgsz: 输入网络的图片分辨率大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # nosave: 不保存模型  默认保存  store_true: only test final epoch
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # workers: dataloader中的最大work数（线程个数）
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # device: 训练的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # single-cls: 数据集是否只有一个类别 默认False
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')

    # --------------------------------------------------- 数据增强参数 ---------------------------------------------
    # rect: 训练集是否采用矩形训练  默认False
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # evolve: 是否进行超参进化，使得数值更好 默认False
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # multi-scale: 是否使用多尺度训练 默认False，要被32整除。
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # optimizer: 是否使用优化器 默认使用SGD
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # cache-image: 是否提前缓存图片到内存cache,以加速训练  默认False
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # image-weights: 是否使用图片采用策略(selection img to training by class weights) 默认False 不使用
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')

    # --------------------------------------------------- 其他参数 ---------------------------------------------
    # bucket: 谷歌云盘bucket 一般用不到
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # project: 训练结果保存的根目录 默认是runs/train
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # name: 训练结果保存的目录 默认是exp  最终: runs/train/100轮+YOLOV6.1原型结果
    parser.add_argument('--name', default='100轮+YOLOV6.1原型结果', help='save to project/name')
    # exist-ok: 如果文件存在就ok不存在就新建或increment name  默认False(默认文件都是不存在的)
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # save_period: Log model after every "save_period" epoch    默认-1 不需要log model 信息
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # artifact_alias: which version of dataset artifact to be stripped  默认lastest  貌似没用到这个参数？
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
    # local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # --------------------------------------------------- V6新增 ---------------------------------------------
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')

    # --------------------------------------------------- 三个W&B(wandb)参数 ---------------------------------------------
    # Weights & Biases arguments
    # entity: wandb entity 默认None
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')

    # parser.parse_known_args()
    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # 1、logging和wandb初始化
    # 日志初始化
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        # 检查代码版本是否是最新的  github: ...
        check_git_status()
        # 检查requirements.txt所需包是否都满足 requirements: ...
        check_requirements(exclude=['thop'])

    # 2、使用断点续训 就从last.pt中读取相关参数；不使用断点续训 就从文件中读取相关参数
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        # 使用断点续训 就从last.pt中读取相关参数
        # 如果resume是str，则表示传入的是模型的路径地址
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # 相关的opt参数也要替换成last.pt中的opt参数
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
         # 不使用断点续训 就从文件中读取相关参数
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        # 根据opt.project生成目录  如: runs/train/exp18
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # 3、DDP mode设置
    # 选择设备  cpu/cuda:0
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        # LOCAL_RANK != -1 进行多GPU训练
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        # 根据GPU编号选择设备
        device = torch.device('cuda', LOCAL_RANK)
        # 初始化进程组  distributed backend
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # 4、不使用进化算法 正常Train
    if not opt.evolve:
        # 如果不进行超参进化 那么就直接调用train()函数，开始训练
        train(opt.hyp, opt, device, callbacks)
        # 如果是使用多卡训练, 那么销毁进程组
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # 5、遗传进化算法，边进化边训练
        # 否则使用超参进化算法(遗传算法) 求出最佳超参 再进行训练
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参进化列表 (突变规模, 最小值, 最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 载入初始超参
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml = save_dir / 'hyp_evolve.yaml' # 超参进化后文件保存地址
        evolve_csv =  save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        """
            使用遗传算法进行参数进化 默认是进化300代
            这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
            如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
            有了每个hyp和每个hyp的权重之后有两种进化方式；
                1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
                2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
            evolve.txt会记录每次进化之后的results+hyp
            每次进化时，hyp会根据之前的results进行从大到小的排序，再根据fitness函数计算之前每次进化得到的hyp的权重，再确定哪一种进化方式，从而进行进化。
        """
        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                # 选择超参进化方式 只用single和weighted两种
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.csv
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                # 选取至多前五次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据resluts计算hyp权重
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate 超参进化
                mp, s = 0.8, 0.2  # mutation probability 突变概率, sigma
                npr = np.random
                npr.seed(int(time.time()))
                # 获取突变初始值
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                # [i+7]是因为x中前7个数字为results的指标(P,R,mAP,F1,test_loss=(box,obj,cls)),之后才是超参数hyp
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits 限制超参再规定范围
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # 训练 使用突变后的参超 测试其效果
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()

            # Write mutation results
            # 将结果写入results 并将对应的hyp写到evolve.txt evolve.txt中每一行为一次进化的结果
            # 每行前七个数字 (P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
            # 保存hyp到yaml文件
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    """
        函数功能：使得支持指令执行这个脚本，封装train接口
        Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
