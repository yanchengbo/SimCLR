import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

# 获取 torchvision 中所有可用模型名称
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# 命令行参数
parser = argparse.ArgumentParser(description='PyTorch SimCLR') 
# 数据集路径
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
# 数据集名称
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
# backbone 模型
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
# 数据加载线程数
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
# 训练 epoch 数
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
# batch size 大小
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# 学习率
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# 权重衰减 (L2 正则)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# 随机种子
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
# 是否禁用 GPU
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
# 是否使用混合精度 (FP16)
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
# 输出特征维度 (projection head 输出的维度)
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
# 是否使用 projection head
parser.add_argument('--no-projection-head', dest='use_projection_head',
                    action='store_false',
                    help='Disable the two-layer MLP projection head.')
parser.set_defaults(use_projection_head=True)
# 日志打印频率
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
# 温度参数 (对比学习中的超参数)
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
# 每张图生成多少个数据增强视图
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
# GPU 编号
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
# 数据增强消融设置
parser.add_argument('--augmentation', default='baseline',
                    choices=['baseline', 'no_blur', 'no_color_jitter', 'no_grayscale'],
                    help='Augmentation preset for ablation experiments.')
# 输出目录和实验名
parser.add_argument('--experiment-name', default=None, type=str,
                    help='Optional experiment name used as the run folder name.')
parser.add_argument('--output-dir', default='runs', type=str,
                    help='Directory used to store run outputs when experiment name is set.')


def main():
    # SimCLR 必须是 2 个视图
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    if args.experiment_name is not None:
        args.run_dir = os.path.join(args.output_dir, args.experiment_name)
    else:
        args.run_dir = None
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True # 复现
        cudnn.benchmark = True # 加速
    else:
        # cpu
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # 构建数据集
    dataset = ContrastiveLearningDataset(args.data)
    # 训练集 (包含数据增强视图)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.augmentation)
    # 训练集加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    # 定义模型
    model = ResNetSimCLR(base_model=args.arch,
                         out_dim=args.out_dim,
                         use_projection_head=args.use_projection_head)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=len(train_loader), 
                                                           eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        # 开始训练
        simclr = SimCLR(model=model, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
