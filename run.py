import argparse
from contextlib import nullcontext
import os

import torch
import torch.backends.cudnn as cudnn

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

SUPPORTED_BACKBONES = ["resnet18", "resnet50"]

parser = argparse.ArgumentParser(description="PyTorch SimCLR")
parser.add_argument("-data", "--data", metavar="DIR", default="./datasets", help="path to dataset")
parser.add_argument(
    "-dataset-name",
    "--dataset-name",
    default="stl10",
    help="dataset name",
    choices=["stl10", "cifar10"],
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=SUPPORTED_BACKBONES,
    help="model architecture: " + " | ".join(SUPPORTED_BACKBONES) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=12,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 12)",
)
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument("--lr", "--learning-rate", default=0.0003, type=float, metavar="LR", dest="lr")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training")
parser.add_argument("--disable-cuda", action="store_true", help="run on CPU")
parser.add_argument("--fp16-precision", action="store_true", help="use mixed precision training on GPU")
parser.add_argument("--out_dim", default=128, type=int, help="projection feature dimension (default: 128)")
parser.add_argument(
    "--no-projection-head",
    "--disable-projection-head",
    dest="use_projection_head",
    action="store_false",
    help="disable the two-layer MLP projection head",
)
parser.set_defaults(use_projection_head=True)
parser.add_argument("--log-every-n-steps", default=100, type=int, help="log every n steps")
parser.add_argument("--temperature", default=0.07, type=float, help="softmax temperature (default: 0.07)")
parser.add_argument(
    "--n-views",
    default=2,
    type=int,
    metavar="N",
    help="number of augmented views for contrastive learning",
)
parser.add_argument("--gpu-index", default=0, type=int, help="GPU index")
parser.add_argument(
    "--augmentation",
    default="baseline",
    choices=["baseline", "no_blur", "no_color_jitter", "no_grayscale", "weak", "medium", "strong"],
    help="augmentation setting for ablation experiments",
)
parser.add_argument(
    "--aug-strength",
    default=None,
    choices=["weak", "medium", "strong"],
    help="alias for --augmentation used by Chengbo's branch",
)
parser.add_argument(
    "--experiment-name",
    default=None,
    type=str,
    help="optional experiment name used as the run folder name",
)
parser.add_argument(
    "--output-dir",
    "--run-base-dir",
    default="runs",
    type=str,
    dest="output_dir",
    help="directory used to store run outputs when experiment name is set",
)


def build_run_name(args):
    projection = "proj-on" if args.use_projection_head else "proj-off"
    return "_".join([
        args.dataset_name,
        args.arch,
        args.augmentation,
        projection,
        f"bs-{args.batch_size}",
    ])


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    if args.aug_strength is not None:
        args.augmentation = args.aug_strength

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.run_name = args.experiment_name or build_run_name(args)
    args.run_dir = os.path.join(args.output_dir, args.experiment_name) if args.experiment_name else None

    # 1. Pick CUDA when it is available and the user did not disable it.
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1

    # 2. Build the paired-view dataset and loader for SimCLR pretraining.
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
        drop_last=True,
    )

    # 3. Build the encoder, optimizer, and cosine scheduler.
    model = ResNetSimCLR(
        base_model=args.arch,
        out_dim=args.out_dim,
        use_projection_head=args.use_projection_head,
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=0,
        last_epoch=-1,
    )

    cuda_context = torch.cuda.device(args.gpu_index) if args.gpu_index >= 0 else nullcontext()
    with cuda_context:
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
