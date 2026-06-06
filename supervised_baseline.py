import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import accuracy, save_json, save_linear_eval_history_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised-from-scratch baseline for low-label evaluation"
    )
    parser.add_argument("--data", default="./datasets", type=str)
    parser.add_argument("--dataset-name", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--label-fraction", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()
    if not 0 < args.label_fraction <= 1.0:
        parser.error("--label-fraction must be in the range (0, 1].")
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch):
    if arch == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=10)
    return torchvision.models.resnet50(weights=None, num_classes=10)


def get_dataset_targets(dataset):
    if isinstance(dataset, Subset):
        parent_targets = get_dataset_targets(dataset.dataset)
        return [parent_targets[index] for index in dataset.indices]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise AttributeError("Dataset does not expose targets/labels for low-label sampling.")
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    return [int(target) for target in targets]


def build_subset(dataset, label_fraction, seed):
    if label_fraction >= 1.0:
        return dataset

    rng = random.Random(seed)
    class_to_indices = {}
    for index, label in enumerate(get_dataset_targets(dataset)):
        class_to_indices.setdefault(label, []).append(index)

    selected_indices = []
    for label in sorted(class_to_indices):
        indices = class_to_indices[label]
        rng.shuffle(indices)
        count = max(1, int(len(indices) * label_fraction))
        selected_indices.extend(indices[:count])

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def build_dataset(args, train, transform):
    if args.dataset_name == "cifar10":
        return datasets.CIFAR10(args.data, train=train, transform=transform, download=True)

    split = "train" if train else "test"
    return datasets.STL10(args.data, split=split, transform=transform, download=True)


def build_dataloaders(args):
    # Match Chengbo's linear-evaluation input protocol for a controlled comparison.
    transform = transforms.ToTensor()
    full_train_dataset = build_dataset(args, train=True, transform=transform)
    test_dataset = build_dataset(args, train=False, transform=transform)
    train_dataset = build_subset(full_train_dataset, args.label_fraction, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=not args.disable_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=not args.disable_cuda,
    )
    return train_loader, test_loader, {
        "train_examples": len(train_dataset),
        "full_train_examples": len(full_train_dataset),
        "test_examples": len(test_dataset),
    }


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += top1[0].item()
            total_top5 += top5[0].item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "top1": total_top1 / num_batches,
        "top5": total_top5 / num_batches,
    }


def train(model, train_loader, test_loader, args, device):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        total_train_top1 = 0.0
        num_batches = 0

        for images, labels in tqdm(
            train_loader, desc=f"Supervised baseline epoch {epoch + 1}/{args.epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            top1 = accuracy(logits, labels, topk=(1,))[0]
            total_train_loss += loss.item()
            total_train_top1 += top1[0].item()
            num_batches += 1

        test_metrics = evaluate(model, test_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": total_train_loss / num_batches,
            "train_top1": total_train_top1 / num_batches,
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "test_loss": test_metrics["loss"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}\t"
            f"Train Top1 {epoch_metrics['train_top1']:.2f}\t"
            f"Test Top1 {test_metrics['top1']:.2f}\t"
            f"Test Top5 {test_metrics['top5']:.2f}"
        )

    return history


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    train_loader, test_loader, dataset_stats = build_dataloaders(args)
    model = build_model(args.arch).to(device)
    history = train(model, train_loader, test_loader, args, device)
    best_epoch = max(history, key=lambda item: item["test_top1"])

    result = {
        "method": "supervised_from_scratch",
        "dataset_name": args.dataset_name,
        "label_fraction": args.label_fraction,
        "epochs": args.epochs,
        "arch": args.arch,
        "dataset_stats": dataset_stats,
        "optimizer": "Adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "history": history,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(result, os.path.join(args.output_dir, "results.json"))
    save_linear_eval_history_plot(
        history,
        os.path.join(args.output_dir, "supervised_baseline_curves.png"),
        f"Supervised From Scratch ({args.arch}, labels={args.label_fraction})",
    )
    print(f"Best Test Top1: {best_epoch['test_top1']:.2f}")


if __name__ == "__main__":
    main()
