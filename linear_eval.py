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
    parser = argparse.ArgumentParser(description="Linear evaluation for SimCLR checkpoints")
    parser.add_argument("--data", default="./datasets", type=str, help="dataset root")
    parser.add_argument("--dataset-name", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--checkpoint-path", required=True, type=str, help="SimCLR checkpoint path")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--label-fraction", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument(
        "--train-mode",
        default="linear",
        choices=["linear", "finetune"],
        help="linear freezes the encoder and trains only the classifier; finetune updates the full network",
    )
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


def load_encoder_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint["state_dict"]
    encoder_state = {}

    for key, value in checkpoint_state.items():
        if key.startswith("backbone.") and not key.startswith("backbone.fc"):
            encoder_state[key[len("backbone."):]] = value

    load_result = model.load_state_dict(encoder_state, strict=False)
    expected_missing = {"fc.weight", "fc.bias"}
    if set(load_result.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {load_result.unexpected_keys}")

    return checkpoint


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
    dataset_stats = {
        "train_examples": len(train_dataset),
        "full_train_examples": len(full_train_dataset),
        "test_examples": len(test_dataset),
    }
    return train_loader, test_loader, dataset_stats


def configure_trainable_parameters(model, train_mode):
    if train_mode == "linear":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in {"fc.weight", "fc.bias"}
        return model.fc.parameters()

    for parameter in model.parameters():
        parameter.requires_grad = True
    return model.parameters()


def set_model_mode(model, train_mode):
    model.train()
    if train_mode == "linear":
        for module_name, module in model.named_children():
            if module_name != "fc":
                module.eval()


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


def train_classifier(model, train_loader, test_loader, args, device):
    criterion = nn.CrossEntropyLoss().to(device)
    trainable_parameters = configure_trainable_parameters(model, args.train_mode)
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    history = []
    mode_label = "Linear eval" if args.train_mode == "linear" else "Fine-tune eval"

    for epoch in range(args.epochs):
        set_model_mode(model, args.train_mode)
        total_train_loss = 0.0
        total_train_top1 = 0.0
        num_batches = 0

        for images, labels in tqdm(train_loader, desc=f"{mode_label} epoch {epoch + 1}/{args.epochs}"):
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

        train_loss = total_train_loss / num_batches
        train_top1 = total_train_top1 / num_batches
        test_metrics = evaluate(model, test_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "test_loss": test_metrics["loss"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}\t"
            f"Train Top1 {train_top1:.2f}\t"
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
    checkpoint = load_encoder_weights(model, args.checkpoint_path, device)

    history = train_classifier(model, train_loader, test_loader, args, device)
    best_epoch = max(history, key=lambda item: item["test_top1"])

    result = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "dataset_name": args.dataset_name,
        "label_fraction": args.label_fraction,
        "epochs": args.epochs,
        "arch": args.arch,
        "train_mode": args.train_mode,
        "encoder_frozen": args.train_mode == "linear",
        "dataset_stats": dataset_stats,
        "checkpoint_metadata": {
            "use_projection_head": checkpoint.get("use_projection_head"),
            "aug_strength": checkpoint.get("aug_strength"),
            "feature_dim": checkpoint.get("feature_dim"),
            "projection_dim": checkpoint.get("projection_dim"),
        },
        "best_epoch": best_epoch,
        "history": history,
    }

    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.output_dir = os.path.join(
            checkpoint_dir,
            f"{args.train_mode}_eval_{checkpoint_name}_labels_{args.label_fraction}",
        )

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "results.json")
    save_json(result, result_path)

    plot_path = os.path.join(args.output_dir, "linear_eval_curves.png")
    save_linear_eval_history_plot(
        history,
        plot_path,
        f"{args.train_mode.title()} Evaluation ({args.arch}, labels={args.label_fraction})",
    )

    print(f"Best Test Top1: {best_epoch['test_top1']:.2f}")
    print(f"Saved linear evaluation results to {result_path}")
    print(f"Saved linear evaluation curves to {plot_path}")


if __name__ == "__main__":
    main()