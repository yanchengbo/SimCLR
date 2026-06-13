import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear evaluation for frozen SimCLR encoders.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a SimCLR checkpoint.")
    parser.add_argument("--name", required=True,
                        help="Experiment name used in the saved result file.")
    parser.add_argument("--data", default="./datasets_local",
                        help="Dataset root directory.")
    parser.add_argument("--dataset-name", default="cifar10",
                        choices=["cifar10"],
                        help="Dataset used for linear evaluation.")
    parser.add_argument("--arch", default=None,
                        help="Backbone architecture. Defaults to checkpoint arch.")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs for the linear classifier.")
    parser.add_argument("--batch-size", default=256, type=int,
                        help="Batch size for feature extraction and classifier training.")
    parser.add_argument("--workers", default=0, type=int,
                        help="Number of dataloader workers.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate for the linear classifier.")
    parser.add_argument("--weight-decay", default=0.0, type=float,
                        help="Weight decay for the linear classifier.")
    parser.add_argument("--label-fraction", default=1.0, type=float,
                        help="Fraction of CIFAR10 train labels used for linear eval.")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device used for evaluation.")
    parser.add_argument("--output-dir", default="feature_eval/results",
                        help="Directory for JSON result files.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet(arch):
    model_fn = getattr(models, arch)
    try:
        return model_fn(weights=None)
    except TypeError:
        return model_fn(pretrained=False)


def load_frozen_encoder(checkpoint_path, arch, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = arch or checkpoint.get("arch", "resnet18")
    encoder = build_resnet(arch)
    feature_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()

    encoder_state = {}
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("backbone."):
            continue
        key = key[len("backbone."):]
        if key.startswith("fc."):
            continue
        encoder_state[key] = value

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected encoder keys: {}".format(unexpected))
    if missing:
        print("Warning: missing encoder keys: {}".format(missing))

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, feature_dim, arch


def get_cifar10_loaders(data_root, batch_size, workers, label_fraction, seed):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_dataset = datasets.CIFAR10(
        data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(
        data_root, train=False, transform=transform, download=True)

    if label_fraction < 1.0:
        train_dataset = make_stratified_subset(train_dataset, label_fraction, seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def make_stratified_subset(dataset, label_fraction, seed):
    if label_fraction <= 0.0 or label_fraction > 1.0:
        raise ValueError("--label-fraction must be in (0, 1].")

    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    indices = []
    for class_id in sorted(np.unique(targets)):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        n_keep = max(1, int(len(class_indices) * label_fraction))
        indices.extend(class_indices[:n_keep].tolist())
    rng.shuffle(indices)
    return Subset(dataset, indices)


def extract_features(encoder, loader, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            output = encoder(images).cpu()
            features.append(output)
            labels.append(targets.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item() * 100.0


def evaluate(classifier, data_loader, device):
    classifier.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = torch.argmax(classifier(features), dim=1)
            total_correct += (predictions == targets).sum().item()
            total_count += targets.size(0)
    return 100.0 * total_correct / total_count


def train_linear_classifier(train_features, train_labels, test_features,
                            test_labels, feature_dim, args, device):
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    classifier = nn.Linear(feature_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        total_count = 0
        train_correct = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = classifier(features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_count += targets.size(0)
            train_correct += (torch.argmax(logits, dim=1) == targets).sum().item()

        train_loss = total_loss / total_count
        train_acc = 100.0 * train_correct / total_count
        test_acc = evaluate(classifier, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        history.append(row)
        print("epoch {:03d}/{:03d} train_loss {:.4f} train_acc {:.2f} test_acc {:.2f}".format(
            epoch, args.epochs, train_loss, train_acc, test_acc))

    return {
        "final_test_acc": history[-1]["test_acc"],
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    started = time.time()
    encoder, feature_dim, arch = load_frozen_encoder(
        args.checkpoint, args.arch, device)
    train_loader, test_loader = get_cifar10_loaders(
        args.data, args.batch_size, args.workers,
        args.label_fraction, args.seed)

    print("Extracting frozen encoder features on {}...".format(device))
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    print("train features: {}, test features: {}".format(
        tuple(train_features.shape), tuple(test_features.shape)))

    result = train_linear_classifier(
        train_features, train_labels,
        test_features, test_labels,
        feature_dim, args, device)

    result.update({
        "name": args.name,
        "checkpoint": args.checkpoint,
        "arch": arch,
        "feature_dim": feature_dim,
        "dataset_name": args.dataset_name,
        "label_fraction": args.label_fraction,
        "linear_epochs": args.epochs,
        "linear_lr": args.lr,
        "linear_weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    })

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "{}.json".format(args.name))
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved result to {}".format(output_path))
    print("Final test acc: {:.2f}, best test acc: {:.2f} at epoch {}".format(
        result["final_test_acc"], result["best_test_acc"], result["best_epoch"]))


if __name__ == "__main__":
    main()
