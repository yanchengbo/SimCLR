import json
import os
import shutil

import torch
import yaml


def _yaml_safe_value(value):
    """Convert argparse values to simple YAML-friendly values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_yaml_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _yaml_safe_value(item) for key, item in value.items()}
    return str(value)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Save a model checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def save_config_file(model_checkpoints_folder, args):
    """Save run arguments as a small YAML file."""
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    raw_config = vars(args) if hasattr(args, "__dict__") else dict(args)
    config = {key: _yaml_safe_value(value) for key, value in raw_config.items()}
    with open(os.path.join(model_checkpoints_folder, "config.yml"), "w", encoding="utf-8") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def save_json(payload, filename):
    """Save a dictionary or list as formatted JSON."""
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_history_plot(history, output_path, title, metric_groups):
    """Save simple training or evaluation curves if matplotlib is installed."""
    if not history:
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Warning: matplotlib is not installed, skipping plot generation for {output_path}.")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(len(metric_groups), 1, figsize=(8, 4 * len(metric_groups)), sharex=True)
    if len(metric_groups) == 1:
        axes = [axes]

    for axis, group in zip(axes, metric_groups):
        for metric_name, label in group["series"]:
            axis.plot(epochs, [item[metric_name] for item in history], marker="o", label=label)
        axis.set_ylabel(group["ylabel"])
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Epoch")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def save_simclr_history_plot(history, output_path, title):
    """Save loss, top-k accuracy, and learning-rate curves for SimCLR training."""
    metric_groups = [
        {"ylabel": "Loss", "series": [("train_loss", "Train loss")]},
        {"ylabel": "Accuracy (%)", "series": [("train_top1", "Train top-1"), ("train_top5", "Train top-5")]},
        {"ylabel": "Learning rate", "series": [("learning_rate", "Learning rate")]},
    ]
    return save_history_plot(history, output_path, title, metric_groups)


def save_linear_eval_history_plot(history, output_path, title):
    """Save loss and accuracy curves for linear, fine-tune, or supervised runs."""
    metric_groups = [
        {"ylabel": "Loss", "series": [("train_loss", "Train loss"), ("test_loss", "Test loss")]},
        {
            "ylabel": "Accuracy (%)",
            "series": [
                ("train_top1", "Train top-1"),
                ("test_top1", "Test top-1"),
                ("test_top5", "Test top-5"),
            ],
        },
    ]
    return save_history_plot(history, output_path, title, metric_groups)


def accuracy(output, target, topk=(1,)):
    """Return top-k accuracy values in percent."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 1. Keep the highest scoring class indices for each sample.
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # 2. Compare each prediction with the true label.
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # 3. Count correct predictions for each requested k value.
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
