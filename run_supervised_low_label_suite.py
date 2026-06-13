import argparse
import json
import os
import subprocess
import sys

from utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised baselines for several label fractions")
    parser.add_argument("--data", default="./datasets")
    parser.add_argument("--dataset-name", default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--label-fractions", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--output-dir", default="supervised_low_label_results")
    return parser.parse_args()


def fraction_name(fraction):
    return str(fraction).replace(".", "p")


def main():
    args = parse_args()
    summaries = []

    for fraction in args.label_fractions:
        # 1. Run each fraction in a separate folder so results do not overwrite.
        run_dir = os.path.join(args.output_dir, f"labels_{fraction_name(fraction)}")
        command = [
            sys.executable,
            "supervised_baseline.py",
            "--data",
            args.data,
            "--dataset-name",
            args.dataset_name,
            "--arch",
            args.arch,
            "--label-fraction",
            str(fraction),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--output-dir",
            run_dir,
        ]
        if args.disable_cuda:
            command.append("--disable-cuda")

        print(f"Running supervised baseline with label_fraction={fraction}", flush=True)
        subprocess.run(command, check=True)

        with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as handle:
            result = json.load(handle)
        summaries.append(
            {
                "label_fraction": fraction,
                "train_examples": result["dataset_stats"]["train_examples"],
                "best_test_top1": result["best_epoch"]["test_top1"],
                "best_epoch": result["best_epoch"]["epoch"],
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(
        {
            "method": "supervised_from_scratch",
            "epochs": args.epochs,
            "arch": args.arch,
            "seed": args.seed,
            "results": summaries,
        },
        os.path.join(args.output_dir, "summary.json"),
    )

    for result in summaries:
        print(
            f"labels={result['label_fraction']}: "
            f"Best Test Top1={result['best_test_top1']:.2f} "
            f"at epoch {result['best_epoch']}"
        )


if __name__ == "__main__":
    main()
