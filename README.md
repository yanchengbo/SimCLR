# SimCLR Ablation Study

This repository contains the final code for our COMP3242 SimCLR group project.
The project trains SimCLR on CIFAR-10 and evaluates the learned features with
linear evaluation, fine-tuning, and supervised low-label baselines.

## Main Files

| Path | Purpose |
| --- | --- |
| `run.py` | Train a SimCLR encoder. |
| `simclr.py` | SimCLR training loop and NT-Xent loss. |
| `linear_eval.py` | Run frozen linear evaluation or SimCLR fine-tuning. |
| `supervised_baseline.py` | Train a supervised model from scratch for low-label comparison. |
| `run_supervised_low_label_suite.py` | Run several supervised low-label baselines in one command. |
| `linear_eval_gc.py` | Frozen feature evaluation script used for some ablation runs. |
| `models/resnet_simclr.py` | ResNet-18/ResNet-50 encoder and optional projection head. |
| `data_aug/` | SimCLR augmentation code. |
| `results/` | Saved JSON results for batch size and projection head ablations. |
| `*_figures/` | Figures used in the report. |
| `main.tex` | Final report source. |

## Setup

Install the Python packages:

```bash
pip install -r requirements.txt
```

The scripts can download CIFAR-10 automatically. By default, data is stored in:

```text
./datasets
```

You can change the data path with `--data`.

## Train SimCLR

Example:

```bash
python run.py \
  --data ./datasets \
  --dataset-name cifar10 \
  --arch resnet18 \
  --epochs 100 \
  --batch-size 256 \
  --augmentation baseline \
  --experiment-name cifar10_simclr
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--augmentation` | Use `baseline`, `no_blur`, `no_color_jitter`, `no_grayscale`, `weak`, `medium`, or `strong`. |
| `--no-projection-head` | Train without the MLP projection head. |
| `--batch-size` | Set the SimCLR batch size. |
| `--lr` | Set the pretraining learning rate. |
| `--temperature` | Set the contrastive loss temperature. |
| `--disable-cuda` | Run on CPU. |

Training outputs are saved under `runs/<experiment-name>/` when
`--experiment-name` is given. The main files are:

- `checkpoint_XXXX.pth.tar`
- `config.yml`
- `metrics.csv`
- `summary.json`
- `training_history.json`
- `training_curves.png`

## Linear Evaluation

Frozen linear evaluation trains only the final classifier:

```bash
python linear_eval.py \
  --data ./datasets \
  --dataset-name cifar10 \
  --arch resnet18 \
  --checkpoint-path runs/cifar10_simclr/checkpoint_0100.pth.tar \
  --train-mode linear \
  --label-fraction 1.0 \
  --epochs 100 \
  --output-dir eval/cifar10_linear
```

Fine-tuning updates both the encoder and classifier:

```bash
python linear_eval.py \
  --data ./datasets \
  --dataset-name cifar10 \
  --arch resnet18 \
  --checkpoint-path runs/cifar10_simclr/checkpoint_0100.pth.tar \
  --train-mode finetune \
  --label-fraction 0.1 \
  --epochs 100 \
  --output-dir eval/cifar10_finetune_10p
```

Each run saves `results.json` and `linear_eval_curves.png`.

## Supervised Low-Label Baseline

Train a supervised ResNet from scratch with a small label fraction:

```bash
python supervised_baseline.py \
  --data ./datasets \
  --dataset-name cifar10 \
  --arch resnet18 \
  --label-fraction 0.1 \
  --epochs 100 \
  --output-dir eval/supervised_10p
```

Run several label fractions:

```bash
python run_supervised_low_label_suite.py \
  --data ./datasets \
  --label-fractions 0.01 0.1 1.0 \
  --epochs 100 \
  --output-dir eval/supervised_low_label
```

## Experiments Included

This repository includes code and saved figures for:

- data augmentation ablation
- projection head ablation
- batch size ablation
- large-batch learning-rate follow-up
- low-label frozen evaluation and transfer comparison

Use a GPU when possible. Full SimCLR training is slow on CPU, and large batch
sizes need more GPU memory.
