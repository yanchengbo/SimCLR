# SimCLR Project

This project is a PyTorch project for SimCLR.

SimCLR is a self-supervised learning method. It first trains a model without
class labels. After that, we test the learned features with a classifier.

The code supports CIFAR-10.

## Project Files

| Path | What it is for |
| --- | --- |
| `run.py` | Train the SimCLR model. |
| `simclr.py` | Main SimCLR training loop and loss code. |
| `linear_eval.py` | Test a saved SimCLR checkpoint with linear evaluation. |
| `linear_eval_gc.py` | Linear evaluation script used by Guochen Wang for the projection head and batch size experiments. |
| `models/resnet_simclr.py` | ResNet model used by SimCLR. |
| `data_aug/` | Data augmentation code. |
| `results/` | Saved result files from experiments. |
| `*_figures/` | Figures used to show experiment results. |
| `main.tex` | Project report source file. |

## Data

The scripts can download the datasets for you.

By default, training data is saved in:

```text
./datasets
```

You can change this path with `-data` or `--data`.

Supported datasets:

- `cifar10`

## Train SimCLR

Example for CIFAR-10:

```bash
python run.py \
  -data ./datasets \
  -dataset-name cifar10 \
  --epochs 100 \
  --batch-size 256 \
  --experiment-name cifar10_simclr
```

If only want to check that the code runs, use a short CPU run:

```bash
python run.py \
  -dataset-name cifar10 \
  --epochs 1 \
  --batch-size 64 \
  --workers 0 \
  --disable-cuda \
  --experiment-name smoke_test
```

## Useful Training Options

| Option | Meaning |
| --- | --- |
| `--arch` | Backbone model. The main choices are `resnet18` and `resnet50`. |
| `--epochs` | Number of training epochs. |
| `--batch-size` | Batch size. |
| `--lr` | Learning rate. |
| `--temperature` | Temperature value for the contrastive loss. |
| `--augmentation` | Augmentation setting. Choices are `baseline`, `no_blur`, `no_color_jitter`, and `no_grayscale`. |
| `--no-projection-head` | Turn off the projection head. |
| `--fp16-precision` | Use mixed precision training on GPU. |
| `--disable-cuda` | Run on CPU. |

## Training Output

If `--experiment-name` is used, files are saved in:

```text
runs/<experiment-name>/
```

The main output files are:

- `checkpoint_XXXX.pth.tar`: saved model checkpoint.
- `config.yml`: settings used for the run.
- `summary.json`: short summary of the run.
- `training.log`: training log.

For example, a 100 epoch run saves:

```text
checkpoint_0100.pth.tar
```

## Linear Evaluation

After SimCLR training, run linear evaluation on a checkpoint.

Example:

```bash
python linear_eval.py \
  --data ./datasets \
  --dataset-name cifar10 \
  --arch resnet18 \
  --checkpoint-path runs/cifar10_simclr/checkpoint_0100.pth.tar \
  --epochs 20 \
  --batch-size 256 \
  --output-dir eval/cifar10_simclr
```

The script saves:

- `results.json`
- `linear_eval_curves.png`

You can also fine-tune the full model instead of only training the classifier:

```bash
python linear_eval.py \
  --data ./datasets \
  --dataset-name cifar10 \
  --arch resnet18 \
  --checkpoint-path runs/cifar10_simclr/checkpoint_0100.pth.tar \
  --train-mode finetune \
  --epochs 20 \
  --output-dir eval/cifar10_finetune
```

## Optional Frozen Feature Evaluation

`linear_eval_gc.py` is another script for CIFAR-10. Guochen Wang used this file
for the projection head and batch size experiments. It first extracts frozen
features from the encoder. Then it trains a linear classifier on those features.

Example:

```bash
python linear_eval_gc.py \
  --checkpoint runs/cifar10_simclr/checkpoint_0100.pth.tar \
  --name cifar10_frozen_eval \
  --epochs 100
```

The result is saved as a JSON file in `feature_eval/results/`.

## Experiments in This Repo

This repo includes saved results and figures for:

- batch size experiments
- projection head experiments
- augmentation experiments
- learning rate experiments
- low-label transfer experiments

The figure folders are:

- `batch_figures/`
- `head_figures/`
- `augmentation_figures/`
- `lr_figures/`
- `low_label_figures/`

## Notes

- Use a GPU if possible. SimCLR training can be slow on CPU.
- Large batch sizes need more GPU memory.
- Use `--seed` if you want a run that is easier to repeat.
- The code downloads datasets automatically, so the first run may take longer.
