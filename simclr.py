import csv
import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    accuracy,
    save_checkpoint,
    save_config_file,
    save_json,
    save_simclr_history_plot,
)

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.writer = SummaryWriter(log_dir=self.args.run_dir)
        os.makedirs(self.writer.log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"),
            level=logging.DEBUG,
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def _current_lr(self):
        if hasattr(self.scheduler, "get_last_lr"):
            return float(self.scheduler.get_last_lr()[0])
        return float(self.scheduler.get_lr()[0])

    def info_nce_loss(self, features):
        """Build logits and labels for the NT-Xent contrastive loss."""
        # 1. Mark the two augmented views from the same image as positives.
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for _ in range(self.args.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # 2. Normalize features and compute all pair similarities.
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # 3. Remove self-similarity on the diagonal.
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # 4. Put the positive logit first, followed by all negative logits.
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)

        # 5. The positive class is always column 0 after concatenation.
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        """Run SimCLR pretraining and save checkpoint files."""
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        history = []
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"CUDA disabled: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            total_loss = 0.0
            total_top1 = 0.0
            total_top5 = 0.0
            num_batches = 0

            for images, _ in tqdm(train_loader, desc=f"SimCLR epoch {epoch_counter + 1}/{self.args.epochs}"):
                # 1. Stack two views into one batch of shape [2B, C, H, W].
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                # 2. Apply the contrastive loss to projection features z.
                with autocast(enabled=self.args.fp16_precision):
                    _, projections = self.model(images, return_embedding=True)
                    logits, labels = self.info_nce_loss(projections)
                    loss = self.criterion(logits, labels)

                # 3. Update model parameters.
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                loss_value = float(loss.item())
                top1_value = float(top1[0].item())
                top5_value = float(top5[0].item())
                total_loss += loss_value
                total_top1 += top1_value
                total_top5 += top5_value
                num_batches += 1

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar("loss", loss_value, global_step=n_iter)
                    self.writer.add_scalar("acc/top1", top1_value, global_step=n_iter)
                    self.writer.add_scalar("acc/top5", top5_value, global_step=n_iter)
                    self.writer.add_scalar("learning_rate", self._current_lr(), global_step=n_iter)
                n_iter += 1

            # 4. Keep the original 10-epoch warmup behavior from this project.
            if epoch_counter >= 10:
                self.scheduler.step()

            epoch_metrics = {
                "epoch": epoch_counter + 1,
                "train_loss": total_loss / max(num_batches, 1),
                "train_top1": total_top1 / max(num_batches, 1),
                "train_top5": total_top5 / max(num_batches, 1),
                "learning_rate": self._current_lr(),
            }
            history.append(epoch_metrics)
            self.writer.add_scalar("epoch/loss", epoch_metrics["train_loss"], global_step=epoch_counter + 1)
            self.writer.add_scalar("epoch/top1", epoch_metrics["train_top1"], global_step=epoch_counter + 1)
            self.writer.add_scalar("epoch/top5", epoch_metrics["train_top5"], global_step=epoch_counter + 1)
            self.writer.add_scalar(
                "epoch/learning_rate",
                epoch_metrics["learning_rate"],
                global_step=epoch_counter + 1,
            )
            logging.debug(
                f"Epoch: {epoch_counter + 1}\t"
                f"Loss: {epoch_metrics['train_loss']:.4f}\t"
                f"Top1 accuracy: {epoch_metrics['train_top1']:.2f}\t"
                f"Top5 accuracy: {epoch_metrics['train_top5']:.2f}"
            )
            print(
                f"Epoch {epoch_counter + 1}\t"
                f"Train Loss {epoch_metrics['train_loss']:.4f}\t"
                f"Train Top1 {epoch_metrics['train_top1']:.2f}\t"
                f"Train Top5 {epoch_metrics['train_top5']:.2f}"
            )

        logging.info("Training has finished.")
        return self._save_outputs(history)

    def _save_outputs(self, history):
        checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
        checkpoint_path = os.path.join(self.writer.log_dir, checkpoint_name)
        augmentation = getattr(self.args, "augmentation", None)

        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "arch": self.args.arch,
                "augmentation": augmentation,
                "aug_strength": augmentation,
                "use_projection_head": self.args.use_projection_head,
                "feature_dim": getattr(self.model, "feature_dim", None),
                "projection_dim": getattr(self.model, "projection_dim", None),
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=checkpoint_path,
        )

        metrics_csv_path = os.path.join(self.writer.log_dir, "metrics.csv")
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(
                outfile,
                fieldnames=["epoch", "train_loss", "train_top1", "train_top5", "learning_rate"],
            )
            writer.writeheader()
            writer.writerows(history)

        history_payload = {
            "run_name": self.args.run_name,
            "run_dir": os.path.abspath(self.writer.log_dir),
            "epochs": self.args.epochs,
            "arch": self.args.arch,
            "dataset_name": self.args.dataset_name,
            "history": history,
        }
        history_path = os.path.join(self.writer.log_dir, "training_history.json")
        save_json(history_payload, history_path)

        plot_path = os.path.join(self.writer.log_dir, "training_curves.png")
        plot_saved = save_simclr_history_plot(history, plot_path, f"SimCLR Training: {self.args.run_name}")

        final_epoch = history[-1] if history else {}
        summary = {
            "dataset_name": self.args.dataset_name,
            "arch": self.args.arch,
            "augmentation": augmentation,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "temperature": self.args.temperature,
            "seed": self.args.seed,
            "out_dim": self.args.out_dim,
            "use_projection_head": self.args.use_projection_head,
            "feature_dim": getattr(self.model, "feature_dim", None),
            "projection_dim": getattr(self.model, "projection_dim", None),
            "checkpoint_path": checkpoint_path,
            "metrics_csv_path": metrics_csv_path,
            "training_history_path": history_path,
            "training_curves_path": plot_path if plot_saved else None,
            "final_loss": final_epoch.get("train_loss"),
            "final_top1": final_epoch.get("train_top1"),
            "final_top5": final_epoch.get("train_top5"),
            "epoch_metrics": history,
        }
        save_json(summary, os.path.join(self.writer.log_dir, "summary.json"))

        logging.info(f"Model checkpoint and metadata have been saved at {self.writer.log_dir}.")
        print(f"Saved checkpoint to {checkpoint_path}")
        print(f"Saved training history to {history_path}")
        if plot_saved:
            print(f"Saved training curves to {plot_path}")

        self.writer.close()
        return summary
