import logging
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # 进度条
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        """
        初始化 SimCLR 模型训练器

        kwargs 中传入：
        - args: 训练参数
        - model: 模型
        - optimizer: 优化器
        - scheduler: 学习率调度器
        """
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # TensorBoard 日志记录
        self.writer = SummaryWriter(log_dir=self.args.run_dir)
        os.makedirs(self.writer.log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, 'training.log'), 
            level=logging.DEBUG
            )
        # 交叉熵损失函数 -> 在 SimCLR 中，最终会把 "对比学习" 转成 "分类任务"
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        """
        计算 SimCLR 中的 InfoNCE Loss 所需的 logits 和 labels

        :param features: 模型输出的特征，形状一般为 [2 * batch_size, feature_dim]
        :return:
            logits: 相似度得分
            labels: 正样本标签
        """
        # 1. 构造标签矩阵
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        # 通过两两比较生成一个布尔矩阵 -> 若样本 i 和样本 j 来自同一张原图, 则为 True; 否则为 False
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        # 2. 特征归一化
        features = F.normalize(features, dim=1)
        # 3. 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)

        # 原作者断言, 方便调试时检查维度: 
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # 4. 去掉对角线元素 -> 对角线元素是样本与自身的相似度, 没有意义不参与训练
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # 原作者断言, 方便调试时检查维度: 
        # assert similarity_matrix.shape == labels.shape

        # 5. 提取正样本对 -> labels.bool() 为 True 的位置, 就是正样本
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # 6. 提取负样本对 -> labels.bool() 为 False 的位置, 就是负样本
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # 7. 拼接 logits -> 按照 [正样本 | 负样本] 的顺序拼接
        logits = torch.cat([positives, negatives], dim=1)
        # 构造监督标签 -> 因为每一行的第 0 列是正样本，所以目标标签全是 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        # 8. 温度缩放 -> 除以温度参数来调整 logits 分布, 使模型更容易学习区分正负样本的能力
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        """
        执行训练流程

        :param train_loader: 训练数据加载器
        """
        # 混合精度训练的梯度缩放器, 如果 fp16_precision=False 则不起作用
        scaler = GradScaler(enabled=self.args.fp16_precision)
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0 # 迭代步数
        epoch_metrics = []
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        # 循环 epoch
        for epoch_counter in range(self.args.epochs):
            epoch_loss_total = 0.0
            epoch_top1_total = 0.0
            epoch_top5_total = 0.0
            epoch_steps = 0
            for images, _ in tqdm(train_loader):
                # [batch_size, C, H, W] -> [2 * batch_size, C, H, W]
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                # 前向传播, autocast 用于自动混合精度
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images) # 模型提取特征
                    logits, labels = self.info_nce_loss(features) # 计算 InfoNCE 所需的 logits 和 labels
                    loss = self.criterion(logits, labels) # 用交叉熵计算损失

                # 梯度清零
                self.optimizer.zero_grad()
                # 反向传播
                scaler.scale(loss).backward()
                # 参数更新
                scaler.step(self.optimizer)
                scaler.update()
                # 计算 top1 / top5 准确率 -> 看正样本是否排在前 1 或前 5
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                loss_value = float(loss.item())
                top1_value = float(top1[0].item())
                top5_value = float(top5[0].item())
                epoch_loss_total += loss_value
                epoch_top1_total += top1_value
                epoch_top5_total += top5_value
                epoch_steps += 1
                # 定期记录训练日志
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss_value, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1_value, global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5_value, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # 前 10 epoch 不调整学习率, 之后开始使用 scheduler
            if epoch_counter >= 10:
                self.scheduler.step()
            epoch_summary = {
                'epoch': epoch_counter + 1,
                'loss': epoch_loss_total / max(epoch_steps, 1),
                'top1': epoch_top1_total / max(epoch_steps, 1),
                'top5': epoch_top5_total / max(epoch_steps, 1),
                'learning_rate': float(self.optimizer.param_groups[0]['lr']),
            }
            epoch_metrics.append(epoch_summary)
            logging.debug(
                f"Epoch: {epoch_counter + 1}\tLoss: {epoch_summary['loss']}\tTop1 accuracy: {epoch_summary['top1']}")

        logging.info("Training has finished.")
        # 保存模型检查点
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        metrics_csv_path = os.path.join(self.writer.log_dir, 'metrics.csv')
        with open(metrics_csv_path, 'w', newline='') as outfile:
            writer = csv.DictWriter(
                outfile,
                fieldnames=['epoch', 'loss', 'top1', 'top5', 'learning_rate'])
            writer.writeheader()
            writer.writerows(epoch_metrics)
        summary = {
            'dataset_name': self.args.dataset_name,
            'arch': self.args.arch,
            'augmentation': self.args.augmentation,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'temperature': self.args.temperature,
            'seed': self.args.seed,
            'out_dim': self.args.out_dim,
            'use_projection_head': self.args.use_projection_head,
            'checkpoint_path': os.path.join(self.writer.log_dir, checkpoint_name),
            'metrics_csv_path': metrics_csv_path,
            'final_loss': epoch_metrics[-1]['loss'],
            'final_top1': epoch_metrics[-1]['top1'],
            'final_top5': epoch_metrics[-1]['top5'],
            'epoch_metrics': epoch_metrics,
        }
        with open(os.path.join(self.writer.log_dir, 'summary.json'), 'w') as outfile:
            json.dump(summary, outfile, indent=2)
        self.writer.close()
        return summary
