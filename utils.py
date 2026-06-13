import os
import shutil

import torch
import yaml


def _yaml_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_yaml_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _yaml_safe_value(item) for key, item in value.items()}
    return str(value)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    保存模型检查点

    :param state: 要保存的内容 (一个字典), 包括: 
                  - epoch
                  - 模型参数 state_dict
                  - 优化器参数
    :param is_best: 是否为当前最优模型
    :param filename: 保存文件名
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    """
    保存训练配置文件 (参数)

    :param model_checkpoints_folder: 模型保存目录
    :param args: 训练参数
    """
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        config = {key: _yaml_safe_value(value) for key, value in vars(args).items()}
        yaml.dump(config, outfile, default_flow_style=False) # 保存为 yaml 文件


def accuracy(output, target, topk=(1,)):
    """
    计算 Top-K 准确率

    :param output: 模型输出 logits, 形状为 [batch_size, num_classes]
    :param target: 真实标签，形状为 [batch_size]
    :param topk: 需要计算的 K 值, 例如 (1, 5)
    :return: 返回一个列表，对应每个 topk 的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 取每个样本预测概率最高的前 maxk 个类别
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # 判断预测是否正确 -> pred 和 target 对比
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # 分别计算 top-k 准确率 -> 取前 k 个预测, reshape 后展开统计正确数量
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
