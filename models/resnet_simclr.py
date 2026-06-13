import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, use_projection_head=True):
        """
        初始化 SimCLR 模型

        :param base_model: backbone 的名称 (resnet18 或 resnet50)
        :param out_dim: 最终输出特征维度 (projection head 的输出维度)
        :param use_projection_head: 是否使用两层 MLP projection head
        """
        super(ResNetSimCLR, self).__init__()
        # 定义可选的 ResNet 模型字典
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        # 获取 backbone 模型
        self.backbone = self._get_basemodel(base_model)
        # 获取 ResNet 最后一层全连接层的输入维度
        dim_mlp = self.backbone.fc.in_features
        if use_projection_head:
            # Linear(dim_mlp -> dim_mlp) -> ReLU -> Linear(dim_mlp -> out_dim)
            self.backbone.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.backbone.fc
                )
        else:
            # 无 projection head 时，直接使用 encoder 表征参与对比损失。
            self.backbone.fc = nn.Identity()

    def _get_basemodel(self, model_name):
        """
        根据名称获取 ResNet 模型

        :param model_name: 模型名称
        :return: 对应的 ResNet 模型
        """
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        """
        前向传播

        :param x: 输入图像 (Tensor)
        :return: 投影后的特征向量
        """
        return self.backbone(x)
