import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class GaussianBlur(object):
    """对单张图片进行高斯模糊, 在 CPU 上实现"""
    def __init__(self, kernel_size):
        """
        初始化高斯模糊模块

        :param kernel_size: 卷积核大小 (一般为图像尺寸的10%)
        """
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1 # 计算半径 + 变为奇数
        # 水平方向卷积
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, 
                                padding=0, 
                                bias=False, 
                                groups=3 # RGB 三个通道独立卷积
                                )
        # 垂直方向卷积
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, 
                                padding=0, 
                                bias=False, 
                                groups=3)
        self.k = kernel_size
        self.r = radias
        # 整体模糊操作: 先 padding，再水平卷积，再垂直卷积
        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
        # PIL 图像 与 Tensor 相互转换
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        """
        对输入图片执行高斯模糊

        :param img: PIL Image
        :return: 模糊后的 PIL Image
        """
        img = self.pil_to_tensor(img).unsqueeze(0) # 转为 Tensor 并添加 batch 维度
        # 随机采样高斯分布的 sigma -> 增强数据多样性, 提升鲁棒性
        sigma = np.random.uniform(0.1, 2.0)
        # 构建一维高斯核
        x = np.arange(-self.r, self.r + 1)
        # 高斯函数公式
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        # 归一化 (保证和为1)
        x = x / x.sum()
        # 转为 tensor，并扩展到 3 个通道 (RGB)
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        # 设置水平方向卷积核
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        # 设置垂直方向卷积核
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        # 前向传播: 执行模糊 + 去掉 batch 维度
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img