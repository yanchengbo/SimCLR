import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """从同一张图片生成多个随机增强视图"""

    def __init__(self, base_transform, n_views=2):
        """
        初始化视图生成器

        :param base_transform: 基础数据增强方法 (如随机裁剪、颜色扰动等组合)
        :param n_views: 每张图片生成多少个不同视图 (SimCLR 默认为 2)
        """
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        """
        对输入图片生成多个增强版本

        :param x: 输入图片 (PIL Image)
        :return: 包含多个增强视图的列表
        """
        return [self.base_transform(x) for i in range(self.n_views)] # 对同一张图片重复应用数据增强 n_views 次
