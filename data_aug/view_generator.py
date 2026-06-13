import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Generate several random augmented views from one image."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # 1. Apply the same transform pipeline multiple times to get different views.
        return [self.base_transform(x) for _ in range(self.n_views)]
