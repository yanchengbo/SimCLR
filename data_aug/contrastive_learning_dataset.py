from torchvision import datasets, transforms

from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, augmentation="baseline", s=1):
        """Build the SimCLR augmentation pipeline used for pretraining."""
        valid_settings = {
            "baseline",
            "strong",
            "medium",
            "weak",
            "no_blur",
            "no_color_jitter",
            "no_grayscale",
        }
        if augmentation not in valid_settings:
            raise ValueError(f"Unknown augmentation setting: {augmentation}")

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transform_ops = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
        ]

        # 1. Add color jitter unless this run removes it or uses the weak setting.
        if augmentation not in {"no_color_jitter", "weak"}:
            transform_ops.append(transforms.RandomApply([color_jitter], p=0.8))

        # 2. Add random grayscale for the full SimCLR setting.
        if augmentation not in {"no_grayscale", "medium", "weak"}:
            transform_ops.append(transforms.RandomGrayscale(p=0.2))

        # 3. Add blur for the full SimCLR setting.
        if augmentation not in {"no_blur", "medium", "weak"}:
            transform_ops.append(GaussianBlur(kernel_size=int(0.1 * size)))

        transform_ops.append(transforms.ToTensor())
        return transforms.Compose(transform_ops)

    def get_dataset(self, name, n_views, augmentation="baseline", augmentation_strength=None):
        if augmentation_strength is not None:
            augmentation = augmentation_strength

        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32, augmentation),
                    n_views,
                ),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96, augmentation),
                    n_views,
                ),
                download=True,
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        return dataset_fn()
