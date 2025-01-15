"""This module contains code for transforming the data for training the defect
detection model."""

import random
from typing import Callable, List, Sequence, Tuple

import torch
from omegaconf import DictConfig
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2 as transforms

import wandb

PIPE_MEAN = [0.518, 0.501, 0.45]
PIPE_STD = [0.269, 0.267, 0.255]
SEWERML_MEAN = [0.523, 0.453, 0.345]
SEWERML_STD = [0.210, 0.199, 0.154]


class GaussianBlur:
    """Gaussian blur as a callable object.

    Args:
        sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
            Defaults to [0.1, 2.0].
    """

    def __init__(self, sigma: Sequence[float] = None):
        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class EqualizationAndSolarization:
    """Either applies equalization or solarization."""

    def __call__(self, img: Image) -> Image:
        if random.random() > 0.5:
            img = ImageOps.equalize(img)
        else:
            img = ImageOps.solarize(img)

        return img


class NCropAugmentation:
    """Creates a pipeline that apply a transformation pipeline multiple times.

    Args:
        transform (Callable): transformation pipeline.
        num_crops (int): number of crops to create from the transformation pipeline.
    """

    def __init__(self, transform: Callable, num_crops: int):
        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


def instantiate_transforms(conf_dict: DictConfig, img_size: int) -> transforms.Compose:

    augmentations = []
    augmentations_params = {
        "use_random_resized_crop": conf_dict.use_random_resized_crop,
        "use_color_jitter": conf_dict.use_color_jitter,
    }

    if conf_dict.use_random_resized_crop:
        augmentations.append(
            transforms.RandomResizedCrop(
                (img_size, img_size),
                scale=(
                    conf_dict.min_scale,
                    conf_dict.max_scale,
                ),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        augmentations_params["min_scale"] = conf_dict.min_scale
        augmentations_params["max_scale"] = conf_dict.max_scale
    else:
        augmentations.append(transforms.Resize((img_size, img_size), antialias=True))

    if conf_dict.use_color_jitter:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=conf_dict.brightness,
                        contrast=conf_dict.contrast,
                        saturation=conf_dict.saturation,
                        hue=conf_dict.hue,
                    )
                ],
                p=conf_dict.color_jitter_prob,
            )
        )
        augmentations_params["brightness"] = conf_dict.brightness
        augmentations_params["contrast"] = conf_dict.contrast
        augmentations_params["saturation"] = conf_dict.saturation
        augmentations_params["hue"] = conf_dict.hue

    wandb.config.update({"augmentations": augmentations_params})

    augmentations.append(transforms.ToImage())
    augmentations.append(transforms.ToDtype(torch.float32, scale=True))
    augmentations.append(transforms.RandomHorizontalFlip())
    augmentations.append(transforms.Normalize(mean=SEWERML_MEAN, std=SEWERML_STD))

    return transforms.Compose(augmentations)


def instantiate_supervised_transforms(
    img_size: int, pretrained: bool
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Instantiates the transformations for supervised learning."""

    if pretrained:
        normalize_kwargs = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD}
    else:
        normalize_kwargs = {"mean": SEWERML_MEAN, "std": SEWERML_STD}

    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            ),  # Color jittering
            transforms.RandomEqualize(),  # Equalize the image randomly
            transforms.RandomAutocontrast(),  # Invert the image colors randomly
            transforms.RandomAffine(5),  # Random affine transformation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize(
                (img_size, img_size), antialias=True
            ),  # Resize to 224 x 224
            transforms.RandomHorizontalFlip(),  # Image flipping
            transforms.RandomErasing(
                scale=(0.01, 0.05), ratio=(0.1, 1)
            ),  # Random erasing
            transforms.Normalize(**normalize_kwargs),  # Image normalization
        ]
    )
    val_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(**normalize_kwargs),  # Image normalization
            transforms.Resize((img_size, img_size), antialias=True),
        ]
    )

    return train_transform, val_test_transform


def instantiate_ssl_transforms(
    img_size: int,
    num_crops: int,
    pretrained: bool,
) -> NCropAugmentation:
    """Creates a pipeline of transformations given a"""

    if pretrained:
        normalize_kwargs = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD}
    else:
        normalize_kwargs = {"mean": SEWERML_MEAN, "std": SEWERML_STD}

    augmentations = [
        transforms.RandomResizedCrop(
            (img_size, img_size),
            scale=(0.5, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=(-0.2, 0.2)
                ),
            ],
            p=0.5,
        ),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([GaussianBlur(sigma=(0.1, 1))], p=0.3),
        transforms.RandomApply([EqualizationAndSolarization()], p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),  # Image flipping
        transforms.ToTensor(),
        transforms.Normalize(**normalize_kwargs),  # Image normalization
    ]

    transform = transforms.Compose(augmentations)

    return NCropAugmentation(transform=transform, num_crops=num_crops)
