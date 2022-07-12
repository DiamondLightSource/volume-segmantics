"""
Provides functions to enable image augmentations for training and prediction.
"""

import math

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import volume_segmantics.utilities.config as cfg


def get_train_preprocess_augs(img_size: int) -> A.core.composition.Compose:
    """Returns the augmentations required to pad images to the correct square size
    for training a network.

    Args:
        img_size (int): Length of square image edge.

    Returns:
        A.core.composition.Compose: An augmentation to resize the image if needed.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
        ]
    )


def get_padded_dimension(dimension: int) -> int:
    """Returns the closest image dimension that can be divided by the divisor
    specified in the config.

    Args:
        dimension (int): input dimension.

    Returns:
        int: Dimension that can divided by divisor
    """
    image_divisor = cfg.IM_SIZE_DIVISOR
    if dimension % image_divisor == 0:
        return dimension
    return (math.floor(dimension / image_divisor) + 1) * image_divisor


def get_pred_preprocess_augs(
    img_size_y: int, img_size_x: int
) -> A.core.composition.Compose:
    """Returns the augmentations required to pad images to the correct size for
    prediction

    Args:
        img_size_y (int): Image size in y
        img_size_x (int): Image size in x

    Returns:
        A.core.composition.Compose: An augmentation to resize the image if needed.
    """
    padded_y_dim = get_padded_dimension(img_size_y)
    padded_x_dim = get_padded_dimension(img_size_x)
    return A.Compose(
        [
            A.PadIfNeeded(min_height=padded_y_dim, min_width=padded_x_dim, p=1.0),
        ]
    )


def get_train_augs(img_size: int) -> A.core.composition.Compose:
    """Returns the augmentations used for training a network.

    Args:
        img_size (int): The square image size required.

    Returns:
        A.core.composition.Compose: Augmentations for training.
    """
    return A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(img_size // 2, img_size),
                height=img_size,
                width=img_size,
                p=0.5,
            ),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.04, p=0.5
                    ),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ],
                p=0.5,
            ),
            A.CLAHE(p=0.5),
            A.OneOf([A.RandomBrightnessContrast(p=0.5), A.RandomGamma(p=0.5)], p=0.5),
        ]
    )


def get_postprocess_augs() -> A.core.composition.Compose:
    """Returns the final augmentations applied to the images.

    Returns:
        A.core.composition.Compose: Postprocessing augmentations.
    """
    return A.Compose([ToTensorV2()])
