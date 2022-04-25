import re
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset

from utilities import augmentations as augs
from utilities.settingsdata import SettingsData


class Unet2dDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (pathlib.Path): path to images folder
        masks_dir (pathlib.Path): path to segmentation masks folder
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, contrast adjustments)
        imagenet_norm (bool): Whether to normalise according to imagenet stats
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)


    """

    imagenet_mean = 0.449
    imagenet_std = 0.226

    def __init__(
        self,
        images_dir,
        masks_dir,
        preprocessing=None,
        augmentation=None,
        imagenet_norm=True,
        postprocessing=None,
    ):

        self.images_fps = sorted(list(images_dir.glob("*.png")), key=self.natsort)
        self.masks_fps = sorted(list(masks_dir.glob("*.png")), key=self.natsort)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.imagenet_norm = imagenet_norm
        self.postprocessing = postprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(str(self.images_fps[i]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.masks_fps[i]), 0)

        # apply pre-processing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                # Convert to float
                image = image.astype(np.float32)
                image = image / 255
            image = image - self.imagenet_mean
            image = image / self.imagenet_std

        # apply post-processing
        if self.postprocessing:
            sample = self.postprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)

    @staticmethod
    def natsort(item):
        return [
            int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", str(item))
        ]


def get_2d_training_dataset(
    image_dir: Path, label_dir: Path, settings: SettingsData
) -> Unet2dDataset:

    img_size = settings.image_size
    return Unet2dDataset(
        image_dir,
        label_dir,
        preprocessing=augs.get_preprocess_augs(img_size),
        augmentation=augs.get_train_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
    )


def get_2d_validation_dataset(
    image_dir: Path, label_dir: Path, settings: SettingsData
) -> Unet2dDataset:

    img_size = settings.image_size
    return Unet2dDataset(
        image_dir,
        label_dir,
        preprocessing=augs.get_preprocess_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
    )