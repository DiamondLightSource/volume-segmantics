import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import volume_segmantics.data.augmentations as augs
import volume_segmantics.utilities.config as cfg
from torch.utils.data import Dataset as BaseDataset


class VolSeg2dDataset(BaseDataset):
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

    imagenet_mean = cfg.IMAGENET_MEAN
    imagenet_std = cfg.IMAGENET_STD

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
            int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item))
        ]


class VolSeg2dPredictionDataset(BaseDataset):
    """Splits 3D data volume into 2D images for inference.

    Args:
        images_dir (pathlib.Path): path to images folder
        masks_dir (pathlib.Path): path to segmentation masks folder
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        imagenet_norm (bool): Whether to normalise according to imagenet stats
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)


    """

    imagenet_mean = cfg.IMAGENET_MEAN
    imagenet_std = cfg.IMAGENET_STD

    def __init__(
        self,
        data_vol,
        preprocessing=None,
        imagenet_norm=True,
        postprocessing=None,
    ):
        self.data_vol = data_vol
        self.preprocessing = preprocessing
        self.imagenet_norm = imagenet_norm
        self.postprocessing = postprocessing

    def __getitem__(self, i):

        image = self.data_vol[i]

        # apply pre-processing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        if self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                # Convert to float
                image = image.astype(np.float32)
                image = image / 255
            image = image - self.imagenet_mean
            image = image / self.imagenet_std

        # apply post-processing
        if self.postprocessing:
            sample = self.postprocessing(image=image)
            image = sample["image"]

        return image

    def __len__(self):
        return self.data_vol.shape[0]


def get_2d_training_dataset(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> VolSeg2dDataset:

    img_size = settings.image_size
    return VolSeg2dDataset(
        image_dir,
        label_dir,
        preprocessing=augs.get_train_preprocess_augs(img_size),
        augmentation=augs.get_train_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
    )


def get_2d_validation_dataset(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> VolSeg2dDataset:

    img_size = settings.image_size
    return VolSeg2dDataset(
        image_dir,
        label_dir,
        preprocessing=augs.get_train_preprocess_augs(img_size),
        postprocessing=augs.get_postprocess_augs(),
    )


def get_2d_prediction_dataset(data_vol: np.array) -> VolSeg2dPredictionDataset:
    y_dim, x_dim = data_vol.shape[1:]
    return VolSeg2dPredictionDataset(
        data_vol,
        preprocessing=augs.get_pred_preprocess_augs(y_dim, x_dim),
        postprocessing=augs.get_postprocess_augs(),
    )
