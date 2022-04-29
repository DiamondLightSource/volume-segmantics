from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utilities import base_data_utils as utils
from utilities import config as cfg
from utilities.datasets import get_2d_training_dataset, get_2d_validation_dataset
from utilities.settingsdata import SettingsData


def get_2d_training_dataloaders(
    image_dir: Path, label_dir: Path, settings: SettingsData
) -> Tuple[DataLoader, DataLoader]:
    """Returns 2d training and validation dataloaders with indices split at random
    according to the percentage split specified in settings.

    Args:
        image_dir (Path): Directory of data images
        label_dir (Path): Directory of label images
        settings (SettingsData): Settings object

    Returns:
        Tuple[DataLoader, DataLoader]: 2d training and validation dataloaders
    """

    training_set_prop = settings.training_set_proportion
    batch_size = utils.get_batch_size(settings)

    full_training_dset = get_2d_training_dataset(image_dir, label_dir, settings)
    full_validation_dset = get_2d_validation_dataset(image_dir, label_dir, settings)
    # split the dataset in train and test set
    dset_length = len(full_training_dset)
    indices = torch.randperm(dset_length).tolist()
    train_idx, validate_idx = np.split(indices, [int(dset_length * training_set_prop)])
    training_dataset = Subset(full_training_dset, train_idx)
    validation_dataset = Subset(full_validation_dset, validate_idx)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader
