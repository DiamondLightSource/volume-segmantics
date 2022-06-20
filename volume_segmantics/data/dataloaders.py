from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch.utils.data import DataLoader, Subset
from volume_segmantics.data.datasets import (get_2d_prediction_dataset,
                                             get_2d_training_dataset,
                                             get_2d_validation_dataset)


def get_2d_training_dataloaders(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    """Returns 2d training and validation dataloaders with indices split at random
    according to the percentage split specified in settings.

    Args:
        image_dir (Path): Directory of data images
        label_dir (Path): Directory of label images
        settings (SimpleNamespace): Settings object

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
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader


def get_2d_prediction_dataloader(
    data_vol: np.array, settings: SimpleNamespace
) -> DataLoader:
    pred_dataset = get_2d_prediction_dataset(data_vol)
    batch_size = utils.get_batch_size(settings, prediction=True)
    return DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for prediction
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
