import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch


def create_unet_on_device(device_num: int, model_struc_dict: dict) -> smp.Unet:

    unet = smp.Unet(**model_struc_dict)
    logging.info(f"Sending the U-Net model to device {device_num}")
    return unet.to(device_num)


def create_unet_from_file(
    weights_fn: Path, gpu: bool = True, device_num: int = 0
) -> smp.Unet:
    """Creates and returns a U-Net model."""
    if gpu:
        map_location = f"cuda:{device_num}"
    else:
        map_location = "cpu"
    weights_fn = weights_fn.resolve()
    logging.info("Loading 2d U-net model.")
    model_dict = torch.load(weights_fn, map_location=map_location)
    unet_model = create_unet_on_device(device_num, model_dict["model_struc_dict"])
    logging.info("Loading in the saved weights.")
    unet_model.load_state_dict(model_dict["model_state_dict"])
    return unet_model
