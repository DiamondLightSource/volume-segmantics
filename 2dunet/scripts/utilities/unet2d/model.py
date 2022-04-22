import logging

import segmentation_models_pytorch as smp


def create_unet_on_device(device_num: int, model_struc_dict: dict) -> smp.Unet:

    unet = smp.Unet(**model_struc_dict)
    logging.info(f"Sending the U-Net model to device {device_num}")
    return unet.to(device_num)
