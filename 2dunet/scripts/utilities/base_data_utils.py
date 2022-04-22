import logging

import torch

from utilities.settingsdata import SettingsData
from utilities import config as cfg


def get_batch_size(settings: SettingsData) -> int:

    cuda_device_num = settings.cuda_device
    total_gpu_mem = torch.cuda.get_device_properties(cuda_device_num).total_memory
    allocated_gpu_mem = torch.cuda.memory_allocated(cuda_device_num)
    free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024 ** 3

    if free_gpu_mem < cfg.BIG_CUDA_SIZE:
        batch_size = cfg.SMALL_CUDA_BATCH
    else:
        batch_size = cfg.BIG_CUDA_BATCH
    logging.info(
        f"Free GPU memory is {free_gpu_mem:0.2f} GB. Batch size will be "
        f"{batch_size}."
    )
    return batch_size
