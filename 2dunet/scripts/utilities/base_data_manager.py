import logging
from pathlib import Path

import numpy as np

from utilities.base_data_utils import (clip_to_uint8, downsample_data,
                                       get_numpy_from_path)
from utilities.settingsdata import SettingsData


class BaseDataManager:
    def __init__(self, data_vol_path: str, settings: SettingsData) -> None:
        self.data_vol_shape = None
        self.data_mean = None
        self.data_vol_path = Path(data_vol_path)
        self.settings = settings
        self.st_dev_factor = settings.st_dev_factor
        self.downsample = settings.downsample
        self.data_vol = get_numpy_from_path(
            self.data_vol_path, internal_path=settings.data_hdf5_path
        )
        self.preprocess_data()

    def preprocess_data(self):
        if self.downsample:
            self.data_vol = downsample_data(self.data_vol)
        self.data_vol_shape = self.data_vol.shape
        logging.info("Calculating mean of data...")
        self.data_mean = np.nanmean(self.data_vol)
        logging.info(f"Mean value: {self.data_mean}")
        if self.settings.clip_data:
            self.data_vol = clip_to_uint8(self.data_vol,self.data_mean, self.st_dev_factor)
        if np.isnan(self.data_vol).any():
            logging.info(f"Replacing NaN values.")
            self.data_vol = np.nan_to_num(self.data_vol, copy=False)
