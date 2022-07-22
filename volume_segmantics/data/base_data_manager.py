import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np
import volume_segmantics.utilities.base_data_utils as utils


class BaseDataManager:
    def __init__(
        self, data_vol: Union[Path, str, np.ndarray], settings: SimpleNamespace
    ) -> None:
        self.data_vol_shape = None
        self.data_mean = None
        self.data_vol_path = utils.setup_path_if_exists(data_vol)
        self.settings = settings
        self.st_dev_factor = settings.st_dev_factor
        self.downsample = settings.downsample
        if self.data_vol_path is not None:
            self.data_vol, self.input_data_chunking = utils.get_numpy_from_path(
                self.data_vol_path, internal_path=settings.data_hdf5_path
            )
        elif isinstance(data_vol, np.ndarray):
            self.data_vol = data_vol
            self.input_data_chunking = True
        self._preprocess_data()

    def _preprocess_data(self):
        if self.downsample:
            self.data_vol = utils.downsample_data(self.data_vol)
        self.data_vol_shape = self.data_vol.shape
        logging.info("Calculating mean of data...")
        self.data_mean = np.nanmean(self.data_vol)
        logging.info(f"Mean value: {self.data_mean}")
        if self.settings.clip_data:
            self.data_vol = utils.clip_to_uint8(
                self.data_vol, self.data_mean, self.st_dev_factor
            )
        if np.isnan(self.data_vol).any():
            logging.info(f"Replacing NaN values.")
            self.data_vol = np.nan_to_num(self.data_vol, copy=False)
