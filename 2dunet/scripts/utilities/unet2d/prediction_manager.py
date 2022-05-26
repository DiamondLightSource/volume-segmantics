from pathlib import Path

import numpy as np
import utilities.base_data_utils as utils
from utilities.base_data_manager import BaseDataManager
from utilities.settingsdata import SettingsData
from utilities.unet2d.predictor import Unet2dPredictor


class Unet2DPredictionManager(BaseDataManager):
    def __init__(
        self, predictor: Unet2dPredictor, data_vol_path: str, settings: SettingsData
    ) -> None:
        super().__init__(data_vol_path, settings)
        self.input_data_chunking = None
        self.predictor = predictor
        self.settings = settings

    def predict_volume_to_path(self, output_path: Path, one_hot: bool = False) -> None:
        quality = utils.get_prediction_quality(self.settings)
        if quality == utils.Quality.LOW:
            if one_hot:
                prediction = self.predictor.predict_single_axis_to_one_hot(self.data_vol)
            else:
                prediction, _ = self.predictor.predict_single_axis(self.data_vol)
        if quality == utils.Quality.MEDIUM:
            if one_hot:
                prediction = self.predictor.predict_3ways_to_one_hot(self.data_vol)
            else:
                prediction = self.predictor.predict_3ways_max_probs(self.data_vol)
        utils.save_data_to_hdf5(prediction, output_path)
