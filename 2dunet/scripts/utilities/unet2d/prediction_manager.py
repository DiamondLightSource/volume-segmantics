from pathlib import Path

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

    def predict_volume_to_path(self, output_path: Path) -> None:
        quality = utils.get_prediction_quality(self.settings)
        if quality == utils.Quality.LOW:
            prediction = self.predictor.predict_single_axis(self.data_vol)
        if quality == utils.Quality.MEDIUM:
            prediction = self.predictor.predict_triple_axis(self.data_vol)
        utils.save_data_to_hdf5(prediction, output_path)
