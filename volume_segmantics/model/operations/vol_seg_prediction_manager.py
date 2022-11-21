from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.data.base_data_manager import BaseDataManager
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor


class VolSeg2DPredictionManager(BaseDataManager):
    """Class that manages prediction of data volumes to disk using a
    2d deep learning network.
    """

    def __init__(
        self,
        model_file_path: str,
        data_vol: Union[str, np.ndarray],
        settings: SimpleNamespace,
    ) -> None:
        """Inits VolSeg2DPredictionManager.

        Args:
            model_file_path (str): String of filepath to trained model to use for prediction.
            data_vol (Union[str, np.ndarray]): String of filepath to data volume or numpy array of data to predict segmentation of.
            settings (SimpleNamespace): A prediction settings object.
        """
        super().__init__(data_vol, settings)
        self.predictor = VolSeg2dPredictor(model_file_path, settings)
        self.settings = settings

    def get_label_codes(self) -> dict:
        """Returns a dictionary of label codes, retrieved from the saved model.

        Returns:
            dict: Label codes. These provide information on the labels that were used
            when training the model along with any associated metadata.
        """
        return self.predictor.label_codes

    def predict_volume_to_path(
        self, output_path: Union[Path, None], quality: Union[utils.Quality, None] = None
    ) -> np.ndarray:
        """Method which triggers prediction of a 3D segmentation to disk at a specified quality.

        Here 'quality' refers to the number of axes/rotations that the segmentation is predicted
        in. e.g. Low quality, single axis (x, y) prediction; medium quality, three axis (x, y),
        (x, z), (y, z) prediction; high quality 12 way (3 axis and 4 rotations) prediction.
        Multi-axis predictions are combined into a final output volume by using maximum probabilities.

        Args:
            output_path (Union[Path, None]): Path to predict volume to.
            quality (Union[utils.Quality, None], optional): A quality to predict the segmentation to. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        probs = None
        one_hot = self.settings.one_hot
        preferred_axis = utils.get_prediction_axis(
            self.settings
        )  # Specify single axis for prediction
        if quality is None:
            quality = utils.get_prediction_quality(self.settings)
        if quality == utils.Quality.LOW:
            if one_hot:
                prediction = self.predictor._predict_single_axis_to_one_hot(
                    self.data_vol, axis=preferred_axis
                )
            else:
                prediction, probs = self.predictor._predict_single_axis(
                    self.data_vol, axis=preferred_axis
                )
        if quality == utils.Quality.MEDIUM:
            if one_hot:
                prediction = self.predictor._predict_3_ways_one_hot(self.data_vol)
            else:
                prediction, probs = self.predictor._predict_3_ways_max_probs(
                    self.data_vol
                )
        if quality == utils.Quality.HIGH:
            if one_hot:
                prediction = self.predictor._predict_12_ways_one_hot(self.data_vol)
            else:
                prediction, probs = self.predictor._predict_12_ways_max_probs(
                    self.data_vol
                )
        if output_path is not None:
            utils.save_data_to_hdf5(
                prediction, output_path, chunking=self.input_data_chunking
            )
            if probs is not None and self.settings.output_probs:
                utils.save_data_to_hdf5(
                    probs,
                    f"{output_path.parent / output_path.stem}_probs.h5",
                    chunking=self.input_data_chunking,
                )
        return prediction
