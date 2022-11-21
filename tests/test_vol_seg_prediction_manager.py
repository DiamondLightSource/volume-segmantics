from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor
from volume_segmantics.model.operations.vol_seg_prediction_manager import (
    VolSeg2DPredictionManager,
)
from volume_segmantics.utilities import Quality


@pytest.fixture()
def volseg_prediction_manager(model_path, rand_int_volume, prediction_settings):
    return VolSeg2DPredictionManager(model_path, rand_int_volume, prediction_settings)


class TestVolSegPredictionManager:
    @pytest.mark.gpu
    def test_prediction_manager_init(self, volseg_prediction_manager):
        assert isinstance(volseg_prediction_manager.predictor, VolSeg2dPredictor)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_volume_to_path_low(self, volseg_prediction_manager, tmp_path):
        output_dir = Path(tmp_path, "prediction_dir")
        output_dir.mkdir(exist_ok=True)
        output_path = Path(output_dir, "output_low.h5")
        prediction = volseg_prediction_manager.predict_volume_to_path(
            output_path, Quality.LOW
        )
        assert len(list(output_dir.glob("*.h5"))) == 1
        assert output_path.exists()
        assert isinstance(prediction, np.ndarray)
        assert prediction.dtype == np.uint8

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_volume_to_path_low_single_axis(
        self, model_path, rand_int_volume, prediction_settings, tmp_path
    ):
        original_shape = deepcopy(list(rand_int_volume.shape))
        print(original_shape)
        output_dir = Path(tmp_path, "prediction_dir")
        output_dir.mkdir(exist_ok=True)
        output_path = Path(output_dir, "output_low.h5")
        prediction_settings.prediction_axis = "y"
        prediction_manager = VolSeg2DPredictionManager(
            model_path, rand_int_volume, prediction_settings
        )
        prediction = prediction_manager.predict_volume_to_path(output_path, Quality.LOW)
        print(prediction.shape)
        assert len(list(output_dir.glob("*.h5"))) == 1
        assert output_path.exists()
        assert (
            prediction.shape[0] == original_shape[0]
        )  # Prediction rotated back to original
        assert (
            prediction.shape[1] == original_shape[1]
        )  # Prediction rotated back to original
        assert (
            prediction.shape[2] == original_shape[2]
        )  # Prediction rotated back to original

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_volume_to_path_low_probs(
        self, model_path, rand_int_volume, tmp_path, prediction_settings
    ):
        prediction_settings.output_probs = True
        pred_manager = VolSeg2DPredictionManager(
            model_path, rand_int_volume, prediction_settings
        )
        assert pred_manager.settings.output_probs == True
        output_dir = Path(tmp_path, "prediction_dir")
        output_dir.mkdir(exist_ok=True)
        output_path = Path(output_dir, "output_low_2.h5")
        prediction = pred_manager.predict_volume_to_path(output_path, Quality.LOW)
        assert len(list(output_dir.glob("*.h5"))) == 2
        assert output_path.exists()
        probs_path = Path(f"{output_path.parent / output_path.stem}_probs.h5")
        assert probs_path.exists()
        assert isinstance(prediction, np.ndarray)
        assert prediction.dtype == np.uint8

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_volume_to_path_medium(self, volseg_prediction_manager, tmp_path):
        output_dir = Path(tmp_path, "prediction_dir")
        output_dir.mkdir(exist_ok=True)
        output_path = Path(output_dir, "output_medium.h5")
        prediction = volseg_prediction_manager.predict_volume_to_path(
            output_path, Quality.MEDIUM
        )
        assert len(list(output_dir.glob("*.h5"))) == 1
        assert output_path.exists()
        assert isinstance(prediction, np.ndarray)
        assert prediction.dtype == np.uint8

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_volume_to_path_high(self, volseg_prediction_manager, tmp_path):
        output_dir = Path(tmp_path, "prediction_dir")
        output_dir.mkdir(exist_ok=True)
        output_path = Path(output_dir, "output_high.h5")
        prediction = volseg_prediction_manager.predict_volume_to_path(
            output_path, Quality.HIGH
        )
        assert len(list(output_dir.glob("*.h5"))) == 1
        assert output_path.exists()
        assert isinstance(prediction, np.ndarray)
        assert prediction.dtype == np.uint8
