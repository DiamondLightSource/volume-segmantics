import numpy as np
import pytest
import torch
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor


@pytest.fixture()
def volseg_2d_predictor(model_path, prediction_settings):
    return VolSeg2dPredictor(model_path, prediction_settings)


class TestVolseg2DPredictor:
    @pytest.mark.gpu
    def test_2d_predictor_init(self, volseg_2d_predictor):
        assert isinstance(volseg_2d_predictor, VolSeg2dPredictor)
        assert isinstance(volseg_2d_predictor.model, torch.nn.Module)
        assert isinstance(volseg_2d_predictor.num_labels, int)
        assert isinstance(volseg_2d_predictor.label_codes, dict)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis(self, volseg_2d_predictor, rand_int_volume):
        labels, probs = volseg_2d_predictor._predict_single_axis(
            rand_int_volume, output_probs=False
        )
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert probs is None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis_probs(self, volseg_2d_predictor, rand_int_volume):
        labels, probs = volseg_2d_predictor._predict_single_axis(
            rand_int_volume, output_probs=True
        )
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_3_ways_max_probs(self, volseg_2d_predictor, rand_int_volume):
        labels, probs = volseg_2d_predictor._predict_3_ways_max_probs(rand_int_volume)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_12_ways_max_probs(self, volseg_2d_predictor, rand_int_volume):
        labels, probs = volseg_2d_predictor._predict_12_ways_max_probs(rand_int_volume)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis_one_hot(self, volseg_2d_predictor, rand_int_volume):
        counts = volseg_2d_predictor._predict_single_axis_to_one_hot(rand_int_volume)
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.uint8
        assert counts.ndim == 4  # one hot output should be 4 dimensional

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_3_axis_one_hot(self, volseg_2d_predictor, rand_int_volume):
        counts = volseg_2d_predictor._predict_3_ways_one_hot(rand_int_volume)
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.uint8
        assert counts.ndim == 4  # one hot output should be 4 dimensional

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_12_ways_one_hot(self, volseg_2d_predictor, rand_int_volume):
        counts = volseg_2d_predictor._predict_12_ways_one_hot(rand_int_volume)
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.uint8
        assert counts.ndim == 4  # one hot output should be 4 dimensional
