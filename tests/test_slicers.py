import numpy as np
from volume_segmantics.data import TrainingDataSlicer
import pytest
from skimage import io


@pytest.fixture()
def training_data_slicer(data_vol, label_vol, training_settings, request):
    data_vol = request.getfixturevalue(data_vol)
    label_vol = request.getfixturevalue(label_vol)
    return TrainingDataSlicer(data_vol, label_vol, training_settings)


class TestTrainingDataSlicer:
    @pytest.mark.parametrize(
        "data_vol, label_vol",
        [
            ("rand_int_volume", "rand_label_volume"),
            ("rand_int_volume", "rand_label_tiff_path"),
            ("rand_int_volume", "rand_label_hdf5_path"),
            ("rand_int_hdf5_path", "rand_label_volume"),
            ("rand_int_hdf5_path", "rand_label_tiff_path"),
            ("rand_int_hdf5_path", "rand_label_hdf5_path"),
            ("rand_int_tiff_path", "rand_label_hdf5_path"),
            ("rand_int_tiff_path", "rand_label_tiff_path"),
            ("rand_int_tiff_path", "rand_label_volume"),
        ],
    )
    def test_training_data_slicer_init(self, training_data_slicer):
        assert isinstance(training_data_slicer, TrainingDataSlicer)
        assert isinstance(training_data_slicer.seg_vol, np.ndarray)
        assert training_data_slicer.codes is not None

    def test_training_data_slicer_fix_labels(
        self, rand_int_volume, rand_label_volume_no_zeros, training_settings
    ):
        assert np.unique(rand_label_volume_no_zeros)[0] != 0
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume_no_zeros, training_settings
        )
        assert np.unique(slicer.seg_vol)[0] == 0

    def test_training_data_slicer_output_data(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "im_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        file_list = list(im_dir_path.glob("*.png"))
        assert len(file_list) != 0
        img = io.imread(file_list[0])
        assert isinstance(img, np.ndarray)
        assert np.issubdtype(img.dtype, np.integer)        

