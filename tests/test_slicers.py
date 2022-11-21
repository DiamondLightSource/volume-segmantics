import numpy as np
from volume_segmantics.data import TrainingDataSlicer
import volume_segmantics.utilities.base_data_utils as utils
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
        # Check the label values are sequential
        values = np.unique(slicer.seg_vol)
        assert np.where(np.diff(values) != 1)[0].size == 0

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

    def test_training_data_slicer_output_data_all_axes(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "im_out"
        if hasattr(training_settings, "training_axis"):
            delattr(training_settings, "training_axis")
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        file_list = list(im_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == sum(rand_int_volume.shape)

    def test_training_data_slicer_output_data_single_axis(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "im_out"
        training_settings.training_axes = "y"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        file_list = list(im_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == rand_int_volume.shape[1]

    def test_training_data_slicer_output_labels(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "label_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        assert len(file_list) != 0
        img = io.imread(file_list[0])
        assert isinstance(img, np.ndarray)
        assert np.issubdtype(img.dtype, np.integer)

    def test_training_data_slicer_output_labels_all_axes(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "label_out"
        if hasattr(training_settings, "training_axis"):
            delattr(training_settings, "training_axis")
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == sum(rand_int_volume.shape)

    def test_training_data_slicer_output_labels_single_axis(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "label_out"
        training_settings.training_axes = "x"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == rand_int_volume.shape[2]

    def test_training_data_slicer_output_binary_labels(
        self, rand_int_volume, rand_binary_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "binary_label_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_binary_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        for fn in file_list:
            img = io.imread(fn)
            assert isinstance(img, np.ndarray)
            assert np.issubdtype(img.dtype, np.integer)
            assert np.array_equal(np.unique(img), np.array([0, 1]))

    def test_training_data_slicer_clean_up(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "temp_im_out"
        label_dir_path = empty_dir / "temp_label_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        slicer.output_label_slices(label_dir_path, "seg")
        im_file_list = list(im_dir_path.glob("*.png"))
        label_file_list = list(label_dir_path.glob("*.png"))
        assert len(im_file_list) != 0
        assert len(label_file_list) != 0
        slicer.clean_up_slices()
        assert not im_dir_path.exists()
        assert not label_dir_path.exists()
