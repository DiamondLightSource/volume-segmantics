import pytest
import volume_segmantics.data.augmentations as augs
import albumentations as A


@pytest.mark.parametrize(
    "test_input,expected", [(32, 32), (64, 64), (33, 64), (13, 32), (0, 0)]
)
def test_get_padded_dimension_some_vals(test_input, expected):
    assert augs.get_padded_dimension(test_input) == expected


def test_other_funcs_return_compose():
    assert isinstance(augs.get_train_preprocess_augs(256), A.Compose)
    assert isinstance(augs.get_pred_preprocess_augs(32, 64), A.Compose)
    assert isinstance(augs.get_train_augs(128), A.Compose)
    assert isinstance(augs.get_postprocess_augs(), A.Compose)
