from volume_segmantics.model.model_2d import (
    create_model_from_file,
    create_model_on_device,
)
import pytest
import volume_segmantics.utilities.base_data_utils as utils
import torch


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_type",
    [
        utils.ModelType.U_NET,
        utils.ModelType.U_NET_PLUS_PLUS,
        utils.ModelType.FPN,
        utils.ModelType.DEEPLABV3,
        utils.ModelType.DEEPLABV3_PLUS,
        utils.ModelType.MA_NET,
        utils.ModelType.LINKNET,
    ],
)
def test_create_model_on_device(binary_model_struc_dict, model_type):
    binary_model_struc_dict["type"] = model_type
    model = create_model_on_device(0, binary_model_struc_dict)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.gpu
def test_create_model_from_file(model_path):
    model, classes, codes = create_model_from_file(model_path)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0
    assert isinstance(classes, int)
    assert isinstance(codes, dict)
