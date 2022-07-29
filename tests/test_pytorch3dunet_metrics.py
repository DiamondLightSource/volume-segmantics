"""
These tests adapted from Adrian Wolny's pytorch-3dunet 
repository
"""

import numpy as np
import torch
from skimage import measure
from volume_segmantics.data.pytorch3dunet_metrics import (
    DiceCoefficient,
    MeanIoU,
)


def _eval_criterion(criterion, batch_shape, n_times=100):
    with torch.no_grad():
        results = []
        # compute criterion n_times
        for i in range(n_times):
            input = torch.rand(batch_shape)
            target = torch.zeros(batch_shape).random_(0, 2)
            output = criterion(input, target)
            results.append(output)

    return results


def _compute_criterion(criterion, n_times=100):
    shape = [1, 0, 30, 30, 30]
    # channel size varies between 1 and 4
    results = []
    for C in range(1, 5):
        batch_shape = list(shape)
        batch_shape[1] = C
        batch_shape = tuple(batch_shape)
        results.append(_eval_criterion(criterion, batch_shape, n_times))

    return results


def test_dice_coefficient():
    results = _compute_criterion(DiceCoefficient())
    # check that all of the coefficients belong to [0, 1]
    results = np.array(results)
    assert np.all(results > 0)
    assert np.all(results < 1)


def test_mean_iou_simple():
    results = _compute_criterion(MeanIoU())
    # check that all of the coefficients belong to [0, 1]
    results = np.array(results)
    assert np.all(results > 0)
    assert np.all(results < 1)


def test_mean_iou():
    criterion = MeanIoU()
    x = torch.randn(3, 3, 3, 3)
    _, index = torch.max(x, dim=0, keepdim=True)
    # create target tensor
    target = torch.zeros_like(x, dtype=torch.long).scatter_(0, index, 1)
    pred = torch.zeros_like(target, dtype=torch.float)
    mask = target == 1
    # create prediction tensor
    pred[mask] = torch.rand(1)
    # make sure the dimensions are right
    target = torch.unsqueeze(target, dim=0)
    pred = torch.unsqueeze(pred, dim=0)
    assert criterion(pred, target) == 1


def test_mean_iou_one_channel():
    criterion = MeanIoU()
    pred = torch.rand(1, 1, 3, 3, 3)
    target = pred > 0.5
    target = target.long()
    assert criterion(pred, target) == 1
