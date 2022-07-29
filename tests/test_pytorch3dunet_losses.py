"""
These tests adapted from Adrian Wolny's pytorch-3dunet 
repository
"""

import numpy as np
import torch
import torch.nn as nn
from volume_segmantics.data.pytorch3dunet_losses import (
    _MaskingLossWrapper,
    BCEDiceLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    SkipLastTargetChannelWrapper,
    WeightedSmoothL1Loss,
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


def test_generalized_dice_loss():
    results = _compute_criterion(GeneralizedDiceLoss())
    # check that all of the coefficients belong to [0, 1]
    results = np.array(results)
    assert np.all(results > 0)
    assert np.all(results < 1)


def test_dice_loss():
    results = _compute_criterion(DiceLoss())
    # check that all of the coefficients belong to [0, 1]
    results = np.array(results)
    assert np.all(results > 0)
    assert np.all(results < 1)


def test_bce_dice_loss():
    results = _compute_criterion(BCEDiceLoss(1.0, 1.0))
    results = np.array(results)
    assert np.all(results > 0)


def test_weighted_smooth_l1loss():
    loss_criterion = WeightedSmoothL1Loss(threshold=0.0, initial_weight=0.1)
    input = torch.randn(3, 16, 64, 64, 64, requires_grad=True)
    target = torch.randn(3, 16, 64, 64, 64)
    loss = loss_criterion(input, target)
    loss.backward()
    assert loss > 0


def test_ignore_index_loss():
    loss = _MaskingLossWrapper(nn.BCEWithLogitsLoss(), ignore_index=-1)
    input = torch.rand((3, 3))
    input[1, 1] = 1.0
    input.requires_grad = True
    target = -1.0 * torch.ones((3, 3))
    target[1, 1] = 1.0
    output = loss(input, target)
    output.backward()

def test_skip_last_target_channel():
    loss = SkipLastTargetChannelWrapper(nn.BCEWithLogitsLoss())
    input = torch.rand(1, 1, 3, 3, 3, requires_grad=True)
    target = torch.empty(1, 2, 3, 3, 3).random_(2)
    output = loss(input, target)
    output.backward()
    assert output.item() > 0
