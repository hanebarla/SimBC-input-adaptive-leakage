import math

import pytest
import torch

from eipl.layer import qcfs_layer
from eipl.model import SpikingSARNN
from robo_manip_baselines.sarnn.lib.LifController import RuleBaseController


def test_rule_controller_first_frame_known_difference_and_reset():
    controller = RuleBaseController(k=32.0, threshold=0.03)
    first = torch.zeros(1, 3, 4, 4)
    assert controller(first) == pytest.approx(1.0 / (1.0 + math.exp(0.96)))
    assert controller.latest_difference == 0.0

    second = torch.full_like(first, 0.1)
    assert controller(second) == pytest.approx(1.0 / (1.0 + math.exp(-2.24)))
    assert controller.latest_difference == pytest.approx(0.1)
    assert all(0.0 <= alpha <= 1.0 for alpha in controller.alphas)

    controller.reset()
    controller(second)
    assert controller.latest_difference == 0.0
    assert len(controller.alphas) == 1


def test_alpha_is_propagated_to_all_if_layers_and_decay_decoders():
    model = SpikingSARNN(
        rec_dim=8,
        k_dim=3,
        joint_dim=7,
        im_size=[16, 16],
        qcfs_L=8,
        qcfs_T=2,
        reset=False,
        spike_dec_type="decay",
    )
    model.update_LIF_param(0.37)

    if_layers = [
        layer
        for encoder in (model.im_encoder, model.pos_encoder)
        for layer in encoder
        if isinstance(layer, qcfs_layer.IF)
    ]
    assert len(if_layers) == 6
    assert all(layer.lif_alpha == pytest.approx(0.37) for layer in if_layers)
    assert model.spike_dec.alpha == pytest.approx(0.37)
    assert model.ss_spike_dec.alpha == pytest.approx(0.37)
