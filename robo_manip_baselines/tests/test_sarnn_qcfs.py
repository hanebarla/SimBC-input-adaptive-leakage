import math
from types import SimpleNamespace

import pytest
import torch

from robo_manip_baselines.policy.sarnn import qcfs_layer
from robo_manip_baselines.policy.sarnn.LIFController import RuleBaseController
from robo_manip_baselines.policy.sarnn.QCFSSarnnPolicy import QCFSSarnnPolicy
from robo_manip_baselines.policy.sarnn.RolloutSarnn import RolloutSarnn
from robo_manip_baselines.policy.sarnn.SarnnPolicy import SarnnPolicy


def make_policy(qcfs_T=0, reset=True, spike_dec_type="default"):
    return QCFSSarnnPolicy(
        state_dim=7,
        num_images=1,
        image_size_list=[(16, 16)],
        num_attentions=3,
        lstm_hidden_dim=8,
        qcfs_L=8,
        qcfs_T=qcfs_T,
        reset=reset,
        spike_dec_type=spike_dec_type,
    )


def policy_inputs(batch_size=2):
    return torch.zeros(batch_size, 7), [torch.rand(batch_size, 3, 16, 16)]


def test_rule_controller_first_difference_clone_and_reset():
    controller = RuleBaseController(k=32.0, threshold=0.03)
    first = torch.zeros(1, 3, 4, 4, requires_grad=True)
    expected_first_alpha = 1.0 / (1.0 + math.exp(0.96))
    assert controller(first) == pytest.approx(expected_first_alpha)
    assert controller.latest_difference == 0.0
    assert controller.previous_input.grad_fn is None

    with torch.no_grad():
        first.add_(1.0)
    second = torch.zeros_like(first)
    controller(second)
    assert controller.latest_difference == 0.0

    controller.reset()
    assert controller.previous_input is None
    assert controller.differences == []
    assert controller.alphas == []


def test_ann_and_regular_sarnn_output_contracts():
    state, images = policy_inputs()
    qcfs_policy = make_policy(qcfs_T=0)
    regular_policy = SarnnPolicy(7, 1, [(16, 16)], 3, 8)

    qcfs_output = qcfs_policy(state, images)
    regular_output = regular_policy(state, images)
    assert qcfs_output[0].shape == regular_output[0].shape == (2, 7)
    assert qcfs_output[1][0].shape == regular_output[1][0].shape == (2, 3, 16, 16)
    assert qcfs_output[2][0].shape == regular_output[2][0].shape == (2, 3, 2)


def test_checkpoint_weights_are_shared_across_ann_and_snn_modes():
    ann_policy = make_policy(qcfs_T=0)
    snn_policy = make_policy(qcfs_T=4)
    assert ann_policy.state_dict().keys() == snn_policy.state_dict().keys()
    snn_policy.load_state_dict(ann_policy.state_dict(), strict=True)


def test_stateful_snn_propagates_alpha_and_resets_all_runtime_state():
    torch.manual_seed(0)
    policy = make_policy(qcfs_T=2, reset=False, spike_dec_type="decay")
    state, images = policy_inputs(batch_size=1)
    policy.update_LIF_param(0.37)

    first_output = policy(state, images)[0]
    if_layers = [
        module for module in policy.modules() if isinstance(module, qcfs_layer.IF)
    ]
    decay_decoders = [
        module
        for module in policy.modules()
        if isinstance(module, qcfs_layer.ST2DECAY_FR)
    ]
    assert len(if_layers) == 6
    assert len(decay_decoders) == 2
    assert all(layer.lif_alpha == pytest.approx(0.37) for layer in if_layers)
    assert all(decoder.alpha == pytest.approx(0.37) for decoder in decay_decoders)
    assert all(layer.mem is not None for layer in if_layers)
    assert all(decoder.prev_fr is not None for decoder in decay_decoders)

    policy(state, images)
    policy.reset_qcfs_state()
    assert all(layer.mem is None for layer in if_layers)
    assert all(decoder.prev_fr is None for decoder in decay_decoders)
    reset_output = policy(state, images)[0]
    torch.testing.assert_close(first_output, reset_output)


def test_rule_controller_rollout_configuration_is_validated():
    rollout = object.__new__(RolloutSarnn)
    rollout.args = SimpleNamespace(
        lif_alpha=0.0,
        qcfs_T=2,
        no_reset=False,
        lif_cnt="rule",
        spike_dec_type="decay",
    )
    with pytest.raises(ValueError, match="requires --no_reset"):
        rollout._validate_qcfs_rollout_args(True, 2)

    rollout.args.no_reset = True
    rollout.args.spike_dec_type = "default"
    with pytest.raises(ValueError, match="requires --spike_dec_type decay"):
        rollout._validate_qcfs_rollout_args(True, 2)

    rollout.args.spike_dec_type = "decay"
    rollout._validate_qcfs_rollout_args(True, 2)
