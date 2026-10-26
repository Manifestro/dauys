"""Architecture and temporal-contract checks for Hybrid v0.1."""

import pytest
import torch

pytest.importorskip("mamba_ssm")

from dauys.config import ModelConfig
from dauys.loss import training_loss
from dauys.model import DauysHybrid


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("The Mamba-2 integration checks require CUDA")
    torch.manual_seed(17)
    config = ModelConfig.stage1()
    instance = DauysHybrid(config).cuda().eval()
    return instance


def _codes(frames: int = 6):
    user = torch.randint(0, 2048, (1, 8, frames), device="cuda")
    output = torch.randint(0, 2048, (1, 8, frames), device="cuda")
    initial = torch.randint(0, 2048, (1, 8), device="cuda")
    return user, output, initial


def test_parameter_budget_and_shapes(model):
    user, output, initial = _codes()
    with torch.inference_mode():
        result = model(user, output, initial)
    assert model.trainable_parameters() <= 65_000_000
    assert result.cb0_logits.shape == (1, 6, 2048)
    assert result.depth_logits.shape == (1, 6, 7, 2048)
    assert result.activity_logits.shape == (1, 6, 2)


def test_future_audio_cannot_change_past_logits(model):
    user, output, initial = _codes()
    changed_user = user.clone()
    changed_output = output.clone()
    changed_user[:, :, 4:] = torch.randint(0, 2048, (1, 8, 2), device="cuda")
    changed_output[:, :, 4:] = torch.randint(0, 2048, (1, 8, 2), device="cuda")
    with torch.inference_mode():
        baseline = model(user, output, initial)
        changed = model(changed_user, changed_output, initial)
    torch.testing.assert_close(
        baseline.cb0_logits[:, :4], changed.cb0_logits[:, :4], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        baseline.depth_logits[:, :4], changed.depth_logits[:, :4],
        rtol=1e-4, atol=1e-4,
    )


def test_listening_phase_predicts_future_user_codes(model):
    user, output, initial = _codes()
    changed_user = user.clone()
    changed_user[:, :, 4:] = torch.randint(0, 2048, (1, 8, 2), device="cuda")
    with torch.inference_mode():
        baseline = model(user, output, initial, initial_user_codes=initial)
        changed = model(changed_user, output, initial, initial_user_codes=initial)
    assert baseline.user_cb0_logits is not None
    assert baseline.user_depth_logits is not None
    assert changed.user_cb0_logits is not None
    torch.testing.assert_close(
        baseline.user_cb0_logits[:, :4], changed.user_cb0_logits[:, :4],
        rtol=1e-4, atol=1e-4,
    )
    assert baseline.user_depth_logits.shape == (1, 6, 7, 2048)


def test_teacher_stream_matches_batched_forward(model):
    user, output, initial = _codes()
    with torch.inference_mode():
        batched = model(user, output, initial)
        state = model.new_stream(initial)
        cb0_steps = []
        activity_steps = []
        for frame in range(user.shape[-1]):
            previous = initial if frame == 0 else output[:, :, frame - 1]
            cb0, activity, _ = model.stream_step_teacher(
                user[:, :, frame], previous, state
            )
            cb0_steps.append(cb0)
            activity_steps.append(activity)
    torch.testing.assert_close(
        torch.stack(cb0_steps, dim=1), batched.cb0_logits, rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        torch.stack(activity_steps, dim=1), batched.activity_logits,
        rtol=2e-3, atol=2e-3,
    )


def test_simultaneous_input_and_output_preserves_session_state(model):
    user, _, initial = _codes(frames=3)
    with torch.inference_mode():
        state = model.new_stream(initial)
        for frame in range(3):
            generated, activity = model.generate_step(user[:, :, frame], state)
            assert generated.shape == (1, 8)
            assert activity.shape == (1, 2)
            assert state.frame_index == frame + 1
        assert len(state.listening_history) == 3


def test_training_loss_backpropagates(model):
    user, output, initial = _codes(frames=2)
    model.train()
    result = model(user, output, initial)
    losses = training_loss(
        result, output, torch.ones(1, 2, device="cuda", dtype=torch.long),
        torch.ones(1, 2, device="cuda", dtype=torch.bool), model.config,
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.cb0_head.weight.grad is not None
    assert model.depth.heads[0].weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.eval()
