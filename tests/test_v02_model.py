"""CPU model-contract checks with a causal Mamba stand-in."""

import json
import math
import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.decoding import DecodingConfig
from dauys.model import DauysHybrid
from dauys.streaming import AudioSession
from dauys.v02_data import CorpusRecord, sha256
from dauys.v02_hohere_backend import HohereProbeBackend
from dauys.v02_runner import (
    TrainingPlan,
    _validate_evaluation,
    load_initial_weights,
    optimizer_groups_for_track,
    toggles_for_track,
    train,
)
from dauys.v02_training import Event, Toggles, V02Batch, objective, schedule


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        return self.projection(x).cumsum(dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen):
        width = self.projection.in_features
        return (torch.zeros(batch_size, 1, width), torch.zeros(1))

    def step(self, x, previous, unused):
        previous.add_(self.projection(x))
        return previous.clone(), previous, unused


@pytest.fixture(scope="module")
def model():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _CausalMamba
    old = sys.modules.get("mamba_ssm")
    sys.modules["mamba_ssm"] = module
    try:
        torch.manual_seed(5)
        config = replace(ModelConfig.v02_proxy(), dropout=0.0, attention_dropout=0.0)
        yield DauysHybrid(config)
    finally:
        if old is None:
            del sys.modules["mamba_ssm"]
        else:
            sys.modules["mamba_ssm"] = old


def _batch(source):
    torch.manual_seed(11)
    codes_a = torch.randint(0, 2048, (1, 8, 6))
    codes_b = torch.randint(0, 2048, (1, 8, 6))
    return V02Batch(
        user_codes=codes_a,
        output_codes=codes_b,
        user_activity=torch.tensor([[0, 1, 1, 0, 0, 0]]),
        output_activity=torch.tensor([[0, 0, 1, 1, 0, 0]]),
        valid_frames=torch.ones((1, 6), dtype=torch.bool),
        initial_output_codes=torch.zeros((1, 8), dtype=torch.long),
        silence_codes=torch.zeros((1, 8), dtype=torch.long),
        sources=(source,),
        conversation_ids=("one",),
        events=((),),
    )


def test_v02_future_input_cannot_change_past_outputs(model):
    model.eval()
    batch = _batch("hohere")
    changed = batch.user_codes.clone()
    changed[:, :, 4:] = torch.randint(0, 2048, (1, 8, 2))
    with torch.no_grad():
        first = model(batch.user_codes, batch.output_codes, batch.initial_output_codes)
        second = model(changed, batch.output_codes, batch.initial_output_codes)
    torch.testing.assert_close(first.cb0_logits[:, :4], second.cb0_logits[:, :4])
    torch.testing.assert_close(
        first.activity_ahead_logits[:, :4], second.activity_ahead_logits[:, :4]
    )
    torch.testing.assert_close(
        first.listening_cb0_logits[:, :4], second.listening_cb0_logits[:, :4]
    )


def test_v02_batched_and_incremental_logits_agree_without_regularizers(model):
    model.eval()
    batch = _batch("hohere")
    with torch.no_grad():
        batched = model(
            batch.user_codes, batch.output_codes, batch.initial_output_codes
        )
        state = model.new_stream(batch.initial_output_codes)
        cb0 = []
        activity = []
        for frame in range(batch.user_codes.shape[-1]):
            previous = (
                batch.initial_output_codes
                if frame == 0
                else batch.output_codes[:, :, frame - 1]
            )
            cb0_step, activity_step, _ = model.stream_step_teacher(
                batch.user_codes[:, :, frame], previous, state
            )
            cb0.append(cb0_step)
            activity.append(activity_step)
    torch.testing.assert_close(
        torch.stack(cb0, dim=1), batched.cb0_logits, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        torch.stack(activity, dim=1), batched.activity_logits, atol=1e-4, rtol=1e-4
    )


def test_hohere_e1_scores_real_model_logits(model, tmp_path):
    model.eval()
    backend = HohereProbeBackend(tmp_path, {}, object(), torch.zeros(8))
    backend._token_cache["probe"] = (
        torch.randint(0, 2048, (8, 3)),
        torch.randint(0, 2048, (8, 3)),
    )
    scores = backend.e1(model, {"id": "probe"})
    assert math.isfinite(scores["cb0_nll"])
    assert math.isfinite(scores["depth_nll"])
    assert set(scores["fusion_off_cb0_nll_by_block"]) == {
        str(index) for index in model.fusion_block_indices()
    }


def test_proxy_pretraining_does_not_require_two_party_gate():
    scores = {
        "eval/pretrain/cont_word_rate": 0.75,
        "eval/pretrain/cont_empty_rate": 0.1,
        "eval/codebook/cb0_accuracy_gap": 0.08,
    }
    _validate_evaluation(scores, "pretrain", "P")
    with pytest.raises(ValueError, match="heldout_1b_activity_f1"):
        _validate_evaluation(scores, "pretrain", "PRE")


def test_source_masks_isolate_gradients(model):
    model.eval()
    toggles = Toggles(
        regularization=False,
        history_dropout=False,
        acoustic_dropout=False,
        transition=False,
        contrast=False,
    )
    for source in ("1a_on_A", "1a_on_B"):
        model.zero_grad(set_to_none=True)
        loss = objective(
            model, _batch(source), "pretrain", schedule(0, "pretrain"), toggles
        )
        loss.total.backward()
        listening_grad = model.listening_projection.weight.grad
        dialogue_grad = model.dialogue_projection.weight.grad
        assert listening_grad is not None
        assert dialogue_grad is not None
        if source == "1a_on_A":
            assert listening_grad.abs().sum() > 0
            assert dialogue_grad.abs().sum() == 0
        else:
            assert listening_grad.abs().sum() == 0
            assert dialogue_grad.abs().sum() > 0


def test_shared_acoustic_keep_mask_survives_a_second_pass(model):
    model.train()
    try:
        codes = torch.randint(0, 2048, (2, 8, 5))
        keep = torch.rand(2, 5, 7) >= 0.5
        original = keep.clone()
        items = torch.tensor([True, False])
        first = model.encode_listening(codes, 0.5, items, keep)
        second = model.encode_listening(codes, 0.5, items, keep)
        (first.sum() + second.sum()).backward()
        assert torch.equal(keep, original)
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


def test_b_only_detach_preserves_silence_context_logits(model):
    model.eval()
    batch = _batch("1a_on_B")
    user = torch.zeros_like(batch.user_codes)
    with torch.no_grad():
        regular = model(user, batch.output_codes, batch.initial_output_codes)
        detached = model(
            user,
            batch.output_codes,
            batch.initial_output_codes,
            detach_listening=torch.tensor([True]),
        )
    torch.testing.assert_close(regular.cb0_logits, detached.cb0_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        regular.activity_logits, detached.activity_logits, rtol=0, atol=0
    )


def test_teacher_forced_fusion_ablation_selects_one_block(model):
    model.eval()
    batch = _batch("hohere")
    blocks = model.fusion_block_indices()
    assert len(blocks) >= 1
    with torch.no_grad():
        normal = model(batch.user_codes, batch.output_codes, batch.initial_output_codes)
        ablated = model(
            batch.user_codes,
            batch.output_codes,
            batch.initial_output_codes,
            fusion_off_blocks=frozenset({blocks[0]}),
        )
    assert not torch.equal(normal.cb0_logits, ablated.cb0_logits)
    with pytest.raises(ValueError, match="fusion blocks"):
        model(
            batch.user_codes,
            batch.output_codes,
            batch.initial_output_codes,
            fusion_off_blocks=frozenset({999}),
        )


def test_s0_logs_conditional_delta_without_optimizing_contrast(model):
    model.eval()
    batch = _batch("hohere")
    batch = replace(
        batch,
        negative_user_codes=(batch.user_codes + 1) % 2048,
        negative_ids=("different",),
        events=((Event("response_onset", "B", 2, 3),),),
    )
    scores = objective(
        model,
        batch,
        "dialogue",
        schedule(0, "dialogue"),
        toggles_for_track("S0", "dialogue"),
        measure_delta=True,
    )
    assert scores.delta_cb0_onset is not None
    assert scores.contrast.item() == 0.0


def test_s0_uses_baseline_decay_and_s1_uses_matrix_only_decay(model):
    baseline = optimizer_groups_for_track(model, toggles_for_track("S0", "dialogue"))
    regularized = optimizer_groups_for_track(model, toggles_for_track("S1", "dialogue"))
    assert len(baseline) == 1 and baseline[0]["weight_decay"] == 0.01
    assert {group["weight_decay"] for group in regularized} == {0.0, 0.05}


def test_eval_disables_acoustic_dropout_and_label_smoothing(model):
    model.eval()
    batch = _batch("hohere")
    with torch.no_grad():
        baseline = model(
            batch.user_codes, batch.output_codes, batch.initial_output_codes
        )
        acoustic = model(
            batch.user_codes,
            batch.output_codes,
            batch.initial_output_codes,
            acoustic_dropout=0.5,
        )
        regularized_loss = objective(
            model,
            batch,
            "dialogue",
            schedule(0, "dialogue"),
            Toggles(
                history_dropout=False,
                acoustic_dropout=False,
                transition=False,
                contrast=False,
            ),
        )
        plain_loss = objective(
            model,
            batch,
            "dialogue",
            schedule(0, "dialogue"),
            Toggles(
                regularization=False,
                history_dropout=False,
                acoustic_dropout=False,
                transition=False,
                contrast=False,
            ),
        )
    torch.testing.assert_close(baseline.cb0_logits, acoustic.cb0_logits)
    torch.testing.assert_close(regularized_loss.total, plain_loss.total)


def test_all_greedy_decoding_matches_legacy_generation(model):
    model.eval()
    batch = _batch("hohere")
    first_state = model.new_stream(batch.initial_output_codes)
    second_state = model.new_stream(batch.initial_output_codes)
    with torch.no_grad():
        legacy = model.generate_step(batch.user_codes[:, :, 0], first_state)
        configured = model.generate_step(
            batch.user_codes[:, :, 0], second_state, DecodingConfig()
        )
    torch.testing.assert_close(legacy[0], configured[0])
    torch.testing.assert_close(legacy[1].argmax(dim=-1).bool(), configured[1])


def test_generate_step_returns_threshold_decision(model):
    model.eval()
    batch = _batch("hohere")
    always_on = replace(
        DecodingConfig(), activity_mode="hysteresis", theta_on=0.0, theta_off=0.0
    )
    always_off = replace(
        DecodingConfig(), activity_mode="hysteresis", theta_on=1.0, theta_off=1.0
    )
    with torch.no_grad():
        on = model.generate_step(
            batch.user_codes[:, :, 0],
            model.new_stream(batch.initial_output_codes),
            always_on,
        )[1]
        off = model.generate_step(
            batch.user_codes[:, :, 0],
            model.new_stream(batch.initial_output_codes),
            always_off,
        )[1]
    assert on.item() is True
    assert off.item() is False


def test_streaming_rejects_training_mode(model):
    model.train()
    with pytest.raises(ValueError, match="model.eval"), AudioSession(model, None, 1):
        pass
    model.eval()


def test_regularization_toggle_changes_dropout_without_changing_weights(model):
    model.config = replace(model.config, dropout=0.1, attention_dropout=0.1)
    baseline = {name: value.clone() for name, value in model.state_dict().items()}
    model.set_regularization(False)
    assert model.frame_dropout.p == 0.0
    model.set_regularization(True)
    assert model.frame_dropout.p == 0.1
    assert all(
        torch.equal(baseline[name], value) for name, value in model.state_dict().items()
    )
    model.set_regularization(False)


def test_local_training_step_saves_scored_checkpoint(model, tmp_path):
    records = []
    for index, skill in enumerate(("simple_dialogue", "interruption")):
        batch = _batch("hohere")
        path = tmp_path / f"example_{index}.pt"
        torch.save(
            {
                "user": batch.user_codes[0],
                "target": batch.output_codes[0],
                "user_activity": batch.user_activity[0],
                "activity": batch.output_activity[0],
            },
            path,
        )
        records.append(
            CorpusRecord(
                id=f"example_{index}",
                source="hohere",
                skill=skill,
                family=f"family_{index}",
                user_voice=f"voice_{index}",
                frames=6,
                token_path=path,
                token_sha256=sha256(path),
                activity_a=tuple(batch.user_activity[0].tolist()),
                activity_b=tuple(batch.output_activity[0].tolist()),
                events=(),
            )
        )
    plan = TrainingPlan(
        track="FT",
        stage="dialogue",
        steps=1,
        batch_size=2,
        learning_rate=2e-5,
        checkpoint_interval=1,
        num_workers=0,
    )
    result = train(
        model,
        records,
        torch.zeros(8, dtype=torch.long),
        plan,
        tmp_path / "run",
        evaluator=lambda _model, step: {"eval/step": float(step)},
        allow_unscored_smoke=True,
    )
    saved = torch.load(result, map_location="cpu", weights_only=True)
    assert saved["step"] == 1
    assert saved["evaluation"] == {"eval/step": 1.0}
    metrics = json.loads(
        (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0]
    )
    groups = [
        value
        for key, value in metrics.items()
        if key.startswith("train/gradient_norm/")
    ]
    assert math.sqrt(sum(value * value for value in groups)) == pytest.approx(
        metrics["train/gradient_norm"], rel=1e-4
    )


def test_resume_restores_checkpoint_and_removes_uncommitted_metric_rows(
    model, tmp_path
):
    records = []
    for index, skill in enumerate(("simple_dialogue", "interruption")):
        batch = _batch("hohere")
        path = tmp_path / f"resume_{index}.pt"
        torch.save(
            {
                "user": batch.user_codes[0],
                "target": batch.output_codes[0],
                "user_activity": batch.user_activity[0],
                "activity": batch.output_activity[0],
            },
            path,
        )
        records.append(
            CorpusRecord(
                id=f"resume_{index}",
                source="hohere",
                skill=skill,
                family=f"family_{index}",
                user_voice=f"voice_{index}",
                frames=6,
                token_path=path,
                token_sha256=sha256(path),
                activity_a=tuple(batch.user_activity[0].tolist()),
                activity_b=tuple(batch.output_activity[0].tolist()),
                events=(),
            )
        )
    plan = TrainingPlan(
        track="FT",
        stage="dialogue",
        steps=3,
        batch_size=2,
        learning_rate=2e-5,
        checkpoint_interval=1,
        num_workers=0,
    )
    output = tmp_path / "resumed-run"

    def interrupt(_model, step):
        if step == 2:
            raise RuntimeError("interrupted")
        return {"eval/step": float(step)}

    with pytest.raises(RuntimeError, match="interrupted"):
        train(
            model,
            records,
            torch.zeros(8, dtype=torch.long),
            plan,
            output,
            evaluator=interrupt,
            allow_unscored_smoke=True,
        )
    with pytest.raises(FileExistsError):
        train(
            model,
            records,
            torch.zeros(8, dtype=torch.long),
            plan,
            output,
            evaluator=interrupt,
            allow_unscored_smoke=True,
        )
    result = train(
        model,
        records,
        torch.zeros(8, dtype=torch.long),
        plan,
        output,
        evaluator=lambda _model, step: {"eval/step": float(step)},
        allow_unscored_smoke=True,
        resume=True,
    )
    assert torch.load(result, weights_only=False)["step"] == 3
    metrics = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in metrics if "train/total" in row] == [1, 2, 3]

    sft_plan = TrainingPlan(
        track="SFT",
        stage="posttrain",
        steps=4,
        batch_size=1,
        learning_rate=2e-5,
        checkpoint_interval=1,
        num_workers=0,
        stage2_peak_learning_rate=2e-4,
        early_stop_patience=1,
    )
    sft_output = tmp_path / "early-stopped-run"
    evaluator = lambda _model, _step: {"eval/codebook/e1_cb0_nll": 1.0}
    stopped = train(
        model,
        records,
        torch.zeros(8, dtype=torch.long),
        sft_plan,
        sft_output,
        evaluator=evaluator,
        allow_unscored_smoke=True,
    )
    saved = torch.load(stopped, weights_only=False)
    assert saved["step"] == 2 and saved["stopped_early"]
    before = (sft_output / "metrics.jsonl").read_bytes()
    same = train(
        model,
        records,
        torch.zeros(8, dtype=torch.long),
        sft_plan,
        sft_output,
        evaluator=evaluator,
        allow_unscored_smoke=True,
        resume=True,
    )
    assert same == stopped
    assert (sft_output / "metrics.jsonl").read_bytes() == before


def test_v01_weights_load_with_only_new_heads_missing(model, tmp_path):
    old_config = replace(
        ModelConfig.v02_proxy(),
        name="dauys-hybrid-v0.1",
        dropout=0.0,
        attention_dropout=0.0,
        listening_prediction=False,
        activity_ahead_horizon=0,
    )
    old_model = DauysHybrid(old_config)
    path = tmp_path / "r0.pt"
    torch.save({"model": old_model.state_dict()}, path)
    load_initial_weights(model, path, from_v01=True)
    torch.testing.assert_close(model.cb0_head.weight, old_model.cb0_head.weight)


def test_durable_link_copies_when_hard_links_are_unsupported(tmp_path, monkeypatch):
    from dauys import v02_runner

    source = tmp_path / "latest.pt"
    source.write_bytes(b"weights")

    def refuse(*_args):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(v02_runner.os, "link", refuse)
    v02_runner._durable_link(source, tmp_path / "best.pt")
    assert (tmp_path / "best.pt").read_bytes() == b"weights"
