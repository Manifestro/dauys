"""Stage 1 data layout, speech masks, same-speaker batching and intent loss."""

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_data import sha256
from dauys.v02_stage1 import (
    INTENT_FRAMES,
    Stage1Batch,
    Stage1BatchSource,
    Stage1Record,
    Stage1Weights,
    _hide_history,
    _load_tokens,
    ab_layout,
    assert_stage1_pools,
    continuation_window,
    crop_around_cut,
    find_cut,
    infonce,
    load_batch,
    load_stage1_corpus,
    make_batch_spec,
    spread_score,
    stage1_masks,
    stage1_objective,
    stage1_weights,
)
from dauys.v03_intent import IntentConfig, IntentModule


class _CausalMamba(nn.Module):
    def __init__(self, d_model, layer_idx=None, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        return self.projection(x)

    def allocate_inference_cache(self, batch_size, max_seqlen):
        width = self.projection.in_features
        return (torch.zeros(batch_size, 1, width), torch.zeros(1))

    def step(self, x, previous, unused):
        return self.projection(x), previous, unused


@pytest.fixture(autouse=True)
def stand_in_mamba():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _CausalMamba
    old = sys.modules.get("mamba_ssm")
    sys.modules["mamba_ssm"] = module
    try:
        yield
    finally:
        if old is None:
            del sys.modules["mamba_ssm"]
        else:
            sys.modules["mamba_ssm"] = old


def _activity(frames: int, pause: tuple[int, int]) -> torch.Tensor:
    activity = torch.ones(frames, dtype=torch.long)
    activity[pause[0] : pause[1]] = 0
    return activity


def test_find_cut_requires_speech_on_both_sides_of_a_pause():
    activity = [1] * 40 + [0] * 5 + [1] * 30
    assert find_cut(activity, min_before=38, min_after=25) == 40
    assert find_cut(activity, min_before=41, min_after=25) is None
    assert find_cut(activity, min_before=38, min_after=31) is None
    # A two-frame pause is too short.
    assert find_cut([1] * 40 + [0, 0] + [1] * 30, min_before=38, min_after=25) is None
    assert find_cut([1] * 5 + [0] * 5 + [1] * 5, min_before=38, min_after=25) is None


def test_ab_layout_keeps_the_pause_on_b_and_speech_disjoint():
    frames = 60
    codes = torch.arange(8 * frames).reshape(8, frames)
    activity = _activity(frames, (30, 34))
    silence = torch.full((8,), 7)
    layout = ab_layout(codes, activity, 30, silence)
    assert torch.equal(layout.user[:, :30], codes[:, :30])
    assert torch.equal(layout.user[:, 30:], silence[:, None].expand(8, 30))
    assert torch.equal(layout.target[:, :30], silence[:, None].expand(8, 30))
    assert torch.equal(layout.target[:, 30:], codes[:, 30:])
    assert layout.user_activity.tolist()[:30] == [1] * 30
    assert layout.user_activity.tolist()[30:] == [0] * 30
    assert layout.target_activity.tolist()[:30] == [0] * 30
    assert layout.target_activity.tolist()[30] == 0
    # Frames 0..29 only live on A, frames 30..59 only live on B.
    assert not set(layout.user[:, :30].flatten().tolist()) & set(
        layout.target[:, 30:].flatten().tolist()
    )


def test_crop_around_cut_preserves_both_context_windows():
    start, end = crop_around_cut(1000, 500, 256, 7, min_before=38, min_after=25)
    assert end - start == 256
    assert 500 - start >= 38
    assert end - 500 >= 25
    assert (start, end) == crop_around_cut(
        1000, 500, 256, 7, min_before=38, min_after=25
    )
    assert crop_around_cut(100, 50, 256, 0, min_before=38, min_after=25) == (0, 100)


def test_stage1_masks_use_speech_only_for_cb0_and_all_frames_for_activity():
    valid = torch.ones((1, 4), dtype=torch.bool)
    output = torch.tensor([[0, 1, 1, 0]])
    user = torch.tensor([[1, 1, 0, 0]])
    masks = stage1_masks(valid, output, user)
    assert masks.cb0.tolist() == [[False, True, True, False]]
    assert masks.activity.tolist() == [[True, True, True, True]]
    assert masks.listening.tolist() == [[True, False, False]]


def test_stage1_weights_follow_the_pre_registered_schedule():
    assert stage1_weights(0.0).listening == pytest.approx(1.0)
    assert stage1_weights(0.4).listening == pytest.approx(0.5)
    assert stage1_weights(1.0).listening == pytest.approx(0.3)
    for progress in (0.0, 0.5, 1.0):
        weights = stage1_weights(progress)
        assert (weights.cb0, weights.depth, weights.activity) == (1.0, 0.5, 0.2)
        assert weights.intent == 0.5


def test_infonce_prefers_the_matching_candidate():
    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).reshape(2, 1, 2)
    candidates = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    target = torch.zeros((2, 1), dtype=torch.long)
    loss, accuracy = infonce(prediction, candidates, target)
    assert float(accuracy) == 1.0
    assert float(loss) < 0.1


def _record(
    root: Path, identity: str, speaker: str, frames: int, cut: int
) -> Stage1Record:
    codes = torch.randint(0, 2048, (8, frames))
    activity = _activity(frames, (cut, cut + 3))
    layout = ab_layout(codes, activity, cut, torch.zeros(8, dtype=torch.long))
    path = root / f"{identity}.pt"
    torch.save(
        {
            "user": layout.user.short(),
            "target": layout.target.short(),
            "user_activity": layout.user_activity.to(torch.uint8),
            "activity": layout.target_activity.to(torch.uint8),
        },
        path,
    )
    return Stage1Record(
        id=identity,
        speaker=speaker,
        frames=frames,
        cut_frame=cut,
        token_path=path,
        token_sha256=sha256(path),
    )


def test_same_speaker_batches_keep_candidates_in_one_voice(tmp_path):
    torch.manual_seed(0)
    records = []
    for speaker in ("s1", "s2", "s3"):
        for index in range(8):
            records.append(_record(tmp_path, f"{speaker}-{index}", speaker, 128, 64))
    spec = make_batch_spec(records, 0, seed=3, speakers=2, anchors=2, pool=4)
    assert len(spec) == 2
    assert all(len(pool) == 4 and len(anchors) == 2 for pool, anchors in spec)
    batch = load_batch(records, spec, torch.zeros(8, dtype=torch.long), crop_frames=64)
    batch.validate()
    assert batch.user_codes.shape == (4, 8, 64)
    assert batch.candidate_codes.shape == (2, 4, 8, INTENT_FRAMES)
    assert batch.speaker_index.tolist() == [0, 0, 1, 1]
    for group, (pool, anchors) in enumerate(spec):
        speakers = {records[index].speaker for index in pool}
        assert len(speakers) == 1
        for local, position in enumerate(anchors):
            assert int(batch.positive_index[group, local]) == position
    # A ends before its cut and B starts after it: no source frame is shared.
    for group, (_, anchors) in enumerate(spec):
        for local, _ in enumerate(anchors):
            row = group * 2 + local
            cut = int(batch.intent_frames[row]) + 1
            assert not batch.user_codes[row, :, cut:].any()
            assert not batch.output_codes[row, :, :cut].any()
    assert all(len(set(ids)) == 4 for ids in batch.candidate_ids)


def test_continuation_window_skips_leading_pause():
    target = torch.arange(8 * 20).reshape(8, 20)
    activity = torch.tensor([0, 0, 0] + [1] * 17)
    window, mask = continuation_window(target, activity, 0, length=5)
    assert torch.equal(window, target[:, 3:8])
    assert mask.all()


def test_stage1_config_uses_the_widened_ear():
    config = ModelConfig.v02_stage1()
    assert (config.listening_dim, config.listening_layers) == (512, 6)
    assert (config.dialogue_dim, config.dialogue_d_state) == (768, 128)
    assert (config.depth_layers, config.depth_dim, config.depth_heads) == (4, 512, 8)
    assert config.proxy and config.zero_init_fusion_outputs
    assert config.self_hearing == "none"


def test_load_stage1_corpus_verifies_checksums(tmp_path):
    record = _record(tmp_path, "a", "s1", 128, 64)
    manifest = {
        "id": record.id,
        "speaker": record.speaker,
        "frames": record.frames,
        "cut_frame": record.cut_frame,
        "tokens_path": record.token_path.name,
        "sha256": record.token_sha256,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    loaded = load_stage1_corpus(tmp_path, tmp_path / "manifest.jsonl")
    assert [(item.id, item.speaker, item.cut_frame) for item in loaded] == [
        (record.id, record.speaker, record.cut_frame)
    ]
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({**manifest, "sha256": "0" * 64}) + "\n"
    )
    with pytest.raises(ValueError, match="checksum"):
        load_stage1_corpus(tmp_path, tmp_path / "manifest.jsonl")


def _tiny_config() -> ModelConfig:
    return replace(
        ModelConfig.v02_proxy(),
        embedding_dim=64,
        listening_dim=64,
        listening_layers=1,
        dialogue_dim=64,
        dialogue_blocks=("mamba2", "transformer_cross"),
        attention_heads=4,
        ffn_dim=128,
        depth_dim=64,
        depth_layers=1,
        depth_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        activity_ahead_horizon=0,
        max_trainable_parameters=10_000_000,
    )


def test_stage1_objective_trains_dialogue_and_intent_heads():
    torch.manual_seed(0)
    model = DauysHybrid(_tiny_config()).train()
    intent = IntentModule(model, IntentConfig(horizon=2))
    frames = 12
    batch = Stage1Batch(
        user_codes=torch.randint(0, 2048, (4, 8, frames)),
        output_codes=torch.randint(0, 2048, (4, 8, frames)),
        user_activity=torch.ones((4, frames), dtype=torch.long),
        output_activity=torch.ones((4, frames), dtype=torch.long),
        valid_frames=torch.ones((4, frames), dtype=torch.bool),
        initial_output_codes=torch.zeros((4, 8), dtype=torch.long),
        silence_codes=torch.zeros((4, 8), dtype=torch.long),
        intent_frames=torch.tensor([3, 5, 7, 9]),
        candidate_codes=torch.randint(0, 2048, (2, 4, 8, INTENT_FRAMES)),
        candidate_activity=torch.ones((2, 4, INTENT_FRAMES), dtype=torch.bool),
        positive_index=torch.tensor([[0, 1], [2, 3]]),
        speaker_index=torch.tensor([0, 0, 1, 1]),
        ids=("a", "b", "c", "d"),
        speakers=("s1", "s2"),
        candidate_ids=(("a", "b", "c", "d"), ("e", "f", "g", "h")),
    )
    losses = stage1_objective(model, batch, Stage1Weights(acoustic_dropout=0.0), intent)
    assert torch.isfinite(losses.total)
    assert float(losses.intent_accuracy) >= 0.0
    losses.total.backward()
    assert intent.head.net[0].weight.grad is not None
    assert model.dialogue_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in intent.target.parameters())
    model.zero_grad()
    far = stage1_objective(
        model,
        batch,
        Stage1Weights(acoustic_dropout=0.0, listening_horizon=4, history_dropout=1.0),
        intent,
    )
    assert torch.isfinite(far.total)
    assert not torch.isclose(far.listening_cb0, losses.listening_cb0)


def test_find_cut_can_skip_a_long_pause():
    activity = [1] * 40 + [0] * 40 + [1] * 30 + [0] * 5 + [1] * 30
    assert find_cut(activity, min_before=38, min_after=25) == 40
    assert find_cut(activity, min_before=38, min_after=25, max_pause=10) == 110


def test_pool_preflight_rejects_too_few_speakers(tmp_path):
    records = [
        _record(tmp_path, f"s{speaker}-{index}", f"s{speaker}", 128, 64)
        for speaker in range(2)
        for index in range(8)
    ]
    assert_stage1_pools(records, speakers=2, pool=8)
    with pytest.raises(ValueError, match="need 3"):
        assert_stage1_pools(records, speakers=3, pool=8)


def test_find_cut_seeded_choice_is_reproducible():
    activity = [1] * 40 + [0] * 5 + [1] * 30 + [0] * 5 + [1] * 30
    assert find_cut(activity, min_before=38, min_after=25) == 40
    chosen = find_cut(activity, min_before=38, min_after=25, seed=5)
    assert chosen in {40, 75}
    assert chosen == find_cut(activity, min_before=38, min_after=25, seed=5)


def test_make_batch_spec_prefers_one_chapter(tmp_path):
    records = [
        replace(
            _record(tmp_path, f"s1-{chapter}-{index:04d}", "s1", 128, 64),
            chapter=f"s1-{chapter}",
        )
        for chapter in (1, 2)
        for index in range(8)
    ]
    spec = make_batch_spec(
        records, 0, seed=1, speakers=1, anchors=2, pool=4, chapter_only=True
    )
    pool, _ = spec[0]
    assert len({records[index].chapter for index in pool}) == 1


def test_chapter_only_rejects_records_without_chapter_pools(tmp_path):
    records = [
        replace(_record(tmp_path, f"s{speaker}-0-0000", f"s{speaker}", 128, 64))
        for speaker in range(4)
    ]
    with pytest.raises(ValueError, match="speaker-chapter"):
        make_batch_spec(
            records, 0, seed=1, speakers=1, anchors=1, pool=2, chapter_only=True
        )


def test_swap_user_changes_a_and_keeps_b(tmp_path):
    records = [
        replace(
            _record(tmp_path, f"s1-1-{index:04d}", "s1", 128, 64),
            chapter="s1-1",
        )
        for index in range(4)
    ]
    spec = [(tuple(range(4)), (0, 1))]
    silence = torch.zeros(8, dtype=torch.long)
    normal = load_batch(records, spec, silence, crop_frames=64)
    swapped = load_batch(records, spec, silence, crop_frames=64, swap_user=True)
    assert torch.equal(normal.output_codes, swapped.output_codes)
    assert not torch.equal(normal.user_codes, swapped.user_codes)
    for row in range(2):
        cut = int(swapped.intent_frames[row]) + 1
        assert not swapped.user_codes[row, :, cut:].any()
        assert not swapped.user_activity[row, cut:].any()


def test_pool_preflight_counts_speaker_chapters(tmp_path):
    records = [
        replace(
            _record(tmp_path, f"s1-{chapter}-{index:04d}", "s1", 128, 64),
            chapter=f"s1-{chapter}",
        )
        for chapter in (1, 2)
        for index in range(8)
    ]
    assert_stage1_pools(records, speakers=2, pool=8, chapter_only=True)
    with pytest.raises(ValueError, match="speaker-chapter"):
        assert_stage1_pools(records, speakers=3, pool=8, chapter_only=True)


def test_tokens_without_activity_are_rejected(tmp_path):
    path = tmp_path / "plain.pt"
    torch.save(
        {
            "user": torch.zeros((8, 12), dtype=torch.short),
            "target": torch.zeros((8, 12), dtype=torch.short),
        },
        path,
    )
    with pytest.raises(ValueError, match="activity"):
        _load_tokens(path)


def test_spread_score_is_scale_normalized():
    torch.manual_seed(0)
    healthy = torch.randn(64, 512)
    assert 0.7 < spread_score(healthy) < 1.3
    collapsed = torch.ones(64, 512) + 1e-3 * torch.randn(64, 512)
    assert spread_score(collapsed) < 0.05


def test_batch_source_start_offset_skips_early_steps(tmp_path):
    records = []
    for speaker in ("s1", "s2", "s3"):
        for index in range(8):
            records.append(_record(tmp_path, f"{speaker}-{index}", speaker, 128, 64))
    source = Stage1BatchSource(
        records,
        torch.zeros(8, dtype=torch.long),
        steps=5,
        start=3,
        seed=0,
        speakers=2,
        anchors=1,
        pool=4,
        crop_frames=64,
    )
    assert len(source) == 2
    batch = source[0]
    batch.validate()
    assert batch.user_codes.shape[0] == 2


def test_hidden_history_silences_only_b_after_the_cut(tmp_path):
    torch.manual_seed(0)
    records = []
    for speaker in ("s1", "s2"):
        for index in range(8):
            records.append(_record(tmp_path, f"{speaker}-{index}", speaker, 128, 64))
    spec = make_batch_spec(records, 0, seed=3, speakers=2, anchors=2, pool=4)
    silence = torch.full((8,), 7, dtype=torch.long)
    batch = load_batch(records, spec, silence, crop_frames=64)
    kept, none_hidden = _hide_history(batch, 0.0)
    assert kept is batch.output_codes and not none_hidden.any()
    inputs, hidden = _hide_history(batch, 1.0)
    assert hidden.all()
    for row in range(inputs.shape[0]):
        cut = int(batch.intent_frames[row]) + 1
        assert torch.equal(inputs[row, :, :cut], batch.output_codes[row, :, :cut])
        assert bool((inputs[row, :, cut:] == 7).all())
