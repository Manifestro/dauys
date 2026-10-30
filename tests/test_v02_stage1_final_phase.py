"""Stage 1 final phase: the batch's speaker mix changes for the last steps."""

from pathlib import Path

import pytest
import torch

from dauys.v02_data import sha256
from dauys.v02_stage1 import (
    Stage1BatchSource,
    Stage1Record,
    ab_layout,
    make_batch_spec,
)

PER_SPEAKER = 4


def _record(
    root: Path, identity: str, speaker: str, frames: int = 128, cut: int = 64
) -> Stage1Record:
    torch.manual_seed(abs(hash(identity)) % (2**31))
    codes = torch.randint(0, 2048, (8, frames))
    activity = torch.ones(frames, dtype=torch.long)
    activity[cut : cut + 3] = 0
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


def _corpus(root: Path, prefix: str, speakers: int) -> list[Stage1Record]:
    return [
        _record(root, f"{prefix}{speaker}-{index}", f"{prefix}{speaker}")
        for speaker in range(speakers)
        for index in range(PER_SPEAKER)
    ]


@pytest.fixture
def corpora(tmp_path):
    return (
        _corpus(tmp_path, "m", 4),
        _corpus(tmp_path, "a", 3),
        _corpus(tmp_path, "b", 3),
    )


def _source(corpora, *, steps=5, start=0, final_start=None, final_mix=(), speakers=4):
    main, first, second = corpora
    return Stage1BatchSource(
        main,
        torch.zeros(8, dtype=torch.long),
        steps=steps,
        start=start,
        seed=0,
        speakers=speakers,
        anchors=1,
        pool=PER_SPEAKER,
        crop_frames=64,
        mix=[(first, 1), (second, 1)],
        final_start=final_start,
        final_mix=final_mix,
    )


def test_mix_counts_switch_at_the_final_start(corpora):
    source = _source(corpora, final_start=3, final_mix=(2, 1))
    assert [source.mix_counts(step) for step in range(3)] == [[1, 1]] * 3
    assert [source.mix_counts(step) for step in range(3, 6)] == [[2, 1]] * 3


def test_final_mix_leaves_the_rest_to_the_main_corpus(corpora):
    source = _source(corpora, final_start=2, final_mix=(2, 1))
    assert len(source.mix_counts(0)) == 2
    assert source.speakers - sum(source.mix_counts(0)) == 2
    assert source.speakers - sum(source.mix_counts(2)) == 1


def test_final_start_uses_the_step_not_the_index(corpora):
    # Steps 3 and 4 remain, so every index of a source starting at 3 is final.
    source = _source(corpora, start=3, final_start=3, final_mix=(2, 1))
    assert len(source) == 2
    assert [source.mix_counts(3 + index) for index in range(2)] == [[2, 1], [2, 1]]


def test_batches_before_and_after_the_final_start(corpora):
    source = _source(corpora, final_start=2, final_mix=(2, 1))
    before = source[0]
    before.validate()
    speakers = list(before.speakers)
    assert sum(s.startswith("a") for s in speakers) == 1
    assert sum(s.startswith("b") for s in speakers) == 1
    assert sum(s.startswith("m") for s in speakers) == 2
    after = source[2]
    after.validate()
    speakers = list(after.speakers)
    assert sum(s.startswith("a") for s in speakers) == 2
    assert sum(s.startswith("b") for s in speakers) == 1
    assert sum(s.startswith("m") for s in speakers) == 1
    assert after.user_codes.shape[0] == source.speakers * source.anchors


def test_no_final_phase_matches_the_old_behaviour(corpora):
    main, first, second = corpora
    plain = Stage1BatchSource(
        main,
        torch.zeros(8, dtype=torch.long),
        steps=3,
        seed=0,
        speakers=4,
        anchors=1,
        pool=PER_SPEAKER,
        crop_frames=64,
        mix=[(first, 1), (second, 1)],
    )
    explicit = _source(corpora, steps=3, final_start=None, final_mix=())
    for index in range(3):
        assert explicit.mix_counts(index) == plain.mix_counts(index) == [1, 1]
        left = plain[index]
        right = explicit[index]
        left.validate()
        right.validate()
        assert left.speakers == right.speakers
        assert left.ids == right.ids
        assert torch.equal(left.user_codes, right.user_codes)
        assert torch.equal(left.output_codes, right.output_codes)
        assert torch.equal(left.candidate_codes, right.candidate_codes)
        assert torch.equal(left.positive_index, right.positive_index)


def test_a_final_start_at_zero_is_all_final(corpora):
    source = _source(corpora, steps=2, final_start=0, final_mix=(2, 1))
    assert [source.mix_counts(step) for step in range(2)] == [[2, 1], [2, 1]]


def test_final_start_at_the_end_keeps_every_normal_step(corpora):
    source = _source(corpora, steps=3, final_start=3, final_mix=(2, 1))
    assert [source.mix_counts(step) for step in range(3)] == [[1, 1]] * 3


def test_the_main_corpus_keeps_the_remaining_speakers(corpora):
    source = _source(corpora, final_start=1, final_mix=(2, 1))
    batch = source[1]
    batch.validate()
    assert batch.speakers[0].startswith("m")
    assert sum(s.startswith("a") for s in batch.speakers) == 2
    assert sum(s.startswith("b") for s in batch.speakers) == 1
    # The normal counts would give the main corpus two groups instead of one.
    normal_main = make_batch_spec(
        source.records[: source.main_count],
        1,
        seed=0,
        speakers=source.speakers - sum(source.mix_counts(0)),
        anchors=1,
        pool=PER_SPEAKER,
    )
    assert len(normal_main) == 2
    final_main = make_batch_spec(
        source.records[: source.main_count],
        1,
        seed=0,
        speakers=source.speakers - sum(source.mix_counts(1)),
        anchors=1,
        pool=PER_SPEAKER,
    )
    assert len(final_main) == 1
    assert {source.records[index].speaker for index in final_main[0][0]} == {
        batch.speakers[0]
    }


def test_wrong_final_mix_length_is_rejected(corpora):
    with pytest.raises(ValueError, match="one final count per mixed corpus"):
        _source(corpora, final_start=2, final_mix=(2,))


def test_final_mix_without_a_start_is_rejected(corpora):
    with pytest.raises(ValueError, match="final phase start"):
        _source(corpora, final_start=None, final_mix=(2, 1))


@pytest.mark.parametrize("counts", [(0, 1), (2, 0), (-1, 2)])
def test_final_counts_must_be_positive(corpora, counts):
    with pytest.raises(ValueError, match="at least 1"):
        _source(corpora, final_start=1, final_mix=counts)


@pytest.mark.parametrize("counts", [(2, 2), (3, 1), (4, 2)])
def test_final_counts_must_leave_speakers_for_the_main_corpus(corpora, counts):
    with pytest.raises(ValueError, match="leave speakers"):
        _source(corpora, final_start=1, final_mix=counts)


@pytest.mark.parametrize("final_start", [-1, 6])
def test_final_start_must_lie_inside_the_plan(corpora, final_start):
    with pytest.raises(ValueError, match="final phase must start"):
        _source(corpora, steps=5, final_start=final_start, final_mix=(2, 1))
