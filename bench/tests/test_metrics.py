"""CPU checks for the listening bench arithmetic and test-set readers."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from duplexbench import data, metrics, run


def test_partners_are_never_self_and_similar_in_length():
    lengths = [100, 500, 120, 480, 110]
    partners = metrics.length_matched_partners(lengths)
    assert all(partner != index for index, partner in enumerate(partners))
    assert sorted(partners) == list(range(5))
    assert lengths[partners[0]] in (110, 120)


def test_fit_length_crops_and_pads_with_silence():
    codes = torch.arange(8 * 3).reshape(8, 3)
    silence = torch.full((8,), 7)
    assert metrics.fit_length(codes, 2, silence).shape == (8, 2)
    padded = metrics.fit_length(codes, 5, silence)
    assert padded.shape == (8, 5) and (padded[:, 3:] == 7).all()


def test_echo_rate_counts_heard_ngrams_and_skips_ignored():
    heard = [1, 2, 3, 4, 5, 6]
    assert metrics.echo_rate([1, 2, 3, 4], heard, n=4) == 1.0
    assert metrics.echo_rate([9, 9, 9, 9], heard, n=4) == 0.0
    assert metrics.echo_rate([1, 2, 3, 4, 9], heard, n=4) == 0.5
    ignored = frozenset({(1, 2, 3, 4)})
    assert metrics.echo_rate([1, 2, 3, 4], heard, n=4, ignore=ignored) != 1.0


def test_echo_does_not_bridge_pauses():
    tokens = torch.tensor([1, 2, 3, 0, 0, 4, 5, 6])
    mask = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1]).bool()
    spoken = run._speech_tokens(tokens, mask)
    assert spoken == [1, 2, 3, -1, 4, 5, 6]
    assert run._echo([1, 2, 3, -1, 4, 5, 6], [1, 2, 3, 9], 3) == 0.5


def test_pair_accuracy_and_bootstrap():
    assert metrics.pair_accuracy([2.0, 1.0, 3.0], [1.0, 1.0, 4.0]) == 0.5
    ci = metrics.bootstrap_ci([1.0] * 10 + [0.0] * 10, reps=500)
    assert ci["n"] == 20 and ci["low"] < 0.5 < ci["high"]
    assert metrics.bootstrap_ci([float("nan")])["n"] == 0


def test_slm21_pairs_group_by_id_and_voice(tmp_path):
    split = tmp_path / "lexical" / "dev"
    split.mkdir(parents=True)
    (split / "gold.csv").write_text(
        "id,filename,voice,frequency,word,phones,length,correct\n"
        "1,aaa,A,2,obscenely,X,7,1\n"
        "1,bbb,A,,opsenely,X,7,0\n"
        "1,ccc,C,2,obscenely,X,7,1\n"
    )
    pairs = data.slm21_pairs(tmp_path, "lexical")
    assert len(pairs) == 1
    assert pairs[0]["good"].name == "aaa.wav" and pairs[0]["bad"].name == "bbb.wav"


def test_single_stream_prefers_the_cheaper_item():
    class Fake:
        silence = torch.zeros(8, dtype=torch.long)

        def frame_nll(self, user, own):
            nll = own[:, 0].float()  # token value as cost: lower codes score higher
            return nll, nll

    pairs = [
        {"good": torch.ones(8, 4, dtype=torch.long), "bad": torch.full((8, 4), 3)},
        {"good": torch.ones(8, 2, dtype=torch.long), "bad": torch.full((8, 6), 2)},
    ]
    result = run.single_stream(Fake(), pairs, batch=3)
    assert result["accuracy_sum"] == 1.0 and result["pairs"] == 2
