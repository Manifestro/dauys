"""CPU contract checks for the V10 step 1 model, data and estimators."""

import importlib.util
import itertools
import json
import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from hohere.ready import check
from hohere.records import write_record
from hohere.shards import ShardWriter
from hohere.units import build_m
from torch import nn

from dauys.v10_step1 import (
    V10Step1Config,
    V10Step1Model,
)
from dauys.v10_step1_data import (
    ARM_FILES,
    PackedCorpus,
    V10BatchSource,
    chunk_plan,
    fetch_tier,
    pack_tier,
    read_tier,
)
from dauys.v10_step1_eval import (
    LEAD_FRAMES,
    exact_log_ps_pad,
    joint_log_pms,
    log_marginal,
    marginal_log_ps,
    nll_s_given_true_m,
    normalized_scores,
    pair_marginal_log_ps,
    pair_outcomes,
    stability,
)


class _CausalMamba(nn.Module):
    """A cumulative-sum stand-in with Mamba-2's streaming state protocol."""

    def __init__(self, d_model, layer_idx=None, **_kwargs):
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


@pytest.fixture(scope="module", autouse=True)
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


def _config(**changes) -> V10Step1Config:
    base = replace(
        V10Step1Config.v10_step1_tiny(),
        width=64,
        repeats=1,
        units=8,
        attention_heads=4,
        ffn_dim=160,
        attention_window=32,
        dropout=0.0,
        attention_dropout=0.0,
        max_trainable_parameters=10_000_000,
    )
    return replace(base, **changes)


def _model(config: V10Step1Config, seed: int = 3) -> V10Step1Model:
    torch.manual_seed(seed)
    return V10Step1Model(config).eval()


@pytest.fixture(scope="module")
def model():
    return _model(_config())


def _tokens(config: V10Step1Config, batch: int, frames: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    m = torch.randint(0, config.units, (batch, frames), generator=generator)
    s = torch.randint(0, 2048, (batch, frames), generator=generator)
    return m, s


def test_tiny_config_has_the_planned_block_pattern():
    config = V10Step1Config.v10_step1_tiny()
    model = V10Step1Model(config)
    kinds = model.block_kinds()
    assert len(kinds) == 6 * config.repeats
    assert kinds == tuple(
        "mamba2" if index % 6 < 5 else "attention" for index in range(len(kinds))
    )
    assert config.m_vocab_size == config.units + 3
    assert (config.sil_id, config.pad_id, config.m_bos_id) == (
        config.units,
        config.units + 1,
        config.units + 2,
    )
    assert config.s_vocab_with_bos == 2049 and config.s_bos_id == 2048


def test_planned_configs_load():
    big = V10Step1Config.v10_step1()
    assert (big.width, big.repeats, big.units) == (1024, 3, 8192)
    assert big.attention_window == 1000
    assert big.m_vocab_size == 8195 and big.s_vocab_with_bos == 2049
    tiny = V10Step1Config.v10_step1_tiny()
    assert tiny.name == "dauys-v1.0-step1-tiny" and tiny.repeats >= 1


def test_forward_shapes_and_learns_both_streams(model):
    m, s = _tokens(model.config, 2, 5)
    lengths = torch.tensor([5, 3])
    output = model(m, s)
    assert output.m_logits.shape == (2, 5, model.config.m_vocab_size)
    assert output.s_logits.shape == (2, 5, model.config.s_vocab_with_bos)
    losses = model.loss(output, m, s, lengths)
    assert torch.isfinite(losses.total)
    assert losses.total.item() == pytest.approx((losses.m + losses.s).item(), rel=1e-6)


def test_future_frames_cannot_change_past_hidden(model):
    m, s = _tokens(model.config, 2, 8, seed=1)
    changed_m, changed_s = m.clone(), s.clone()
    changed_m[:, 4:] = (m[:, 4:] + 1) % model.config.units
    changed_s[:, 4:] = (s[:, 4:] + 7) % 2048
    baseline = model(m, s)
    changed = model(changed_m, changed_s)
    torch.testing.assert_close(baseline.hidden[:, :4], changed.hidden[:, :4])
    torch.testing.assert_close(baseline.m_logits[:, :4], changed.m_logits[:, :4])
    torch.testing.assert_close(baseline.s_logits[:, :4], changed.s_logits[:, :4])


def test_streaming_matches_forward(model):
    m, s = _tokens(model.config, 2, 6, seed=2)
    output = model(m, s)
    state = model.init_state(2)
    m_prev = torch.full((2,), model.config.m_bos_id, dtype=torch.long)
    s_prev = torch.full((2,), model.config.s_bos_id, dtype=torch.long)
    for frame in range(m.shape[1]):
        h, state = model.step(state, m_prev, s_prev)
        torch.testing.assert_close(
            model.head_m(h), output.m_logits[:, frame], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            model.head_s(h + model.emb_m(m[:, frame])),
            output.s_logits[:, frame],
            atol=1e-5,
            rtol=1e-5,
        )
        m_prev, s_prev = m[:, frame], s[:, frame]


def test_streaming_respects_the_attention_window(model):
    config = replace(model.config, attention_window=2)
    shortened = _model(config)
    m, s = _tokens(config, 1, 7, seed=3)
    output = shortened(m, s)
    state = shortened.init_state(1)
    m_prev = torch.full((1,), config.m_bos_id, dtype=torch.long)
    s_prev = torch.full((1,), config.s_bos_id, dtype=torch.long)
    for frame in range(m.shape[1]):
        h, state = shortened.step(state, m_prev, s_prev)
        torch.testing.assert_close(
            shortened.head_m(h), output.m_logits[:, frame], atol=1e-5, rtol=1e-5
        )
        m_prev, s_prev = m[:, frame], s[:, frame]


def test_overfit_cuts_both_cross_entropies():
    config = _config(units=16, width=64)
    model = _model(config, seed=7)
    m, s = _tokens(config, 1, 24, seed=5)
    lengths = torch.tensor([24])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = model.loss(model(m, s), m, s, lengths)
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(model(m, s), m, s, lengths)
        losses.total.backward()
        optimizer.step()
    last = model.loss(model(m, s), m, s, lengths)
    assert last.m < 0.6 * first.m
    assert last.s < 0.6 * first.s
    assert (first.m - last.m) > 0.3 and (first.s - last.s) > 0.3


def test_one_seed_gives_equal_initial_weights():
    config = _config(units=12)
    first = _model(config, seed=11)
    second = _model(config, seed=11)
    left = first.state_dict()
    right = second.state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items())


def _record(frames: int, seed: int, units=(), sil=()) -> dict:
    """One chunk as hohere stores it; ``units`` and ``sil`` are 50 Hz frames."""
    generator = np.random.default_rng(seed)
    starts = np.asarray(units, dtype=np.int32)
    return {
        "codes": generator.integers(0, 2048, (32, frames)),
        "unit_start": starts,
        "unit_end": starts + 2,
        "unit_k4096": generator.integers(0, 4096, len(starts)),
        "unit_k8192": generator.integers(0, 8192, len(starts)),
        "sil_start": np.asarray(sil, dtype=np.int32),
    }


# Shards are not sorted by rank, numbers repeat across parts, and part-002 holds
# only a later tier. In a-000 the SIL shares Mimi frame 10 with a unit.
SHARDS = {
    "part-000": (
        ("b-001", 1, _record(155, 2, units=(8, 80))),
        ("a-000", 0, _record(40, 0, units=(40, 44), sil=(40,))),
    ),
    "part-001": (("b-000", 1, _record(700, 1, units=(4, 2000), sil=(400,))),),
    "part-002": (("c-000", 2, _record(64, 3)),),
}
TIER_FRAMES = {"a-000": 40, "b-000": 700, "b-001": 155}


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "shards"
    manifests = []
    for part, rows in SHARDS.items():
        writer = ShardWriter(root / "librilight" / part)
        for identity, rank, record in rows:
            meta = {
                "frames": record["codes"].shape[1],
                "order_rank": rank,
                "duration_s": record["codes"].shape[1] / 12.5,
                "units_k4096": "hash4096",
                "units_k8192": "hash8192",
            }
            writer.add(identity, write_record(**record), meta)
        writer.close()
        manifest = root / "librilight" / part / "shard-000000.jsonl"
        lines = [json.loads(line) for line in manifest.read_text().splitlines()]
        manifests.append((part, lines))
    report = check("500", [("a", 1, 0), ("b", 2, 1)], manifests)
    assert report["ready"]
    (root / "librilight" / "READY-500.json").write_text(json.dumps(report))
    return root


def _packed(shards, tmp_path, config=None, arm="B"):
    config = config or _config(units=4096)
    pack_tier(shards, "500", tmp_path / "packed", units=config.units, workers=1)
    return PackedCorpus.open(tmp_path / "packed", config, arm)


def test_pack_stores_what_hohere_builds_in_order_rank_order(shards, tmp_path):
    config = _config(units=4096)
    corpus = _packed(shards, tmp_path, config)
    b_sil = PackedCorpus.open(tmp_path / "packed", config, "B-sil")
    meta = json.loads((tmp_path / "packed" / "meta.json").read_text())
    assert meta["ids"] == ["a-000", "b-000", "b-001"]
    assert corpus.frames.tolist() == [40, 700, 155]
    assert meta["centroids_sha256"] == "hash4096" and meta["tier"] == "500"
    records = {row[0]: row[2] for rows in SHARDS.values() for row in rows}
    begin = 0
    for identity, frames in TIER_FRAMES.items():
        record = records[identity]
        end = begin + frames
        assert corpus.s[begin:end].tolist() == record["codes"][0].tolist()
        assert corpus.m[begin:end].tolist() == build_m(record, "B", 4096).tolist()
        assert b_sil.m[begin:end].tolist() == build_m(record, "B-sil", 4096).tolist()
        begin += frames


def test_b_sil_keeps_the_pause_frame_that_units_move_in_b(shards, tmp_path):
    config = _config(units=4096)
    b = _packed(shards, tmp_path, config).m.long()
    b_sil = PackedCorpus.open(tmp_path / "packed", config, "B-sil").m.long()
    assert b[11] == config.sil_id and b[10] < config.units
    assert b_sil[10] == config.sil_id and b_sil[11] == config.pad_id
    assert int((b_sil < config.units).sum()) == 0
    meta = json.loads((tmp_path / "packed" / "meta.json").read_text())
    assert (meta["sil_b_sil"], meta["sil_same_frame_in_b"]) == (2, 1)


def test_arm_a_has_no_file_and_sees_only_pad(shards, tmp_path):
    config = _config(units=4096)
    _packed(shards, tmp_path, config)
    corpus = PackedCorpus.open(tmp_path / "packed", config, "A")
    assert corpus.m is None
    source = V10BatchSource(
        corpus, config, chunk_frames=16, steps=1, batch_size=2, seed=4
    )
    assert int(source[0]["m"].min()) == int(source[0]["m"].max()) == config.pad_id
    assert sorted(ARM_FILES) == ["B", "B-sil"]


def test_m_only_sees_stream_b_and_no_s(shards, tmp_path):
    config = _config(units=4096)
    b = _packed(shards, tmp_path, config)
    only = PackedCorpus.open(tmp_path / "packed", config, "m-only")
    assert only.s is None and torch.equal(only.m.long(), b.m.long())
    kwargs = {"chunk_frames": 16, "steps": 2, "batch_size": 2, "seed": 4}
    batch = V10BatchSource(only, config, **kwargs)[0]
    other = V10BatchSource(b, config, **kwargs)[0]
    assert torch.equal(batch["m"], other["m"])
    assert int(batch["s"].min()) == int(batch["s"].max()) == config.s_bos_id
    model = _model(config)
    output = model(batch["m"], batch["s"])
    losses = model.loss(output, batch["m"], batch["s"], batch["lengths"], use_s=False)
    assert losses.total.item() == pytest.approx(losses.m.item())


def test_m_loss_is_also_reported_by_token_kind(model):
    config = model.config
    m = torch.tensor([[0, config.sil_id, config.pad_id, config.pad_id, 1, 2]])
    _, s = _tokens(config, 1, 6, seed=3)
    output = model(m, s)
    values = -torch.log_softmax(output.m_logits, dim=-1).gather(-1, m[..., None])[
        ..., 0
    ]
    losses = model.loss(output, m, s, torch.tensor([5]))
    assert losses.m_unit.item() == pytest.approx(values[0, [0, 4]].mean().item())
    assert losses.m_sil.item() == pytest.approx(values[0, 1].item())
    assert losses.m_pad.item() == pytest.approx(values[0, [2, 3]].mean().item())
    no_sil = model.loss(output, m, s, torch.tensor([1]))
    assert no_sil.m_sil.item() == 0.0


def test_packed_corpus_rejects_another_vocabulary_or_arm(shards, tmp_path):
    _packed(shards, tmp_path)
    with pytest.raises(ValueError, match="unit vocabulary"):
        PackedCorpus.open(tmp_path / "packed", _config(units=8192), "B")
    with pytest.raises(ValueError, match="arm"):
        PackedCorpus.open(tmp_path / "packed", _config(units=4096), "C")


def test_tier_needs_its_ready_report(shards):
    with pytest.raises(FileNotFoundError, match="not ready"):
        read_tier(shards, "5000", 8192)
    with pytest.raises(ValueError, match="4096 or 8192"):
        read_tier(shards, "500", 6000)


def test_tier_rejects_mixed_centroids(shards):
    manifest = shards / "librilight" / "part-001" / "shard-000000.jsonl"
    manifest.write_text(manifest.read_text().replace("hash8192", "other"))
    with pytest.raises(ValueError, match="centroids"):
        read_tier(shards, "500", 8192)
    assert len(read_tier(shards, "500", 4096)[1]) == 3


def test_tier_notices_a_missing_manifest(shards):
    (shards / "librilight" / "part-001" / "shard-000000.jsonl").unlink()
    with pytest.raises(ValueError, match="lists 3 chunks, found 2"):
        read_tier(shards, "500", 8192)


def test_fetch_downloads_only_the_shards_of_the_tier(shards, monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, **kwargs: calls.append(
        kwargs["allow_patterns"]
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    fetch_tier("org/repo", "500", shards, units=8192)
    assert calls == [
        ["librilight/READY-500.json", "librilight/*.jsonl"],
        [
            "librilight/part-000/shard-000000.tar",
            "librilight/part-001/shard-000000.tar",
        ],
    ]


def _script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def stimuli(tmp_path):
    """One sWUGGY pair: the labels start 0.24 s into each file, as ``label`` trims."""
    root = tmp_path / "slm21"
    split = root / "lexical" / "dev"
    split.mkdir(parents=True)
    (split / "gold.csv").write_text(
        "id,voice,correct,filename,type\n1,v,1,aaa,x\n1,v,0,bbb,x\n"
    )
    labels = tmp_path / "labels"
    (labels / "lexical" / "dev").mkdir(parents=True)
    rows = []
    for stem in ("aaa", "bbb"):
        source = f"lexical/dev/{stem}"
        record = _record(20, 0, units=(28,), sil=(28,))
        (labels / f"{source}.npz").write_bytes(write_record(**record))
        rows.append(
            {
                "id": source,
                "source": source,
                "start_s": 0.24,
                "n_units": 1,
                "units_k4096": "hash4096",
            }
        )
    (labels / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    codes = torch.zeros(8, 30, dtype=torch.long)
    torch.save(
        [{"id": "1-v", "kind": "x", "good": codes, "bad": codes}], cache / "swuggy.pt"
    )
    packed = tmp_path / "packed"
    packed.mkdir()
    (packed / "meta.json").write_text(
        json.dumps({"units": 4096, "centroids_sha256": "hash4096"})
    )
    return root, labels, cache, packed


def test_stimulus_m_moves_to_the_cached_frames_for_each_variant(stimuli):
    root, labels, cache, packed = stimuli
    report = _script("v10_step1_labels.py").align_test(
        "swuggy", root, labels, cache, packed
    )
    assert report == {"test": "swuggy", "pairs": 1, "stimuli_without_units": 0}
    config = _config(units=4096)
    b = torch.load(cache / "swuggy_m_B.pt", weights_only=True)[0]["good"]
    b_sil = torch.load(cache / "swuggy_m_B-sil.pt", weights_only=True)[0]["bad"]
    assert b.shape == b_sil.shape == (30,)
    # 50 Hz frame 28 + 12 of the trim = 40, Mimi frame 10; the unit keeps it in B.
    assert b[10] < config.units and b[11] == config.sil_id
    assert b_sil[10] == config.sil_id and int((b_sil < config.units).sum()) == 0


def test_label_cache_holds_the_pairs_on_labels_own_codes(stimuli, tmp_path):
    root, labels, cache, packed = stimuli
    torch.save(torch.zeros(8, dtype=torch.long), cache / "silence.pt")
    out = tmp_path / "label-cache"
    _script("v10_step1_labels.py").align_test(
        "swuggy", root, labels, cache, packed, out
    )
    rows = torch.load(out / "swuggy.pt", weights_only=True)
    expected = _record(20, 0, units=(28,), sil=(28,))["codes"][:8]
    assert rows[0]["id"] == "1-v" and rows[0]["good"].tolist() == expected.tolist()
    assert (out / "silence.pt").is_file()


def test_stimulus_alignment_refuses_what_would_not_line_up(stimuli):
    root, labels, cache, packed = stimuli
    script = _script("v10_step1_labels.py")
    good = torch.load(cache / "swuggy.pt", weights_only=True)
    short = [{**good[0], "good": good[0]["good"][:, :5]}]
    torch.save(short, cache / "swuggy.pt")
    with pytest.raises(ValueError, match="outside 5 frames"):
        script.align_test("swuggy", root, labels, cache, packed)
    torch.save([{**good[0], "id": "other"}], cache / "swuggy.pt")
    with pytest.raises(ValueError, match="different pairs"):
        script.align_test("swuggy", root, labels, cache, packed)
    torch.save(good, cache / "swuggy.pt")
    (packed / "meta.json").write_text(
        json.dumps({"units": 4096, "centroids_sha256": "other"})
    )
    with pytest.raises(ValueError, match="centroids"):
        script.align_test("swuggy", root, labels, cache, packed)


def test_each_arm_reads_its_own_stimulus_labels(tmp_path):
    script = _script("v10_step1_eval.py")
    torch.save([{"good": torch.zeros(3)}], tmp_path / "swuggy_m_B-sil.pt")
    assert script._sidecar(tmp_path, "swuggy", "A", 0) is None
    assert script._sidecar(tmp_path, "swuggy", "B", 0) is None
    assert len(script._sidecar(tmp_path, "swuggy", "B-sil", 0)) == 1


def test_epoch_covers_recordings_once_without_crossing_them():
    frames = torch.tensor([40, 700, 155])
    starts, lengths = chunk_plan(
        frames, chunk_frames=64, steps=12, batch_size=1, seed=1
    )
    ends = frames.cumsum(0)
    begins = ends - frames
    seen = torch.zeros(int(frames.sum()), dtype=torch.long)
    for start, length in zip(starts.flatten().tolist(), lengths.flatten().tolist()):
        owner = int(torch.searchsorted(ends, torch.tensor(start), right=True))
        assert begins[owner] <= start and start + length <= ends[owner]
        seen[start : start + length] += 1
    assert int(seen.max()) == 1
    assert int(seen.sum()) > 0.8 * int(frames.sum())


def test_chunk_order_is_identical_across_arms(shards, tmp_path):
    config = _config(units=4096)
    _packed(shards, tmp_path, config)
    batches = {
        arm: V10BatchSource(
            PackedCorpus.open(tmp_path / "packed", config, arm),
            config,
            chunk_frames=16,
            steps=6,
            batch_size=2,
            seed=4,
        )
        for arm in ("A", "B-sil", "B")
    }
    assert torch.equal(batches["A"].starts, batches["B"].starts)
    assert torch.equal(batches["B-sil"].starts, batches["B"].starts)
    different = V10BatchSource(
        PackedCorpus.open(tmp_path / "packed", config, "B"),
        config,
        chunk_frames=16,
        steps=6,
        batch_size=2,
        seed=5,
    )
    assert not torch.equal(batches["B"].starts, different.starts)
    assert batches["B"][0]["m"].shape == batches["B"][0]["s"].shape == (2, 16)
    for arm in ("A", "B-sil"):
        for index in range(6):
            assert torch.equal(batches[arm][index]["s"], batches["B"][index]["s"])
            assert torch.equal(
                batches[arm][index]["lengths"], batches["B"][index]["lengths"]
            )


def test_batch_source_resumes_on_the_same_plan(shards, tmp_path):
    config = _config(units=4096)
    corpus = _packed(shards, tmp_path, config)
    full = V10BatchSource(
        corpus, config, chunk_frames=16, steps=3, batch_size=2, seed=2
    )
    resumed = V10BatchSource(
        corpus, config, chunk_frames=16, steps=3, batch_size=2, seed=2, start=1
    )
    assert len(resumed) == 2
    torch.testing.assert_close(resumed[0]["s"], full[1]["s"])
    torch.testing.assert_close(resumed[0]["m"], full[1]["m"])


def test_log_marginal_uses_logsumexp():
    log_weights = torch.tensor([[-1.0, 0.0], [2.0, 2.0]])
    expected = torch.tensor(
        [
            math.log(1.0 + math.exp(-1.0)) - math.log(2.0),
            2.0,
        ]
    )
    torch.testing.assert_close(log_marginal(log_weights), expected)
    mean_of_logs = log_weights.mean(dim=-1)
    assert not torch.allclose(log_marginal(log_weights), mean_of_logs)


def _toy_config(**changes) -> V10Step1Config:
    return _config(
        units=2,
        width=32,
        attention_heads=2,
        ffn_dim=64,
        attention_window=8,
        **changes,
    )


def test_marginal_estimator_matches_enumerated_paths():
    config = _toy_config()
    model = _model(config, seed=4)
    frames = 4
    s = torch.tensor([[3, 7, 11, 5]])
    lengths = torch.tensor([frames])
    paths = torch.tensor(
        list(itertools.product(range(config.m_vocab_size), repeat=frames))
    )
    joint = joint_log_pms(
        model, paths, s.expand(len(paths), -1), lengths.expand(len(paths))
    )
    exact_log_p = torch.logsumexp(joint, dim=0)
    exact_p = exact_log_p.exp().double()
    runs = 4000

    def estimate(k: int, resample: bool, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return marginal_log_ps(
            model,
            s.expand(runs, -1),
            lengths.expand(runs),
            k,
            generator,
            resample,
        )

    for k, resample, seed in ((1, False, 10), (8, False, 11), (8, True, 12)):
        values = estimate(k, resample, seed).exp().double()
        mean = values.mean()
        error = values.std() / math.sqrt(runs)
        assert abs(mean - exact_p) < 4.0 * error + 1e-15
        assert abs(mean / exact_p - 1.0) < 0.15


def test_resampling_one_item_leaves_the_others_alone():
    """Items of different content in one batch keep their own particles."""
    config = _toy_config()
    model = _model(config, seed=4)
    s = torch.tensor([[3, 7, 11, 5], [9, 2, 6, 14]])
    lengths = torch.tensor([4, 4])
    paths = torch.tensor(list(itertools.product(range(config.m_vocab_size), repeat=4)))
    runs = 3000
    together = marginal_log_ps(
        model,
        s.repeat_interleave(runs, dim=0),
        lengths.repeat_interleave(runs),
        8,
        torch.Generator().manual_seed(21),
        True,
    ).view(2, runs)
    for row in range(2):
        joint = joint_log_pms(
            model,
            paths,
            s[row].expand(len(paths), -1),
            lengths[row].expand(len(paths)),
        )
        exact = torch.logsumexp(joint, dim=0).exp().double()
        mean = together[row].exp().double().mean()
        assert abs(mean / exact - 1.0) < 0.15


def test_lead_frames_condition_the_estimate():
    """With a lead the estimate is ``log p(item | lead)``, not the prior path mean."""
    config = _toy_config()
    model = _model(config, seed=4)
    s = torch.tensor([[3, 7, 11, 5]])
    lengths = torch.tensor([4])
    paths = torch.tensor(list(itertools.product(range(config.m_vocab_size), repeat=4)))
    joint = joint_log_pms(
        model, paths, s.expand(len(paths), -1), lengths.expand(len(paths))
    )
    lead_paths = torch.tensor(
        list(itertools.product(range(config.m_vocab_size), repeat=2))
    )
    lead_joint = joint_log_pms(
        model,
        lead_paths,
        s[:, :2].expand(len(lead_paths), -1),
        torch.tensor([2]).expand(len(lead_paths)),
    )
    exact = torch.logsumexp(joint, 0) - torch.logsumexp(lead_joint, 0)
    estimate = marginal_log_ps(
        model, s, lengths, 200_000, torch.Generator().manual_seed(5), score_from=2
    )
    assert abs(float(estimate[0] - exact)) < 0.02


def test_decision_rule_needs_three_seeds_and_compares_the_right_way():
    path = Path(__file__).parents[1] / "scripts" / "v10_step1_eval.py"
    spec = importlib.util.spec_from_file_location("v10_step1_eval_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    points = {
        "A": [60.0, 60.2, 60.1],
        "B-sil": [60.0, 60.1, 60.2],
        "B": [62, 62.1, 62.2],
    }
    rule = script.decision_rule(points)
    assert rule["B_vs_B-sil"]["yes"] is True
    assert rule["B-sil_vs_A"]["yes"] is False
    assert abs(rule["B-sil_vs_A"]["delta"]) < 0.01
    worse = {
        "A": [62, 62.1, 62.2],
        "B-sil": [60.0, 60.1, 60.2],
        "B": [60.0, 60.1, 60.2],
    }
    assert script.decision_rule(worse)["B-sil_vs_A"]["delta"] < 0
    assert script.decision_rule({"B-sil": [60.0], "B": [62.0]}) == {}


def test_marginal_estimates_repeat_for_one_seed():
    config = _toy_config()
    model = _model(config, seed=6)
    s = torch.tensor([[3, 7, 11, 5], [3, 7, 11, 5]])
    lengths = torch.tensor([4, 4])

    def run() -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        return marginal_log_ps(model, s, lengths, 3, generator, resample=True)

    torch.testing.assert_close(run(), run())


def test_reindex_state_keeps_particles_consistent(model):
    config = model.config
    m, s = _tokens(config, 3, 5, seed=8)
    state = model.init_state(3)
    m_prev = torch.full((3,), config.m_bos_id, dtype=torch.long)
    s_prev = torch.full((3,), config.s_bos_id, dtype=torch.long)
    for frame in range(3):
        _, state = model.step(state, m_prev, s_prev)
        m_prev, s_prev = m[:, frame], s[:, frame]
    order = torch.tensor([2, 0, 2])
    moved = model.reindex_state(state, order)
    original = model.reindex_state(state, torch.arange(3))
    h_moved, _ = model.step(moved, m_prev[order], s_prev[order])
    h_original, _ = model.step(original, m_prev, s_prev)
    torch.testing.assert_close(h_moved, h_original[order], atol=1e-6, rtol=1e-6)


def test_common_random_numbers_repeat_pair_scores(model):
    config = model.config
    _, s = _tokens(config, 2, 6, seed=9)
    good, bad = s, (s + 1) % 2048
    lengths = torch.tensor([6, 5])
    first = pair_marginal_log_ps(model, good, bad, lengths, lengths, k=4, seed=21)
    second = pair_marginal_log_ps(model, good, bad, lengths, lengths, k=4, seed=21)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_joint_and_leak_diagnostic_match_forward(model):
    config = model.config
    m, s = _tokens(config, 2, 6, seed=12)
    lengths = torch.tensor([6, 3])
    output = model(m, s)
    log_m = torch.log_softmax(output.m_logits.float(), dim=-1)
    log_s = torch.log_softmax(output.s_logits.float(), dim=-1)
    picked_m = log_m.gather(-1, m[..., None])[..., 0]
    picked_s = log_s.gather(-1, s[..., None])[..., 0]
    valid = torch.arange(6)[None] < lengths[:, None]
    torch.testing.assert_close(
        joint_log_pms(model, m, s, lengths), ((picked_m + picked_s) * valid).sum(-1)
    )
    mean_nll = -(picked_s * valid).sum(-1) / valid.sum(-1)
    torch.testing.assert_close(nll_s_given_true_m(model, m, s, lengths), mean_nll)


def test_arm_a_exact_matches_pad_forward(model):
    config = model.config
    _, s = _tokens(config, 2, 7, seed=10)
    lengths = torch.tensor([7, 4])
    pad = torch.full_like(s, config.pad_id)
    output = model(pad, s)
    values = torch.log_softmax(output.s_logits.float(), dim=-1)
    picked = values.gather(-1, s[..., None])[..., 0]
    valid = torch.arange(7)[None] < lengths[:, None]
    expected = (picked * valid).sum(dim=-1)
    torch.testing.assert_close(
        exact_log_ps_pad(model, s, lengths), expected, atol=1e-5, rtol=1e-5
    )


def test_pair_accuracy_and_stability_arithmetic():
    outcomes = pair_outcomes([3.0, 1.0, 2.0, 2.0], [1.0, 3.0, 2.0, 1.0])
    assert outcomes == (1.0, 0.0, 0.5, 1.0)
    assert sum(outcomes) / len(outcomes) == pytest.approx(0.625)
    assert stability(outcomes, outcomes) == 0.0
    assert stability((1.0, 0.0), (0.0, 1.0)) == 1.0
    assert normalized_scores([10.0, 5.0], [5, 10]) == [2.0, 0.5]
    assert LEAD_FRAMES == 6
