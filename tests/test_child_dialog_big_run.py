"""Environment/default checks for the big-pool shell driver."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "child_dialog_big_run.sh"


def _executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def _fake_commands(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kpy_impl = tmp_path / "fake_kpy.py"
    kpy_impl.write_text(
        """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"cmd": "synth", "args": args}) + "\\n")
out = Path(args[args.index("--output") + 1])
(out / "wavs").mkdir(parents=True, exist_ok=True)
(out / "plan.jsonl").write_text("")
"""
    )
    uv_impl = tmp_path / "fake_uv.py"
    uv_impl.write_text(
        """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def log(**row):
    with open(os.environ["CALL_LOG"], "a") as handle:
        handle.write(json.dumps(row) + "\\n")
if "scripts/child_dialog_encode.py" in args:
    out = Path(args[args.index("--output") + 1])
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "train" / "manifest.jsonl").write_text('{"frames": 45000}\\n')
    log(cmd="encode")
elif any("upload_file" in arg for arg in args):
    log(cmd="upload", dest=args[-1])
else:
    print("0.0001")
"""
    )
    kpy = _executable(
        bin_dir / "kpy",
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} "
        f'{shlex.quote(str(kpy_impl))} "$@"\n',
    )
    _executable(
        bin_dir / "uv",
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} "
        f'{shlex.quote(str(uv_impl))} "$@"\n',
    )
    return bin_dir, kpy


def _run(tmp_path: Path, bin_dir: Path, kpy: Path, extra: dict, log: Path) -> str:
    work = tmp_path / f"work-{log.stem}"
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{}\n")
    environment = {
        **os.environ,
        "PAIRS": str(pairs),
        "RENDITIONS": "1",
        "SHARDS": "1",
        "FIRST": "0",
        "LAST": "0",
        "WORK": str(work),
        "KPY": str(kpy),
        "WORKERS": "1",
        "MIMI": str(tmp_path / "mimi.pt"),
        "SILENCE": str(tmp_path / "silence.json"),
        "CALL_LOG": str(log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        **extra,
    }
    for key in ("TAG", "SEED", "DEST_PREFIX"):
        if key not in extra:
            environment.pop(key, None)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "BIG SHARD 0 OK" in result.stdout
    return result.stdout


def _records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def _argument(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_big_run_defaults_match_existing_pool(tmp_path):
    bin_dir, kpy = _fake_commands(tmp_path)
    log = tmp_path / "defaults.log"
    _run(tmp_path, bin_dir, kpy, {}, log)
    records = _records(log)
    synth = next(row for row in records if row["cmd"] == "synth")
    assert _argument(synth["args"], "--id-tag") == ""
    assert _argument(synth["args"], "--seed") == "0"
    upload = next(row for row in records if row["cmd"] == "upload")
    assert upload["dest"] == "big/shard-0000.tar"


def test_big_run_env_overrides_tag_seed_and_destination(tmp_path):
    bin_dir, kpy = _fake_commands(tmp_path)
    log = tmp_path / "tagged.log"
    _run(
        tmp_path,
        bin_dir,
        kpy,
        {"TAG": "pool2", "SEED": "7", "DEST_PREFIX": "big2"},
        log,
    )
    records = _records(log)
    synth = next(row for row in records if row["cmd"] == "synth")
    assert _argument(synth["args"], "--id-tag") == "pool2"
    assert _argument(synth["args"], "--seed") == "7"
    upload = next(row for row in records if row["cmd"] == "upload")
    assert upload["dest"] == "big2/shard-0000.tar"
