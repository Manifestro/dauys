#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${PAIRS:?PAIRS must point to pairs.jsonl}"
: "${RENDITIONS:?RENDITIONS must be set}"
: "${SHARDS:?SHARDS must be set}"
: "${FIRST:?FIRST must be set}"
: "${LAST:?LAST must be set}"
: "${WORK:?WORK must be set}"
: "${KPY:?KPY must point to the Kokoro venv python}"
: "${WORKERS:?WORKERS must be set}"
: "${MIMI:?MIMI must point to the Mimi checkpoint}"
: "${SILENCE:?SILENCE must point to silence-codes.json}"
TAG="${TAG:-}"
SEED="${SEED:-0}"
DEST_PREFIX="${DEST_PREFIX:-big}"

mkdir -p "$WORK"
for ((K = FIRST; K <= LAST; K++)); do
    START=$(date +%s)
    SHARD="$WORK/shard-$K"
    TAR="$WORK/shard-$K.tar"
    rm -rf "$SHARD" "$TAR"
    mkdir -p "$SHARD"
    if "$KPY" scripts/child_dialog_synth.py \
        --pairs "$PAIRS" \
        --output "$SHARD/synth" \
        --big-renditions "$RENDITIONS" \
        --shard "$K" \
        --shards "$SHARDS" \
        --seed "$SEED" \
        --id-tag "$TAG" \
        --workers "$WORKERS" \
        --pcm16; then
        if uv run --no-sync python scripts/child_dialog_encode.py \
            --synth "$SHARD/synth" \
            --mimi-checkpoint "$MIMI" \
            --silence-codes "$SILENCE" \
            --output "$SHARD/enc"; then
            if ! HOURS=$(uv run --no-sync python -c \
                'import json, sys; print(round(sum(json.loads(line)["frames"] for line in open(sys.argv[1])) / 12.5 / 3600, 4))' \
                "$SHARD/enc/train/manifest.jsonl"); then
                echo "BIG SHARD $K FAILED"
                continue
            fi
            if tar -cf "$TAR" -C "$SHARD" enc synth/plan.jsonl; then
                DEST=$(printf "%s/shard-%04d.tar" "$DEST_PREFIX" "$K")
                if uv run --no-sync python -c \
                    'import sys; from huggingface_hub import upload_file; upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2], repo_id="karl4th/dauys-child-dialog-text", repo_type="dataset")' \
                    "$TAR" "$DEST"; then
                    rm -rf "$SHARD/synth/wavs" "$SHARD/enc"
                    ELAPSED=$(($(date +%s) - START))
                    echo "BIG SHARD $K OK $HOURS $ELAPSED"
                    continue
                fi
            fi
        fi
    fi
    echo "BIG SHARD $K FAILED"
done
