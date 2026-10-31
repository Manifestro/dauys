"""Pack one ready tier of the hohere shards for V10 step 1.

    python scripts/v10_step1_pack.py --hf-repo manifestro/hohere-v2-librilight \
        --root SHARDS --tier 500 --units 8192 --output PACKED

Without ``--hf-repo`` the shards must already be under ``--root``. The output is
reusable across arms and seeds, so upload it to ``manifestro/dauys-runs`` the
first time.
"""

import argparse
import json
from pathlib import Path

from dauys.v10_step1_data import fetch_tier, pack_tier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--units", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.hf_repo:
        fetch_tier(args.hf_repo, args.tier, args.root, units=args.units)
    meta = pack_tier(
        args.root,
        args.tier,
        args.output,
        units=args.units,
        workers=args.workers,
    )
    hours = sum(meta["frames"]) / 12.5 / 3600
    print(
        json.dumps(
            {
                "records": len(meta["ids"]),
                "hours": round(hours, 1),
                "centroids_sha256": meta["centroids_sha256"],
                "sil_b_sil": meta["sil_b_sil"],
                "sil_same_frame_in_b": meta["sil_same_frame_in_b"],
            }
        )
    )


if __name__ == "__main__":
    main()
