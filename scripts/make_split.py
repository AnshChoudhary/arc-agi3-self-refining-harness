"""Deterministically split all public games into refine (~60%) / held-out (~40%).

Writes config/split.json with versioned game ids so runs are reproducible even
if the API later publishes a new version of a game.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone

from arc_harness.env import list_game_ids
from arc_harness.trajectory import SPLIT_FILE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine-frac", type=float, default=0.6)
    args = ap.parse_args()

    ids = list_game_ids()  # sorted, so the shuffle depends only on the seed
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_refine = round(len(ids) * args.refine_frac)
    split = {
        "seed": args.seed,
        "refine_frac": args.refine_frac,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "refine": sorted(ids[:n_refine]),
        "heldout": sorted(ids[n_refine:]),
    }
    SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n")
    print(f"{len(ids)} games -> {n_refine} refine / {len(ids) - n_refine} held-out, seed={args.seed}")
    print(f"wrote {SPLIT_FILE}")


if __name__ == "__main__":
    main()
