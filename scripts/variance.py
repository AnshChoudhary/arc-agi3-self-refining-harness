"""Per-game score variance across repeated runs of one harness.

    python scripts/variance.py --harness bc653ad4e040 --cap 60

Reads trajectories only; comparisons are only meaningful within one harness,
cap and effort, so all three are filters rather than groupings.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics

from arc_harness.trajectory import TRAJECTORIES_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="fingerprint suffix, e.g. bc653ad4e040")
    ap.add_argument("--cap", type=int, default=None, help="max_actions_per_level the run used")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--set", choices=["refine", "heldout"], default="refine")
    args = ap.parse_args()

    samples: dict[str, list[tuple[int, bool, float]]] = collections.defaultdict(list)
    for path in sorted((TRAJECTORIES_DIR / args.set).glob("*_student_*.json")):
        t = json.loads(path.read_text())
        if not str(t.get("harness_snapshot", "")).endswith(args.harness):
            continue
        if args.cap is not None and t.get("max_actions_per_level") != args.cap:
            continue
        if args.effort is not None and t.get("effort") not in (None, args.effort):
            continue
        if t["outcome"] and str(t["outcome"]).startswith("aborted"):
            continue  # a run cut short by the provider is not a sample
        samples[t["game_id"][:4]].append(
            (t["level_actions"][0] if t["level_actions"] else 0, t["levels_completed"] > 0, t["score"]))

    if not samples:
        raise SystemExit(f"no trajectories match harness {args.harness}")
    print(f"{'game':<7}{'n':<4}{'solved':<9}{'runs':<34}{'mean':<10}{'sd':<10}{'sem'}")
    means = []
    for g in sorted(samples):
        sc = [x[2] for x in samples[g]]
        sd = statistics.stdev(sc) if len(sc) > 1 else 0.0
        sem = sd / len(sc) ** 0.5 if sc else 0.0
        means.append(statistics.mean(sc))
        runs = " ".join(f"{a}{'' if ok else 'F'}" for a, ok, _ in samples[g])
        print(f"{g:<7}{len(sc):<4}{sum(ok for _, ok, _ in samples[g])}/{len(sc):<7}{runs:<34}"
              f"{statistics.mean(sc):<10.4f}{sd:<10.4f}{sem:.4f}")
    n = len(means)
    sems = []
    for v in samples.values():
        scores = [s[2] for s in v]
        if len(scores) > 1:
            sems.append(statistics.stdev(scores) / len(scores) ** 0.5)
    combined = (sum(e ** 2 for e in sems) ** 0.5) / n if sems and n else 0.0
    print(f"\nset mean {statistics.mean(means):.4f}; standard error of that mean ~{combined:.4f}")
    print(f"a change smaller than about {2 * combined:.4f} is not distinguishable from run-to-run noise here")


if __name__ == "__main__":
    main()
