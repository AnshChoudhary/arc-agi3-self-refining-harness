"""One coach round: snapshot -> read refine trajectories -> propose -> validate -> apply -> (eval, rollback).

    python scripts/refine.py --effort high --dry-run          # propose only, print
    python scripts/refine.py --effort high                    # apply and log
    python scripts/refine.py --effort high --eval --jobs 5 --max-actions-per-level 60

Isolation: this script and coach.py never read trajectories/heldout/ or the
held-out half of the split. The evaluation step is the referee (eval.py) run
as a subprocess; only its printed RHAE for the *refine* set feeds the rollback rule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from arc_harness import harness_state as hs
from arc_harness.coach import digest, load_refine_trajectories, propose
from arc_harness.env import PROJECT_ROOT
from arc_harness.llm import LLMClient
from arc_harness.repl import harness_fingerprint
from config.models import COACH_MAX_TOKENS, DEFAULT_EFFORT, DEFAULT_MODEL, EFFORTS, MODELS, get_model

RESULTS_DIR = PROJECT_ROOT / "results"


def latest_refine_rhae(fingerprint: str) -> float | None:
    """Refine-set RHAE of the most recent student eval run on a given harness fingerprint."""
    rows = []
    for path in RESULTS_DIR.glob("*_refine_student.json"):
        r = json.loads(path.read_text())
        if str(r.get("harness_snapshot", "")).endswith(fingerprint):
            rows.append(r)
    if not rows:
        return None
    return max(rows, key=lambda r: r["started_at"])["rhae"]


def run_eval(which: str, args: argparse.Namespace) -> float:
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "eval.py"), "--set", which, "--agent", "student",
           "--model", args.model, "--effort", args.effort, "--jobs", str(args.jobs), "--offline"]
    if args.max_levels is not None:
        cmd += ["--max-levels", str(args.max_levels)]
    if args.max_actions_per_level is not None:
        cmd += ["--max-actions-per-level", str(args.max_actions_per_level)]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"eval on {which} failed")
    for line in proc.stdout.splitlines():
        if line.startswith(f"RHAE({which})"):
            return float(line.split("=")[1].split()[0])
    raise SystemExit(f"could not read RHAE({which}) from eval output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    ap.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT)
    ap.add_argument("--max-edits", type=int, default=3)
    ap.add_argument("--retries", type=int, default=1, help="repair attempts after a validator rejection")
    ap.add_argument("--any-harness", action="store_true",
                    help="use the newest trajectory per game regardless of which harness produced it")
    ap.add_argument("--dry-run", action="store_true", help="propose and validate only; touch nothing")
    ap.add_argument("--eval", action="store_true", help="after applying, run eval.py on refine then heldout; roll back on refine regression")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--max-levels", type=int, default=1)
    ap.add_argument("--max-actions-per-level", type=int, default=60)
    ap.add_argument("--no-llm-cache", action="store_true")
    args = ap.parse_args()

    round_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    before_fp = harness_fingerprint()
    # Reason about the harness we are about to edit: trajectories produced by a since-reverted
    # harness would otherwise be read as evidence about the current one.
    trajectories = load_refine_trajectories(agent="student", harness_fingerprint=None if args.any_harness else before_fp)
    if not trajectories:
        raise SystemExit(f"no student trajectories under trajectories/refine/ for harness {before_fp}; "
                         "run eval.py --set refine first, or pass --any-harness")
    digests = [digest(t) for t in trajectories]
    known_ids = {t["trajectory_id"] for t in trajectories}
    print(f"round {round_id}: harness {before_fp}, {len(digests)} refine trajectories "
          f"({sum(d.solved for d in digests)} solved)")

    model = get_model(args.model)
    llm = LLMClient(model, effort=args.effort, use_cache=not args.no_llm_cache, max_tokens=COACH_MAX_TOKENS)
    check = lambda proposed: hs.validate(proposed, known_ids, args.max_edits)  # noqa: E731
    analysis, edits, problems = propose(llm, digests, hs.read_files(), args.max_edits, validate=check,
                                        retries=args.retries)
    print(f"coach ({model.name}, effort {args.effort}, {llm.usage.calls} call, ${llm.usage.cost_usd(model):.4f}):")
    print("  analysis:", analysis)
    for i, e in enumerate(edits, 1):
        target = e.name if e.op == "write_tool" else (f"rule {e.rule_number}" if e.rule_number else "")
        print(f"  [{i}] {e.tag:<10} {e.op:<15} {target:<10} evidence={e.evidence}")
        print(f"       {e.rationale}")
        if e.op in ("add_rule", "replace_rule"):
            print(f"       -> {e.text}")
        else:  # a dry run must show the code that would be installed
            for line in e.text.rstrip().splitlines():
                print(f"       | {line}")

    if problems:
        print("REJECTED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    if args.dry_run:
        print("dry run: nothing written")
        return

    snap = hs.snapshot(round_id)
    entries = hs.apply(edits, round_id, model.name, snap)
    after_fp = harness_fingerprint()
    print(f"applied {len(entries)} edits; harness {before_fp} -> {after_fp}; snapshot {snap.relative_to(PROJECT_ROOT)}")
    if not args.eval:
        return

    baseline = latest_refine_rhae(before_fp)
    new_refine = run_eval("refine", args)
    print(f"refine RHAE: {baseline} -> {new_refine}")
    if baseline is not None and new_refine < baseline:
        hs.restore(snap)
        hs.log_rollback(round_id, snap, f"refine RHAE regressed {baseline:.4f} -> {new_refine:.4f}")
        print(f"ROLLED BACK to {before_fp}; harness now {harness_fingerprint()}")
        return
    new_heldout = run_eval("heldout", args)  # referee only; the coach never sees these trajectories
    print(f"held-out RHAE (referee): {new_heldout}")


if __name__ == "__main__":
    main()
