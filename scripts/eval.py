"""Referee: run an agent on a game set, report RHAE and cost, append a results row.

    python scripts/eval.py --set refine --agent random --max-levels 1
    python scripts/eval.py --set heldout --agent student --model haiku --harness current

The referee is fixed: same budget, same scoring, same base prompt for every
run. It records everything needed to reproduce the row (CLAUDE.md "How to work").
Cost is projected before anything runs and the run aborts above COST_CAP_USD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version

from arc_harness.agents import Agent, RandomAgent, ScriptedHandleAgent
from arc_harness.env import PROJECT_ROOT, ArcEnv, baselines_for
from arc_harness.llm import CostCapExceeded, LLMClient, Usage
from arc_harness.repl import harness_fingerprint
from arc_harness.scoring import GameScore, game_score, rhae
from arc_harness.student import StudentAgent
from arc_harness.trajectory import SPLIT_FILE, build, load_split, save
from config.budget import BUDGET_MULTIPLIER, level_budget
from config.models import COST_CAP_USD, DEFAULT_EFFORT, DEFAULT_MODEL, EFFORTS, MODELS, ModelSpec, get_model

RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class GameRow:
    game_id: str
    score: float
    levels_completed: int
    outcome: str | None
    level_actions: list[int]
    baselines: list[int]
    trajectory: str
    input_tokens: int
    output_tokens: int


def make_agent(name: str, model: ModelSpec | None, effort: str, agent_seed: int, use_cache: bool) -> Agent:
    if name == "random":
        return RandomAgent(agent_seed)
    if name == "scripted":
        return ScriptedHandleAgent()
    if name == "student":
        assert model is not None
        return StudentAgent(LLMClient(model, effort=effort, use_cache=use_cache, budget_usd=COST_CAP_USD))
    raise SystemExit(f"unknown agent {name!r}")


@dataclass
class GameSpec:
    """Everything a worker process needs to play one game; picklable."""

    game_id: str
    agent: str
    model_key: str | None
    effort: str
    agent_seed: int
    use_cache: bool
    env_seed: int
    offline: bool | None
    max_levels: int | None
    harness: str
    started_iso: str


def run_game(spec: GameSpec) -> dict:
    """Play one game to completion (or abort) and save its trajectory. Runs in a worker process."""
    model = get_model(spec.model_key) if spec.model_key else None
    agent = make_agent(spec.agent, model, spec.effort, spec.agent_seed, spec.use_cache)
    env = ArcEnv(spec.game_id, seed=spec.env_seed, offline=spec.offline)
    tag = env.game_id.split("-")[0]
    log = lambda msg: print(f"[{tag}]{msg}", flush=True)  # noqa: E731
    if isinstance(agent, StudentAgent):
        agent.log = log
    log(f" baselines={env.baselines} actions={env.available_actions}")

    aborted = None
    try:
        agent_steps, used = agent.play(env, spec.max_levels)
    except (CostCapExceeded, KeyboardInterrupt, Exception) as e:  # keep what we have; the row says why
        aborted = type(e).__name__
        agent_steps, used = [], Usage()
        log(f" aborted: {aborted}: {e}")
        if not isinstance(e, (CostCapExceeded, KeyboardInterrupt)):
            import traceback
            traceback.print_exc()
    if isinstance(agent, StudentAgent):
        used = agent.llm.usage  # authoritative, includes calls made before an abort

    res = env.result()
    gs = game_score(res.game_id, res.baselines, res.level_actions, res.levels_completed)
    started = datetime.fromisoformat(spec.started_iso)
    path = save(build(env, agent.name, agent_steps, gs.score, model.name if model else None,
                      f"{spec.harness}@{harness_fingerprint()}", started))
    if aborted:
        outcome = f"aborted_{aborted}"
    elif res.outcome is None:  # env.done is False: either we stopped it or the agent gave up
        hit_cap = spec.max_levels is not None and res.levels_completed >= spec.max_levels
        outcome = "stopped_max_levels" if hit_cap else "agent_gave_up"
    else:
        outcome = res.outcome
    row = GameRow(res.game_id, gs.score, res.levels_completed, outcome, res.level_actions,
                  res.baselines, str(path.relative_to(PROJECT_ROOT)), used.input_tokens, used.output_tokens)
    log(f" score={gs.score:.3f} levels={res.levels_completed}/{len(res.baselines)} "
        f"actions={res.level_actions} outcome={outcome} cost=${used.cost_usd(model):.4f}")
    return {"row": row, "score": gs, "usage": used, "aborted": aborted}


def select_games(which: str, only: list[str] | None) -> list[str]:
    split = load_split()
    pool = split["refine"] + split["heldout"] if which == "all" else split[which]
    if not only:
        return pool
    chosen = []
    for g in only:
        base = g.split("-", 1)[0]
        match = [p for p in pool if p.split("-", 1)[0] == base]
        if not match:
            raise SystemExit(f"{g!r} is not in the {which} set; refusing to cross the split")
        chosen.append(match[0])
    return chosen


def project_cost(agent: Agent, games: list[str], model: ModelSpec | None, max_levels: int | None,
                 multiplier: int, offline: bool | None) -> tuple[int, Usage, float]:
    total_budget = 0
    for g in games:
        bl = baselines_for(g, offline)
        levels = bl if max_levels is None else bl[:max_levels]
        total_budget += sum(level_budget(b, multiplier) for b in levels)
    usage = agent.estimate_usage(total_budget)
    return total_budget, usage, usage.cost_usd(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["refine", "heldout", "all"], required=True)
    ap.add_argument("--agent", choices=["random", "scripted", "student"], default="random")
    ap.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                    help=f"LLM alias; default is the cheapest ({DEFAULT_MODEL})")
    ap.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT,
                    help=f"reasoning effort for the student's model; default is the cheapest ({DEFAULT_EFFORT})")
    ap.add_argument("--no-llm-cache", action="store_true", help="always call the API (default: reuse cached replies)")
    ap.add_argument("--harness", default="current", help="harness snapshot id (M2); 'current' = harness/ as is")
    ap.add_argument("--games", help="comma-separated subset (must belong to --set)")
    ap.add_argument("--seed", type=int, default=0, help="environment seed")
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--max-levels", type=int, default=None, help="stop each game after N levels")
    ap.add_argument("--offline", action="store_true", help="never contact the ARC API")
    ap.add_argument("--jobs", type=int, default=1, help="games played concurrently (separate processes)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost projection and exit")
    args = ap.parse_args()

    offline = True if args.offline else None
    model = get_model(args.model) if args.agent == "student" else None
    agent = make_agent(args.agent, model, args.effort, args.agent_seed, not args.no_llm_cache)
    games = select_games(args.set, args.games.split(",") if args.games else None)

    total_budget, est, est_usd = project_cost(agent, games, model, args.max_levels, BUDGET_MULTIPLIER, offline)
    print(f"{len(games)} games ({args.set}), agent={agent.name}, model={model.name if model else '-'}, "
          f"effort={args.effort if model else '-'}, harness={args.harness}@{harness_fingerprint()}")
    print(f"projected: <= {total_budget} env actions, ~{est.input_tokens + est.output_tokens} tokens, "
          f"${est_usd:.2f} (cap ${COST_CAP_USD:.2f})")
    if est_usd > COST_CAP_USD:
        raise SystemExit(f"projected cost ${est_usd:.2f} exceeds COST_CAP_USD; raise it in config/models.py")
    if args.dry_run:
        return

    # A plain `kill` (SIGTERM) should still save what we have, like Ctrl-C does.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    specs = [GameSpec(g, args.agent, args.model if model else None, args.effort, args.agent_seed,
                      not args.no_llm_cache, args.seed, offline, args.max_levels, args.harness, started.isoformat())
             for g in games]
    rows: list[GameRow] = []
    scores: list[GameScore] = []
    usage = Usage()

    def collect(result: dict) -> None:
        rows.append(result["row"])
        scores.append(result["score"])
        usage.add(result["usage"])

    if args.jobs <= 1:
        for spec in specs:
            collect(run_game(spec))
            if rows[-1].outcome.startswith("aborted_"):
                break
    else:
        # One process per game; each builds its own agent and LLM client. Ctrl-C loses in-flight games.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_game, spec) for spec in specs]
            try:
                for fut in as_completed(futures):
                    collect(fut.result())
            except KeyboardInterrupt:
                pool.shutdown(cancel_futures=True)
                print(f"interrupted: {len(rows)}/{len(specs)} games finished; in-flight games were lost", flush=True)
        rows.sort(key=lambda r: r.game_id)
        scores.sort(key=lambda g: g.game_id)

    total = rhae(scores)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{args.set}_{agent.name}"
    record = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "set": args.set,
        "agent": agent.name,
        "model": model.name if model else None,
        "effort": args.effort if model else None,
        "harness_snapshot": f"{args.harness}@{harness_fingerprint()}",
        "split_file_sha1": hashlib.sha1(SPLIT_FILE.read_bytes()).hexdigest()[:12],
        "split_seed": load_split()["seed"],
        "env_seed": args.seed,
        "agent_seed": args.agent_seed,
        "budget_multiplier": BUDGET_MULTIPLIER,
        "max_levels": args.max_levels,
        "jobs": args.jobs,
        "toolkit": {"arc-agi": version("arc-agi"), "arcengine": version("arcengine")},
        "argv": sys.argv[1:],
        "rhae": total,
        "cost": {**usage.__dict__, "usd": round(usage.cost_usd(model), 4)},
        "games": [asdict(r) for r in rows],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"RHAE({args.set}) = {total:.4f}  cost=${record['cost']['usd']:.4f}  -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
