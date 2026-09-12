"""M0 sanity baseline: uniform random actions until the budget kills the run.

Expected RHAE ~0. Exercises env.py budget enforcement, trajectory logging, and
scoring end to end without spending a single LLM token.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from arc_harness.env import ArcEnv, InvalidAction
from arc_harness.scoring import game_score
from arc_harness.trajectory import AgentStep, build, save


def play(env: ArcEnv, rng: random.Random, max_levels: int | None) -> list[AgentStep]:
    agent_steps: list[AgentStep] = []
    while not env.done:
        if max_levels is not None and env.levels_completed >= max_levels:
            break
        obs = env.observe()
        if obs.state == "GAME_OVER":
            env.reset()
            agent_steps.append(AgentStep(hypothesis="game over -> reset"))
            continue
        action = rng.choice(obs.available_actions)
        try:
            if action == "ACTION6":
                env.step(action, x=rng.randrange(64), y=rng.randrange(64))
            else:
                env.step(action)
        except InvalidAction as e:  # should not happen for random-from-available
            agent_steps.append(AgentStep(invalid_attempts=[str(e)]))
            continue
        agent_steps.append(AgentStep())
    return agent_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="e.g. ls20 or ls20-9607627b")
    ap.add_argument("--seed", type=int, default=0, help="environment seed")
    ap.add_argument("--agent-seed", type=int, default=0)
    ap.add_argument("--max-levels", type=int, default=1, help="stop after this many levels (sanity runs)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    env = ArcEnv(args.game, seed=args.seed)
    print(f"{env.game_id}: {len(env.baselines)} levels, baselines={env.baselines}, "
          f"actions={env.available_actions}, level-0 budget={env.budget_this_level}")

    agent_steps = play(env, random.Random(args.agent_seed), args.max_levels)
    res = env.result()
    gs = game_score(res.game_id, res.baselines, res.level_actions, res.levels_completed)

    print(f"outcome={res.outcome} levels_completed={res.levels_completed} total_actions={res.total_actions}")
    print("level  human  ai   score")
    for i, (h, a, s) in enumerate(zip(gs.baselines, gs.level_actions, gs.level_scores)):
        print(f"{i + 1:>5}  {h:>5}  {a:>3}  {s:.3f}")
    print(f"game RHAE = {gs.score:.4f}")

    if not args.no_save:
        path = save(build(env, agent="random", agent_steps=agent_steps, score=gs.score, started_at=started))
        print(f"trajectory -> {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
