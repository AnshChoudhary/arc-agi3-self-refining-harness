"""Trajectory logging schema and split-aware writer.

Trajectories are the coach's only input, so we log generously. Files land in
trajectories/refine/ or trajectories/heldout/ according to config/split.json;
the coach may only ever open the former.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from arc_harness.env import PROJECT_ROOT, ArcEnv, Step

SPLIT_FILE = PROJECT_ROOT / "config" / "split.json"
TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"


@dataclass
class AgentStep:
    """What the agent thought and did around one env action (all optional for scripted agents)."""

    hypothesis: str | None = None
    analysis_code: list[str] = field(default_factory=list)
    repl_outputs: list[str] = field(default_factory=list)
    invalid_attempts: list[str] = field(default_factory=list)
    notes: str | None = None  # the student's durable scratchpad after this step
    reply: str | None = None  # final raw LLM reply that produced the action
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trajectory:
    trajectory_id: str
    game_id: str
    split: str
    agent: str
    seed: int
    model: str | None
    harness_snapshot: str | None
    budget_multiplier: int
    toolkit: dict[str, str]
    started_at: str
    baselines: list[int]
    env_steps: list[Step]
    agent_steps: list[AgentStep]
    outcome: str | None
    levels_completed: int
    level_actions: list[int]
    score: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


def load_split() -> dict[str, list[str]]:
    return json.loads(SPLIT_FILE.read_text())


def split_of(game_id: str) -> str:
    """'refine' | 'heldout'. Accepts base ids ('ls20') or versioned ids."""
    base = game_id.split("-", 1)[0]
    split = load_split()
    for name in ("refine", "heldout"):
        if any(g.split("-", 1)[0] == base for g in split[name]):
            return name
    raise KeyError(f"{game_id!r} is not in config/split.json; re-run scripts/make_split.py")


def build(
    env: ArcEnv,
    agent: str,
    agent_steps: list[AgentStep],
    score: float,
    model: str | None = None,
    harness_snapshot: str | None = None,
    started_at: datetime | None = None,
) -> Trajectory:
    started = started_at or datetime.now(timezone.utc)
    res = env.result()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    return Trajectory(
        trajectory_id=f"{env.game_id}_{agent}_{stamp}",
        game_id=env.game_id,
        split=split_of(env.game_id),
        agent=agent,
        seed=env.seed,
        model=model,
        harness_snapshot=harness_snapshot,
        budget_multiplier=env.budget_multiplier,
        toolkit={"arc-agi": version("arc-agi"), "arcengine": version("arcengine")},
        started_at=started.isoformat(),
        baselines=env.baselines,
        env_steps=env.steps,
        agent_steps=agent_steps,
        outcome=res.outcome,
        levels_completed=res.levels_completed,
        level_actions=res.level_actions,
        score=score,
    )


def save(traj: Trajectory) -> Path:
    out_dir = TRAJECTORIES_DIR / traj.split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{traj.trajectory_id}.json"
    path.write_text(traj.to_json())
    return path
