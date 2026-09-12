"""Agent protocol shared by scripts/eval.py and the sanity baselines.

student.py (M1) implements the same interface, so the referee never needs to
know which agent it is running.
"""

from __future__ import annotations

import random
from typing import Protocol

from arc_harness.env import ArcEnv, InvalidAction
from arc_harness.llm import Usage
from arc_harness.trajectory import AgentStep


class Agent(Protocol):
    name: str

    def play(self, env: ArcEnv, max_levels: int | None) -> tuple[list[AgentStep], Usage]:
        """Act until env.done or max_levels completed. Must not touch the harness."""

    def estimate_usage(self, total_action_budget: int) -> Usage:
        """Upper-bound token usage for a run with this many budgeted actions (cost projection)."""


class RandomAgent:
    """Uniform random over available actions. Expected RHAE ~0; exercises the plumbing for free."""

    name = "random"

    def __init__(self, agent_seed: int = 0) -> None:
        self.agent_seed = agent_seed

    def play(self, env: ArcEnv, max_levels: int | None) -> tuple[list[AgentStep], Usage]:
        # Seed per game so a game's result does not depend on its position in the eval set.
        self.rng = random.Random(f"{self.agent_seed}:{env.game_id}")
        steps: list[AgentStep] = []
        while not env.done and (max_levels is None or env.levels_completed < max_levels):
            obs = env.observe()
            if obs.state == "GAME_OVER":
                env.reset()
                steps.append(AgentStep(hypothesis="game over -> reset"))
                continue
            action = self.rng.choice(obs.available_actions)
            try:
                if action == "ACTION6":
                    env.step(action, x=self.rng.randrange(64), y=self.rng.randrange(64))
                else:
                    env.step(action)
            except InvalidAction as e:  # cannot happen when choosing from available_actions
                steps.append(AgentStep(invalid_attempts=[str(e)]))
                continue
            steps.append(AgentStep())
        return steps, Usage()

    def estimate_usage(self, total_action_budget: int) -> Usage:
        return Usage()
