"""Budget enforcement and guard rails in env.py, on one real game."""

import pytest

from arc_harness.env import OUTCOME_BUDGET, ArcEnv, InvalidAction

GAME = "ls20"  # used only for plumbing tests; never referenced by the harness


def test_rejects_unavailable_and_malformed_actions():
    env = ArcEnv(GAME, seed=0)
    assert env.frame.shape == (64, 64)
    with pytest.raises(InvalidAction):
        env.step("ACTION6", x=1, y=1)  # not in this game's action space
    with pytest.raises(InvalidAction):
        env.step("ACTION1", x=1)
    with pytest.raises(InvalidAction):
        env.reset()  # zero actions on level -> would be a full reset
    assert env.steps == [] and env.actions_this_level == 0


def test_reset_costs_one_action_and_keeps_level():
    env = ArcEnv(GAME, seed=0)
    env.step("ACTION1")
    obs = env.reset()
    assert obs.actions_this_level == 2
    assert obs.levels_completed == 0
    assert env.steps[-1].action == "RESET" and not env.steps[-1].full_reset


def test_budget_ends_episode():
    env = ArcEnv(GAME, seed=0, budget_multiplier=1)
    budget = env.budget_this_level
    for _ in range(budget):
        obs = env.step("ACTION1")
    assert obs.done and obs.outcome == OUTCOME_BUDGET
    with pytest.raises(InvalidAction):
        env.step("ACTION1")
    res = env.result()
    assert res.level_actions == [budget] and res.levels_completed == 0
