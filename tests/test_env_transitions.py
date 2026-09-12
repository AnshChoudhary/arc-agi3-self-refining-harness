"""Level-transition and win billing in env.py, driven by synthetic frames.

No public game has a known scripted solution yet, so we feed FrameDataRaw
objects straight into ArcEnv._record to check the accounting the scorecard
relies on: the completing action is billed to the level it completed, the
counter restarts on the new level, and WIN / budget end the episode.
"""

import numpy as np
from arcengine import FrameDataRaw, GameState

from arc_harness.env import OUTCOME_BUDGET, OUTCOME_WIN, ArcEnv
from arc_harness.scoring import game_score


def make_env() -> ArcEnv:
    env = ArcEnv("ls20", seed=0, budget_multiplier=1)
    env.baselines = [3, 2, 4]  # synthetic baselines; budgets 3, 2, 4 at multiplier 1
    return env


def raw(levels_completed: int, state: GameState = GameState.NOT_FINISHED) -> FrameDataRaw:
    f = FrameDataRaw(state=state, levels_completed=levels_completed, win_levels=3, available_actions=[1, 2, 3, 4])
    f.frame = [np.full((64, 64), levels_completed, dtype=np.int8)]
    return f


def test_completing_action_is_billed_to_finished_level():
    env = make_env()
    env._record(raw(0), "ACTION1", {})
    obs = env._record(raw(1), "ACTION2", {})  # second action completes level 1
    assert env.level_actions == [2]
    assert obs.actions_this_level == 0 and obs.level == 1 and obs.budget_this_level == 2
    assert env.steps[-1].level == 0 and env.steps[-1].level_changed
    assert not env.done


def test_budget_applies_to_new_level_after_transition():
    env = make_env()
    env._record(raw(1), "ACTION1", {})  # level 1 done in 1 action (score capped 1.15)
    env._record(raw(1), "ACTION1", {})
    obs = env._record(raw(1), "ACTION1", {})  # 2nd action on level 2 = its whole budget
    assert obs.done and obs.outcome == OUTCOME_BUDGET
    res = env.result()
    assert res.level_actions == [1, 2] and res.levels_completed == 1
    gs = game_score(res.game_id, res.baselines, res.level_actions, res.levels_completed)
    assert gs.level_scores == [1.15, 0.0, 0.0]
    assert abs(gs.score - 1 / 6) < 1e-9  # weight 1 of 6, capped to completed fraction


def test_win_ends_episode_with_all_levels_billed():
    env = make_env()
    env._record(raw(1), "ACTION1", {})
    env._record(raw(2), "ACTION1", {})
    obs = env._record(raw(3, GameState.WIN), "ACTION1", {})
    assert obs.done and obs.outcome == OUTCOME_WIN
    res = env.result()
    assert res.level_actions == [1, 1, 1] and res.levels_completed == 3
    assert game_score(res.game_id, res.baselines, res.level_actions, 3).score == 1.0
