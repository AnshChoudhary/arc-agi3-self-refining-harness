"""The scripted solver on a real game: exercises level transitions end to end.

This is the M0 'scripted agent on one easy game' check. It pins the exact
per-level billing on real level changes, which the synthetic tests only mimic.
"""

from arc_harness.agents import ScriptedHandleAgent, parse_scene
from arc_harness.env import ArcEnv
from arc_harness.scoring import game_score

GAME = "r11l"  # sanity-check target only; never referenced by harness state


def test_parse_initial_scene():
    env = ArcEnv(GAME, seed=0, offline=True)
    scene = parse_scene(env.frame)
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.goal is not None and len(obj.handles) == 2
    assert sum(h.selected for h in obj.handles) == 1


def test_level_one_in_three_actions():
    env = ArcEnv(GAME, seed=0, offline=True)
    ScriptedHandleAgent().play(env, max_levels=1)
    assert env.levels_completed == 1
    assert env.level_actions == [3] and env.actions_this_level == 0
    assert env.steps[-1].level_changed and env.steps[-1].level == 0
    assert env.level == 1 and env.budget_this_level == 5 * env.baselines[1]


def test_clears_first_three_levels_at_cap():
    env = ArcEnv(GAME, seed=0, offline=True)
    ScriptedHandleAgent().play(env, max_levels=None)
    res = env.result()
    assert res.levels_completed >= 3
    gs = game_score(res.game_id, res.baselines, res.level_actions, res.levels_completed)
    assert gs.level_scores[:3] == [1.15, 1.15, 1.15]
    assert all(a < b for a, b in zip(res.level_actions[:3], res.baselines))  # beat the human baseline
