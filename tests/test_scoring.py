"""scoring.py must agree with the toolkit's EnvironmentScoreCalculator."""

from arc_agi import EnvironmentScoreCalculator

from arc_harness.scoring import game_score, level_score


def toolkit_score(baselines, level_actions, levels_completed) -> float:
    calc = EnvironmentScoreCalculator()
    for i, human in enumerate(baselines):
        ai = level_actions[i] if i < len(level_actions) else 0
        calc.add_level(level_index=i + 1, completed=i < levels_completed, actions_taken=ai, baseline_actions=human)
    return calc.to_score().score / 100


def test_level_score_shape():
    assert level_score(10, 10, True) == 1.0
    assert level_score(10, 20, True) == 0.25
    assert level_score(10, 5, True) == 1.15  # capped
    assert level_score(10, 3, False) == 0.0


def test_matches_toolkit():
    cases = [
        ([22, 123, 73], [22, 150, 40], 3),
        ([22, 123, 73], [30, 400], 1),
        ([22, 123, 73], [110], 0),
        ([5, 5], [2, 2], 2),  # capped levels; overall capped at 1.0
        ([7, 18, 44, 61], [7, 18, 44, 61], 4),
    ]
    for baselines, actions, done in cases:
        ours = game_score("x", baselines, actions, done).score
        theirs = toolkit_score(baselines, actions, done)
        assert abs(ours - theirs) < 1e-9, (baselines, actions, done, ours, theirs)
