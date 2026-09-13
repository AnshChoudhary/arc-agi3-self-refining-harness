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


def test_repeated_runs_average_into_one_game_score():
    """--repeats averages a game's runs so a coin-flip game does not swing the set RHAE."""
    from arc_harness.scoring import GameScore, rhae

    runs = [GameScore("g1", 0.036, [], [], [], 1), GameScore("g1", 0.0, [], [], [], 0),
            GameScore("g2", 0.0, [], [], [], 0), GameScore("g2", 0.0, [], [], [], 0)]
    by_game = {}
    for r in runs:
        by_game.setdefault(r.game_id, []).append(r)
    averaged = [GameScore(g, sum(s.score for s in v) / len(v), [], [], [], max(s.levels_completed for s in v))
                for g, v in sorted(by_game.items())]
    assert [round(s.score, 4) for s in averaged] == [0.018, 0.0]
    assert abs(rhae(averaged) - 0.009) < 1e-9
