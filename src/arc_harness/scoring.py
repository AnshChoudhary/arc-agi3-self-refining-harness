"""RHAE (Relative Human Action Efficiency), matching arc_agi's EnvironmentScoreCalculator.

level_score = min((human / ai) ** 2, 1.15) for a completed level, else 0.
game_score  = sum(w_i * level_score_i) / sum(w_i), w_i = 1-indexed level number,
              then capped at the weight-fraction of completed levels (the toolkit
              applies this cap so the 1.15 bonus cannot exceed 100% overall).
total       = mean of game scores.
Scores here are in [0, 1.15]; the toolkit reports the same numbers x100.
"""

from __future__ import annotations

from dataclasses import dataclass

LEVEL_SCORE_CAP = 1.15


def level_score(human_actions: int, ai_actions: int, completed: bool) -> float:
    if not completed or ai_actions <= 0:
        return 0.0
    return min((human_actions / ai_actions) ** 2, LEVEL_SCORE_CAP)


@dataclass(frozen=True)
class GameScore:
    game_id: str
    score: float
    level_scores: list[float]
    level_actions: list[int]
    baselines: list[int]
    levels_completed: int


def game_score(
    game_id: str,
    baselines: list[int],
    level_actions: list[int],
    levels_completed: int,
) -> GameScore:
    """`level_actions[i]` is the action count on level i; entries beyond it were never reached."""
    scores: list[float] = []
    padded_actions: list[int] = []
    for i, human in enumerate(baselines):
        ai = level_actions[i] if i < len(level_actions) else 0
        padded_actions.append(ai)
        scores.append(level_score(human, ai, completed=i < levels_completed))

    weights = [i + 1 for i in range(len(baselines))]
    total_w = sum(weights)
    weighted = sum(w * s for w, s in zip(weights, scores)) / total_w
    completed_w = sum(w for w, s in zip(weights, scores) if s > 0) / total_w
    return GameScore(
        game_id=game_id,
        score=min(weighted, completed_w),
        level_scores=scores,
        level_actions=padded_actions,
        baselines=list(baselines),
        levels_completed=levels_completed,
    )


def rhae(scores: list[GameScore]) -> float:
    return sum(s.score for s in scores) / len(scores) if scores else 0.0
