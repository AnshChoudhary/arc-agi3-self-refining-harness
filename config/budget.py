"""Per-level action budget.

ARC's own evals stop a level once the agent has spent 5x the human median
action count on it. We enforce the same rule inside env.py (CLAUDE.md rule 5)
so the LLM can never talk its way past it.
"""

BUDGET_MULTIPLIER: int = 5


def level_budget(human_baseline_actions: int, multiplier: int = BUDGET_MULTIPLIER) -> int:
    """Max environment actions allowed on one level, RESETs included."""
    if human_baseline_actions <= 0:
        raise ValueError(f"human baseline must be positive, got {human_baseline_actions}")
    return multiplier * human_baseline_actions
