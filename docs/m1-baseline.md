# M1 — v0 student baseline

Date: 2026-09-13. Code `a4f5826`, harness `bc653ad4e040` (hand-written v0), split seed 0.

## Configuration
- Model: `deepseek-v4-flash` via NeuralWatt (OpenAI-compatible), `--effort high`.
  Reasoning off (`effort none`, the config default) burned the full 110-action budget on
  ls20 level 1 for $0.10; reasoning on solved it in 19 actions for $0.07, so the baseline
  uses reasoning explicitly. Thinking tokens bill at the output rate ($0.28/M).
- Referee: 5x human-median budget per level with a 60-action ceiling per level for this scan
  (`--max-actions-per-level 60`), level 1 only (`--max-levels 1`), 5 games in parallel.
- Loop knobs (fixed, not harness): 6 free analysis calls per action, fresh-context escalation
  before abandoning a game, retry at temperature 0.7 after a bad reply, 6-step history window.

## Results (level 1 only; game score weights level 1 at 1/N of the game)

| set | games | level 1 completed | RHAE | cost | wall |
|---|---|---|---|---|---|
| refine | 15 | 4 (ar25 24/32, ft09 31/43, ls20 17/22, tn36 11/32) | 0.0098 | $4.80 | 187 min |
| held-out | 10 | 6 (r11l 3/22, lp85 10/17, sb26 16/18, re86 29/26, sc25 27/36, su15 57/22) | **0.0176** | $2.34 | 93 min |

`actions/human` per completed level. Every completed level except re86 and su15 beat the
human median. wa30 aborted at 51/60 actions when the API key allowance ran out; its row is
marked `aborted_APIStatusError` and counts as 0.

Rows: `results/20260912T193116Z_refine_student.json`, `results/20260912T223905Z_heldout_student.json`.
Random baseline for comparison: 0.0002 / 0.0000 (`results/20260912T1754*`).

## How the failed games fail (refine trajectories, read-only)
- **Wasted actions.** 35–74% of actions in failed games changed <= 2 pixels: clicks on nothing,
  moves into walls. In failed pure-click games three quarters of clicks hit nothing.
- **Analysis hunger.** The student uses nearly the whole 6-call allowance almost every step and
  most invalid-reply counts are it asking for a seventh.
- **Notes decay.** Durable notes shrink over long games (tr87: 261 -> 39 chars); the loop had to
  escalate 5 times on that game.
- **Solved games look different**: few no-ops, a stable model of the controls in the notes by
  step 5, then a direct path to the goal.

These are procedural, game-agnostic failure modes: the coach's target in M2.

## Caveats
- Level 1 only: later levels dominate the official weighting and are untested.
- The 60-action ceiling is below the 5x rule for 20 of 25 games; a level the student would have
  finished between 60 actions and 5x human is scored 0 here.
- Single seed, single run per game; deepseek at temperature 0 is near-deterministic but retries
  and the provider are not.
