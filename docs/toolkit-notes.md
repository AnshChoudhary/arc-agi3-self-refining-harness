# Toolkit notes — answers to SPEC §9

Verified by reading the installed source of `arc-agi 0.9.9` and `arcengine 0.9.3`
(`arc_agi/{base,wrapper,local_wrapper,scorecard}.py`, `arcengine/{enums,base_game}.py`)
and by probing one game locally. Not from docs or memory.

## FrameDataRaw attributes
`arcengine.enums.FrameDataRaw` (pydantic):

| attr | type | notes |
|---|---|---|
| `frame` | `list[np.ndarray]` | each `(64, 64)` `int8`; one per action normally, more when an action animates. Last element is the resulting state. |
| `state` | `GameState` | `NOT_PLAYED` / `NOT_FINISHED` / `WIN` / `GAME_OVER` |
| `levels_completed` | `int` | == current level index until `WIN` |
| `win_levels` | `int` | number of levels in the game |
| `available_actions` | `list[int]` | ids 1–7; `EnvironmentWrapper.action_space` maps them to `GameAction` |
| `action_input` | `ActionInput` | echo of the action just performed |
| `full_reset` | `bool` | True when a RESET restarted the whole game |
| `game_id`, `guid` | `str` | |

There is **no action counter** on the frame. The engine's private `_action_count`
resets to 0 on every level change and excludes RESET, so we count in `env.py`.

`GameAction`: `RESET=0`, `ACTION1..5` (simple), `ACTION6` (needs `{"x","y"}` in 0–63), `ACTION7` (simple).
The local engine **executes any action regardless of `available_actions`** — `env.py` rejects those itself.

## Human baselines
`EnvironmentInfo.baseline_actions: list[int]`, one entry per level. Served by
`GET /api/games` (anonymous API key works) and persisted into
`environment_files/<game>/<version>/metadata.json` when `Arcade.make()` downloads a
game, so it is available offline afterwards. 25 public games, all with baselines.
Budget in `config/budget.py` is `5 × baseline_actions[level]`.

## reset() semantics
`env.reset()` sends `GameAction.RESET`. In `ARCBaseGame.handle_reset`:
- if `_action_count == 0` on the current level, or state is `WIN` → **full reset** (all levels reset, `levels_completed = 0`);
- otherwise → **level reset** (current level reloaded, progress kept).

Scorecard accounting (`Scorecard.update_scorecard`): a RESET that is a full reset opens a
new play (free); any other RESET does `resets += 1` **and `actions += 1`**. So a mid-level
RESET costs one action. `env.py` refuses RESET when no action has been taken on the current
level (it would silently wipe progress) and bills every other RESET as one action.

After `GAME_OVER`, non-RESET actions return an empty frame; you must RESET (level reset, 1 action).

## Does ACTION7 (undo) count?
Yes. `update_scorecard` increments the action count for ids `[1..7]`. Most games don't expose it
(`available_actions` defaults to `[1..6]`).

## Level accounting
The scorecard records `(levels_completed, cumulative_actions)` on every level transition and
attributes the completing action to the level it completed. `env.py` does the same.

## Scoring (matches `EnvironmentScoreCalculator`)
`level = min((human/ai)^2, 1.15)` if completed else 0; game = weighted mean, weight = 1-indexed
level, then capped at the weight-fraction of completed levels. `tests/test_scoring.py` asserts
our implementation equals the toolkit's on several cases.

## Reproducibility
`Arcade.make(game_id, seed=...)` passes `seed` to games whose constructor accepts it.
Versioned ids (`ls20-9607627b`) are pinned in `config/split.json`.
