"""arc_agi wrapper: numpy frame history plus hard per-level action budget.

Behaviour below was verified against arc-agi 0.9.9 / arcengine 0.9.3 source
(docs/toolkit-notes.md has the details). The points that shape this file:

* The local engine executes *any* GameAction, even ones not in
  ``available_actions``, and the official scorecard would bill them. We reject
  them before they reach the engine, so a rejected action costs nothing.
* RESET counts as one action in the official scorecard (except the implicit
  reset that starts the game). RESET with zero actions taken on the current
  level is a *full* reset that discards all level progress, so we refuse it.
* Level index == ``levels_completed`` until the game is won; the action that
  completes a level is billed to that level, matching the scorecard.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from config.budget import BUDGET_MULTIPLIER, level_budget

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS_DIR = PROJECT_ROOT / "environment_files"
# The toolkit insists on a recordings dir for its own scorecards; keep it out of the way.
TOOLKIT_RECORDINGS_DIR = PROJECT_ROOT / ".toolkit_recordings"

OUTCOME_WIN = "win"
OUTCOME_BUDGET = "budget_exhausted"


class InvalidAction(ValueError):
    """Raised before anything is sent to the engine; never consumes budget."""


@dataclass
class Step:
    """One environment action, as recorded for the trajectory log."""

    index: int
    level: int
    action: str
    data: dict
    state: str
    levels_completed: int
    actions_this_level: int
    budget_this_level: int
    n_frames: int
    frame_hash: str
    changed_pixels: int
    level_changed: bool
    full_reset: bool


@dataclass
class Observation:
    frame: np.ndarray
    frames: list[np.ndarray]
    state: str
    level: int
    levels_completed: int
    win_levels: int
    available_actions: list[str]
    actions_this_level: int
    budget_this_level: int
    done: bool
    outcome: str | None = None


@dataclass
class EpisodeResult:
    """Everything scoring.py needs; nothing the coach could overfit to beyond this."""

    game_id: str
    baselines: list[int]
    level_actions: list[int]  # one entry per level attempted; completed levels first
    levels_completed: int
    outcome: str | None
    total_actions: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_actions = sum(self.level_actions)


_arcade_cache: dict[bool, Arcade] = {}


def _quiet_logger() -> logging.Logger:
    # The toolkit also logs via module loggers (arc_agi.scorecard etc.); keep those quiet too.
    # arc_agi.scorecard sets its own INFO level + stdout handler at import, so name it explicitly.
    for name in ("arc_agi", "arc_agi.scorecard"):
        logging.getLogger(name).setLevel(logging.ERROR)
    log = logging.getLogger("arc_harness.toolkit")
    log.setLevel(logging.ERROR)
    log.propagate = False
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


def _is_downloaded(game_id: str) -> bool:
    base, _, version = game_id.partition("-")
    root = ENVIRONMENTS_DIR / base
    if version:
        return (root / version / "metadata.json").exists()
    return root.exists() and any(root.rglob("metadata.json"))


def get_arcade(offline: bool) -> Arcade:
    """OFFLINE never touches the network; NORMAL downloads sources into environment_files/."""
    if offline not in _arcade_cache:
        _arcade_cache[offline] = Arcade(
            operation_mode=OperationMode.OFFLINE if offline else OperationMode.NORMAL,
            environments_dir=str(ENVIRONMENTS_DIR),
            recordings_dir=str(TOOLKIT_RECORDINGS_DIR),
            logger=_quiet_logger(),
        )
    return _arcade_cache[offline]


def list_game_ids() -> list[str]:
    """Versioned ids (e.g. 'ls20-9607627b') of every public game, from the API."""
    return sorted(e.game_id for e in get_arcade(offline=False).get_environments())


def baselines_for(game_id: str, offline: bool | None = None) -> list[int]:
    """Human baseline per level without instantiating the game (for cost projection)."""
    if offline is None:
        offline = _is_downloaded(game_id)
    base = game_id.split("-", 1)[0]
    for info in get_arcade(offline).get_environments():
        if info.game_id == game_id or (info.game_id.split("-", 1)[0] == base and "-" not in game_id):
            if not info.baseline_actions:
                raise RuntimeError(f"{info.game_id} has no human baseline; RHAE is undefined")
            return list(info.baseline_actions)
    raise KeyError(f"{game_id!r} not known to the toolkit (offline={offline})")


def _hash(frame: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()[:16]


class ArcEnv:
    """One episode of one game. Create a new instance per episode."""

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        offline: bool | None = None,
        budget_multiplier: int = BUDGET_MULTIPLIER,
    ) -> None:
        if offline is None:
            offline = _is_downloaded(game_id)
        env = get_arcade(offline).make(game_id, seed=seed)
        if env is None:
            raise RuntimeError(f"could not create environment for {game_id!r}")
        info = env.info
        if not info.baseline_actions:
            raise RuntimeError(f"{info.game_id} has no human baseline; RHAE is undefined")

        self._env = env
        self.game_id: str = info.game_id
        self.seed = seed
        self.baselines: list[int] = list(info.baseline_actions)
        self.budget_multiplier = budget_multiplier

        self.frames: list[np.ndarray] = []  # last frame of every step, index 0 = initial
        self.steps: list[Step] = []
        self.level_actions: list[int] = []  # actions spent on each completed level
        self.actions_this_level = 0
        self.done = False
        self.outcome: str | None = None

        first = env.observation_space
        if first is None or first.is_empty():
            raise RuntimeError(f"{self.game_id}: empty initial frame")
        self._last = first
        self.frames.append(first.frame[-1].copy())

    # ---- read-only views -------------------------------------------------

    @property
    def level(self) -> int:
        return min(self._last.levels_completed, len(self.baselines) - 1)

    @property
    def levels_completed(self) -> int:
        return self._last.levels_completed

    @property
    def state(self) -> str:
        return self._last.state.value

    @property
    def budget_this_level(self) -> int:
        return level_budget(self.baselines[self.level], self.budget_multiplier)

    @property
    def available_actions(self) -> list[str]:
        return [GameAction.from_id(i).name for i in self._last.available_actions]

    @property
    def frame(self) -> np.ndarray:
        return self.frames[-1]

    def observe(self) -> Observation:
        return Observation(
            frame=self.frames[-1],
            frames=list(self._last.frame),
            state=self.state,
            level=self.level,
            levels_completed=self.levels_completed,
            win_levels=self._last.win_levels,
            available_actions=self.available_actions,
            actions_this_level=self.actions_this_level,
            budget_this_level=self.budget_this_level,
            done=self.done,
            outcome=self.outcome,
        )

    # ---- acting ------------------------------------------------------------

    def step(self, action: GameAction | str, x: int | None = None, y: int | None = None) -> Observation:
        act = GameAction.from_name(action) if isinstance(action, str) else action
        if act is GameAction.RESET:
            return self.reset()
        self._check_can_act()
        if act.name not in self.available_actions:
            raise InvalidAction(f"{act.name} not available; choose from {self.available_actions}")
        data: dict = {}
        if act.is_complex():
            if x is None or y is None or not (0 <= x <= 63 and 0 <= y <= 63):
                raise InvalidAction(f"{act.name} needs integer x, y in [0, 63]; got {x=}, {y=}")
            data = {"x": int(x), "y": int(y)}
        elif x is not None or y is not None:
            raise InvalidAction(f"{act.name} takes no coordinates")

        raw = self._env.step(act, data=data)
        if raw is None or raw.is_empty():
            raise RuntimeError(f"{self.game_id}: engine returned no frame for {act.name}")
        return self._record(raw, act.name, data)

    def reset(self) -> Observation:
        """Restart the current level. Costs one action, like the official scorecard."""
        self._check_can_act()
        if self.actions_this_level == 0:
            raise InvalidAction(
                "RESET with no actions on this level restarts the whole game; take an action first"
            )
        raw = self._env.reset()
        if raw is None or raw.is_empty():
            raise RuntimeError(f"{self.game_id}: engine returned no frame for RESET")
        return self._record(raw, "RESET", {})

    def _check_can_act(self) -> None:
        if self.done:
            raise InvalidAction(f"episode is over ({self.outcome})")

    def _record(self, raw, action_name: str, data: dict) -> Observation:
        prev_frame = self.frames[-1]
        prev_completed = self._last.levels_completed
        self._last = raw
        self.actions_this_level += 1

        frame = raw.frame[-1].copy()
        level_changed = raw.levels_completed > prev_completed
        # Bill the completing action to the level it completed.
        billed_level = prev_completed
        budget = level_budget(self.baselines[min(billed_level, len(self.baselines) - 1)], self.budget_multiplier)

        if level_changed:
            self.level_actions.append(self.actions_this_level)
            self.actions_this_level = 0
        if raw.state == GameState.WIN:
            self.done, self.outcome = True, OUTCOME_WIN
        elif not level_changed and self.actions_this_level >= budget:
            # Budget exhausted without finishing the level: the level (and so the run) is failed.
            self.done, self.outcome = True, OUTCOME_BUDGET

        self.frames.append(frame)
        self.steps.append(
            Step(
                index=len(self.steps) + 1,
                level=billed_level,
                action=action_name,
                data=data,
                state=raw.state.value,
                levels_completed=raw.levels_completed,
                actions_this_level=self.level_actions[-1] if level_changed else self.actions_this_level,
                budget_this_level=budget,
                n_frames=len(raw.frame),
                frame_hash=_hash(frame),
                changed_pixels=int((frame != prev_frame).sum()) if frame.shape == prev_frame.shape else -1,
                level_changed=level_changed,
                full_reset=bool(raw.full_reset),
            )
        )
        return self.observe()

    # ---- results -----------------------------------------------------------

    def result(self) -> EpisodeResult:
        attempted = list(self.level_actions)
        if self.outcome != OUTCOME_WIN and self.actions_this_level > 0:
            attempted.append(self.actions_this_level)
        return EpisodeResult(
            game_id=self.game_id,
            baselines=self.baselines,
            level_actions=attempted,
            levels_completed=self.levels_completed,
            outcome=self.outcome,
        )
