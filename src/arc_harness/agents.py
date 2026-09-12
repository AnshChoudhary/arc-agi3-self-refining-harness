"""Agent protocol shared by scripts/eval.py and the sanity baselines.

student.py (M1) implements the same interface, so the referee never needs to
know which agent it is running.
"""

from __future__ import annotations

import random
from typing import Protocol

from arc_harness.env import ArcEnv, InvalidAction
from arc_harness.llm import Usage
from arc_harness.trajectory import AgentStep


class Agent(Protocol):
    name: str

    def play(self, env: ArcEnv, max_levels: int | None) -> tuple[list[AgentStep], Usage]:
        """Act until env.done or max_levels completed. Must not touch the harness."""

    def estimate_usage(self, total_action_budget: int) -> Usage:
        """Upper-bound token usage for a run with this many budgeted actions (cost projection)."""


class RandomAgent:
    """Uniform random over available actions. Expected RHAE ~0; exercises the plumbing for free."""

    name = "random"

    def __init__(self, agent_seed: int = 0) -> None:
        self.agent_seed = agent_seed

    def play(self, env: ArcEnv, max_levels: int | None) -> tuple[list[AgentStep], Usage]:
        # Seed per game so a game's result does not depend on its position in the eval set.
        self.rng = random.Random(f"{self.agent_seed}:{env.game_id}")
        steps: list[AgentStep] = []
        while not env.done and (max_levels is None or env.levels_completed < max_levels):
            obs = env.observe()
            if obs.state == "GAME_OVER":
                env.reset()
                steps.append(AgentStep(hypothesis="game over -> reset"))
                continue
            action = self.rng.choice(obs.available_actions)
            try:
                if action == "ACTION6":
                    env.step(action, x=self.rng.randrange(64), y=self.rng.randrange(64))
                else:
                    env.step(action)
            except InvalidAction as e:  # cannot happen when choosing from available_actions
                steps.append(AgentStep(invalid_attempts=[str(e)]))
                continue
            steps.append(AgentStep())
        return steps, Usage()

    def estimate_usage(self, total_action_budget: int) -> Usage:
        return Usage()


# ---------------------------------------------------------------------------
# Scripted sanity solver for one game (M0). This is a referee-side tool: it
# validates the plumbing on real level transitions. It is not harness state and
# the student never sees it.
#
# Mechanic (reverse-engineered from the game source): each object sits at the
# centroid of its handle diamonds. Clicking empty space moves the selected
# (color 0) handle there; clicking a diamond selects it. A level is won when
# every object overlaps the hollow X of its own color. Handle centres carry the
# colour of their object.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402

BACKGROUND, LINE, SELECTED, UNSELECTED, CORE = 5, 1, 0, 3, 6
COUNTER_COLUMN = 0  # x == 0 is the game's own action counter bar


def components8(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components as arrays of (y, x). Pure numpy/python; 64x64 is tiny."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for y0, x0 in np.argwhere(mask):
        if seen[y0, x0]:
            continue
        stack, pts = [(int(y0), int(x0))], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        out.append(np.array(pts))
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _bbox_center(pts: np.ndarray) -> tuple[int, int, int, int]:
    """(cx, cy, h, w) of a component's bounding box."""
    y0, y1, x0, x1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    return int((x0 + x1) // 2), int((y0 + y1) // 2), int(y1 - y0 + 1), int(x1 - x0 + 1)


@dataclass
class Handle:
    x: int
    y: int
    selected: bool
    color: int  # centre pixel colour = owning object's colour


@dataclass
class Obj:
    color: int
    x: int
    y: int
    goal: tuple[int, int] | None
    handles: list[Handle] = field(default_factory=list)


@dataclass
class Scene:
    objects: list[Obj]
    allowed: set[int]  # colours a handle or object may be placed over


def parse_scene(frame: np.ndarray) -> Scene:
    play = frame.copy()
    play[:, COUNTER_COLUMN] = BACKGROUND
    handles: list[Handle] = []
    for color, sel in ((SELECTED, True), (UNSELECTED, False)):
        for c in components8(play == color):
            cx, cy, h, w = _bbox_center(c)
            if h == 5 and w == 5 and 10 <= len(c) <= 14:
                handles.append(Handle(cx, cy, sel, int(play[cy, cx])))
    objects: dict[int, Obj] = {}
    goals: dict[int, tuple[int, int]] = {}
    for color in np.unique(play):
        color = int(color)
        if color in (BACKGROUND, LINE, SELECTED, UNSELECTED, CORE):
            continue
        mask = play == color
        for h_ in handles:
            mask[h_.y, h_.x] = False  # handle centre pixels
        for c in components8(mask):
            cx, cy, h, w = _bbox_center(c)
            if h == 5 and w == 5 and play[cy, cx] == CORE:
                objects[color] = Obj(color, cx, cy, None)
                mask[c[:, 0], c[:, 1]] = False
        # The hollow X goal is four diagonal arms with a one-pixel gap; dilate to join them.
        for c in components8(_dilate(mask)):
            pts = c[mask[c[:, 0], c[:, 1]]]
            if len(pts) == 0:
                continue
            cx, cy, h, w = _bbox_center(pts)
            if h == 7 and w == 7:
                goals[color] = (cx, cy)
    for color, obj in objects.items():
        obj.goal = goals.get(color)
        obj.handles = [h for h in handles if h.color == color]
    allowed = {BACKGROUND, LINE, SELECTED, UNSELECTED, CORE, *objects, *goals}
    return Scene(list(objects.values()), allowed)


def _footprint_ok(frame: np.ndarray, x: int, y: int, allowed: set[int]) -> bool:
    """A 5x5 sprite centred at (x, y) fits on screen and touches no wall colour."""
    if not (3 <= x <= 60 and 2 <= y <= 61):
        return False
    patch = frame[y - 2:y + 3, x - 2:x + 3]
    return all(int(v) in allowed for v in np.unique(patch))


def _split(total: int, k: int) -> list[int]:
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def _spiral(radius: int):
    """Offsets (dx, dy) in increasing Chebyshev distance, (0, 0) first."""
    yield 0, 0
    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                yield dx, dy
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                yield dx, dy


Click = tuple[int, int, str]


def _target_ok(frame: np.ndarray, tx: int, ty: int, allowed: set[int], avoid: list[tuple[int, int]]) -> bool:
    """Handle may land here: on screen, no wall under its 5x5, and the click won't hit another diamond."""
    if not _footprint_ok(frame, tx, ty, allowed):
        return False
    return not any(abs(tx - ox) <= 2 and abs(ty - oy) <= 2 for ox, oy in avoid)


def _plan_group(obj: Obj, group: list[Handle], frame: np.ndarray, allowed: set[int],
                avoid: list[tuple[int, int]], banned: set[tuple], radius: int = 30) -> list[tuple[Handle, int, int]] | None:
    """Targets for `group` (moved in this order) that put obj on its goal without ever parking it on a hazard."""
    k, n = len(group), len(obj.handles)
    dx, dy = obj.goal[0] - obj.x, obj.goal[1] - obj.y
    shift = (n * dx, n * dy)
    ideal = [(h.x + sx, h.y + sy) for h, (sx, sy) in zip(group, zip(_split(shift[0], k), _split(shift[1], k)))]
    obj_allowed = allowed | {obj.color}

    def valid(h: Handle, tx: int, ty: int, extra: list[tuple[int, int]]) -> bool:
        return (h.x, h.y, tx, ty) not in banned and _target_ok(frame, tx, ty, allowed, avoid + extra)

    def seq_ok(targets: list[tuple[Handle, int, int]]) -> bool:
        # The engine re-centres the object after every single move; each stop must be hazard-free.
        pos = {id(h): (h.x, h.y) for h in obj.handles}
        for h, tx, ty in targets:
            pos[id(h)] = (tx, ty)
            ox = sum(x for x, _ in pos.values()) // n
            oy = sum(y for _, y in pos.values()) // n
            if not _footprint_ok(frame, ox, oy, obj_allowed):
                return False
        return True

    if k == 1:
        h, (tx, ty) = group[0], ideal[0]
        t = [(h, tx, ty)]
        return t if valid(h, tx, ty, []) and seq_ok(t) else None

    # Middle handles snap to the nearest valid cell; the first is swept so the last lands exactly.
    middle: list[tuple[Handle, int, int]] = []
    for h, (ix, iy) in zip(group[1:-1], ideal[1:-1]):
        placed = [(x, y) for _, x, y in middle]
        found = next(((ix + ox, iy + oy) for ox, oy in _spiral(radius) if valid(h, ix + ox, iy + oy, placed)), None)
        if found is None:
            return None
        middle.append((h, *found))
    placed = [(x, y) for _, x, y in middle]
    first, last = group[0], group[-1]
    sum_x = sum(h.x for h in group) + shift[0]
    sum_y = sum(h.y for h in group) + shift[1]
    for ox, oy in _spiral(radius):
        fx, fy = ideal[0][0] + ox, ideal[0][1] + oy
        lx = sum_x - fx - sum(x for _, x, _ in middle)
        ly = sum_y - fy - sum(y for _, _, y in middle)
        if not (valid(first, fx, fy, placed) and valid(last, lx, ly, placed + [(fx, fy)])):
            continue
        t = [(first, fx, fy), *middle, (last, lx, ly)]
        if seq_ok(t):
            return t
    return None


def plan_moves(scene: Scene, frame: np.ndarray, banned: set[tuple]) -> list[Click] | None:
    """Cheapest click sequence that moves one misplaced object onto its goal, or None."""
    from itertools import combinations, permutations

    best: tuple[int, list[Click]] | None = None
    all_handles = [h for o in scene.objects for h in o.handles]
    for obj in scene.objects:
        if obj.goal is None or not obj.handles:
            continue
        if (obj.goal[0], obj.goal[1]) == (obj.x, obj.y):
            continue
        if not _footprint_ok(frame, *obj.goal, scene.allowed | {obj.color}):
            continue
        for k in range(1, len(obj.handles) + 1):
            for combo in combinations(obj.handles, k):
                for group in permutations(combo):
                    # A select click is only free for the currently selected handle, and only if it moves first.
                    cost = k + sum(1 for i, h in enumerate(group) if not (h.selected and i == 0))
                    if best is not None and cost >= best[0]:
                        continue
                    avoid = [(h.x, h.y) for h in all_handles if h not in group]
                    targets = _plan_group(obj, list(group), frame, scene.allowed, avoid, banned)
                    if targets is None:
                        continue
                    clicks: list[Click] = []
                    for i, (h, tx, ty) in enumerate(targets):
                        if not (h.selected and i == 0):
                            clicks.append((h.x, h.y, f"select handle of colour {obj.color} at ({h.x},{h.y})"))
                        clicks.append((tx, ty, f"move handle ({h.x},{h.y}) -> ({tx},{ty}); object {obj.color} to {obj.goal}"))
                    best = (len(clicks), clicks)
    return best[1] if best else None


class ScriptedHandleAgent:
    """Solves the handle/centroid game with numpy parsing and a tiny planner. RHAE should be ~1."""

    name = "scripted"

    def play(self, env: ArcEnv, max_levels: int | None) -> tuple[list[AgentStep], Usage]:
        steps: list[AgentStep] = []
        banned: set[tuple] = set()
        stale = 0
        while not env.done and (max_levels is None or env.levels_completed < max_levels):
            if env.state == "GAME_OVER":
                env.reset()
                steps.append(AgentStep(hypothesis="game over -> reset"))
                continue
            scene = parse_scene(env.frame)
            plan = plan_moves(scene, env.frame, banned)
            if plan is None:
                steps.append(AgentStep(hypothesis="no valid plan; giving up"))
                break
            level_before = env.levels_completed
            for x, y, note in plan:
                before = parse_scene(env.frame)
                obs = env.step("ACTION6", x=x, y=y)
                steps.append(AgentStep(hypothesis=note))
                if obs.done or obs.state == "GAME_OVER" or obs.levels_completed != level_before:
                    break
                after = parse_scene(obs.frame)
                if note.startswith("move") and _same_positions(before, after):
                    # The engine reverted the move (hazard); never retry this exact move.
                    hx, hy = [int(v) for v in note.split("(")[1].split(")")[0].split(",")]
                    banned.add((hx, hy, x, y))
                    break
            stale = stale + 1 if env.levels_completed == level_before else 0
            if stale > 20:
                steps.append(AgentStep(hypothesis="stuck; giving up"))
                break
        return steps, Usage()

    def estimate_usage(self, total_action_budget: int) -> Usage:
        return Usage()


def _same_positions(a: Scene, b: Scene) -> bool:
    pa = sorted((h.x, h.y) for o in a.objects for h in o.handles)
    pb = sorted((h.x, h.y) for o in b.objects for h in o.handles)
    return pa == pb
