"""Persistent Python namespace the student runs analysis code in.

The LLM should never read 4,096 ints as text; it writes numpy against the
frame history here instead. Only *data* is exposed: the env object is kept
out so every action still passes through the student loop, the budget, and
the trajectory log. Variables the LLM defines persist across steps; the data
views are refreshed after every action.

Helper functions come from harness/tools/*.py (learned state, editable by
the coach). The REPL itself is plumbing and stays fixed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import signal
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arc_harness.env import PROJECT_ROOT, ArcEnv

HARNESS_DIR = PROJECT_ROOT / "harness"
TOOLS_DIR = HARNESS_DIR / "tools"
MAX_OUTPUT_CHARS = 4000  # anything longer would just flood the LLM context

DATA_NAMES = (
    "frames", "curr", "prev", "steps", "level", "levels_completed",
    "actions_this_level", "budget_this_level", "available_actions", "baselines",
)


@dataclass
class ReplResult:
    code: str
    stdout: str
    error: str | None
    elapsed_s: float
    truncated: bool

    @property
    def ok(self) -> bool:
        return self.error is None

    def render(self) -> str:
        """Compact text for the LLM."""
        out = self.stdout
        if self.truncated:
            out += f"\n... [output truncated to {MAX_OUTPUT_CHARS} chars]"
        if self.error:
            out += ("\n" if out else "") + self.error
        return out or "(no output)"


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def load_harness_tools(namespace: dict, tools_dir: Path = TOOLS_DIR) -> list[str]:
    """Exec each harness/tools/*.py and expose its public callables. Returns their names."""
    names: list[str] = []
    for path in sorted(tools_dir.glob("*.py")):
        scope: dict = {"np": np, "__name__": f"harness.tools.{path.stem}"}
        exec(compile(path.read_text(), str(path), "exec"), scope)
        for k, v in scope.items():
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", None) == scope["__name__"]:
                namespace[k] = v
                names.append(k)
    return names


def harness_fingerprint(harness_dir: Path = HARNESS_DIR) -> str:
    """Content hash of the learned state (history/ excluded). Recorded with every eval run."""
    h = hashlib.sha1()
    for path in sorted(harness_dir.rglob("*")):
        if path.is_dir() or "history" in path.relative_to(harness_dir).parts or path.name == ".gitkeep":
            continue
        h.update(str(path.relative_to(harness_dir)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


class Repl:
    def __init__(self, env: ArcEnv, timeout_s: float = 10.0, load_tools: bool = True) -> None:
        self.timeout_s = timeout_s
        self.ns: dict = {"np": np}
        self.tool_names = load_harness_tools(self.ns) if load_tools else []
        self.refresh(env)

    def refresh(self, env: ArcEnv) -> None:
        """Re-point the data views at the env's current history. User variables survive."""
        frames = list(env.frames)
        self.ns.update(
            frames=frames,
            curr=frames[-1],
            prev=frames[-2] if len(frames) > 1 else frames[-1],
            steps=[asdict(s) for s in env.steps],
            level=env.level,
            levels_completed=env.levels_completed,
            actions_this_level=env.actions_this_level,
            budget_this_level=env.budget_this_level,
            available_actions=list(env.available_actions),
            baselines=list(env.baselines),
        )

    def run(self, code: str) -> ReplResult:
        buf = io.StringIO()
        error: str | None = None
        use_alarm = threading.current_thread() is threading.main_thread() and self.timeout_s > 0
        t0 = time.perf_counter()
        try:
            if use_alarm:
                signal.signal(signal.SIGALRM, _alarm)
                signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<repl>", "exec"), self.ns)
        except _Timeout:
            error = f"TimeoutError: code exceeded {self.timeout_s}s"
        except BaseException:  # the LLM's code can raise anything; report, don't crash the loop
            error = _short_traceback()
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.perf_counter() - t0

        out = buf.getvalue()
        truncated = len(out) > MAX_OUTPUT_CHARS
        return ReplResult(code, out[:MAX_OUTPUT_CHARS], error, elapsed, truncated)


def _short_traceback() -> str:
    """Only the frames from the LLM's code, not the REPL internals."""
    lines = traceback.format_exc().splitlines()
    keep = [ln for ln in lines if "<repl>" in ln or not ln.startswith("  File")]
    return "\n".join(keep[-6:])
