"""Inner loop: one LLM plays one game with a frozen harness.

Per step: observe (analysis routine output) -> run analysis code any number of
times (free) -> state a hypothesis -> take one action -> log. The base system
prompt below is immutable (CLAUDE.md rule 2); everything learnable is read from
harness/ at episode start and never written here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from arc_harness.env import ArcEnv, InvalidAction
from arc_harness.llm import LLMClient, LLMResponse, Usage
from arc_harness.repl import HARNESS_DIR, Repl
from arc_harness.trajectory import AgentStep
from config.models import (
    EST_CALLS_PER_ACTION,
    EST_CALLS_PER_ACTION_REASONING,
    EST_INPUT_TOKENS_PER_CALL,
    EST_INPUT_TOKENS_PER_CALL_REASONING,
    EST_OUTPUT_TOKENS_PER_CALL,
    EST_OUTPUT_TOKENS_PER_CALL_REASONING,
    HISTORY_WINDOW,
    MAX_ANALYSIS_CALLS_PER_ACTION,
    MAX_BAD_REPLIES,
    MAX_CALLS_PER_ACTION,
    RETRY_TEMPERATURE,
)

# ---------------------------------------------------------------------------
# IMMUTABLE. The coach may not edit this; only files under harness/ change.
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = """You are playing an unknown interactive puzzle game on a 64x64 grid of colours (integers 0-15).
There are no instructions. You must discover the mechanics by acting and observing.

Scoring: each level is scored by (human_actions / your_actions)^2. Environment actions are the only
thing that costs score. Thinking and running analysis code are free. Take as few actions as possible,
but a level you never finish scores zero, so make progress.

Actions: ACTION1-ACTION5 are simple buttons whose meaning differs per game (often movement or a toggle).
ACTION6 is a click at (x, y) with 0 <= x, y <= 63; x is the column, y is the row. ACTION7 is undo where
available. RESET restarts the current level and costs one action. Only actions listed as available work.
Every level has an action budget (shown each step); exceeding it ends the game as a failure.

You have a persistent Python REPL with numpy as np and these variables, refreshed after every action:
  frames  - list of all frames so far (numpy arrays, index [y, x]); curr = frames[-1]; prev = frames[-2]
  steps   - list of dicts describing every action taken so far (action, data, changed_pixels, level, ...)
  level, levels_completed, actions_this_level, budget_this_level, available_actions, baselines
Helper functions from the harness are listed in the HARNESS section. Variables you define persist.

Reply with exactly one JSON object per message, nothing else. Two forms:
  {"hypothesis": "<what you currently believe and what this checks>", "python": "<code to run>"}
     runs code in the REPL; stdout and errors come back; no action is consumed
  {"hypothesis": "<why this action>", "action": "ACTION1", "x": 0, "y": 0, "notes": "<durable notes>"}
     takes one environment action; x and y only matter for ACTION6
"notes" replaces your durable notes, shown to you every step. Everything else you wrote is forgotten
after a few steps, so keep the notes complete: what each action does, objects, goal, ruled-out ideas.
Keep "hypothesis" under 60 words. Do your planning in Python (print what you need), not in prose.
"""

ACT_RE = re.compile(r"\{.*\}", re.S)


@dataclass
class ParsedReply:
    hypothesis: str | None = None
    python: str | None = None
    action: str | None = None
    x: int | None = None
    y: int | None = None
    notes: str | None = None
    error: str | None = None


def parse_reply(text: str) -> ParsedReply:
    """Extract the single JSON object from a reply; tolerate code fences and stray prose."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S)
    m = ACT_RE.search(body)
    if not m:
        return ParsedReply(error="no JSON object found")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return ParsedReply(error=f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        return ParsedReply(error="reply must be a JSON object")
    p = ParsedReply(
        hypothesis=obj.get("hypothesis"),
        python=obj.get("python"),
        action=str(obj["action"]).upper() if obj.get("action") else None,
        notes=obj.get("notes"),
    )
    for k in ("x", "y"):
        v = obj.get(k)
        if v is not None:
            try:
                setattr(p, k, int(v))
            except (TypeError, ValueError):
                return ParsedReply(error=f"{k} must be an integer")
    if p.python is None and p.action is None:
        return ParsedReply(error='reply needs either "python" or "action"')
    if p.python is not None and p.action is not None:
        return ParsedReply(error='reply must contain "python" or "action", not both')
    return p


def load_harness_text(repl: Repl) -> str:
    """Everything the student reads from harness/ at episode start."""
    playbook = (HARNESS_DIR / "playbook.md").read_text() if (HARNESS_DIR / "playbook.md").exists() else ""
    tool_lines = []
    for name in repl.tool_names:
        fn = repl.ns[name]
        doc = (fn.__doc__ or "").strip().splitlines()
        sig = _signature(fn)
        tool_lines.append(f"  {name}{sig}: {doc[0] if doc else ''}")
    return (
        "HARNESS\n\n" + playbook.strip() + "\n\nREPL helper functions:\n" + "\n".join(tool_lines)
        + "\n  analyze(): the per-step summary you see in every observation; call it yourself any time.\n"
    )


def _signature(fn) -> str:
    import inspect

    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(...)"


def load_analysis(repl: Repl) -> None:
    path = HARNESS_DIR / "analysis.py"
    if path.exists():
        repl.run(path.read_text())
    if "analyze" not in repl.ns:
        repl.run("def analyze():\n    return 'no analysis routine installed'")


class StudentAgent:
    name = "student"

    def __init__(self, llm: LLMClient, history_window: int = HISTORY_WINDOW,
                 max_analysis_calls: int = MAX_ANALYSIS_CALLS_PER_ACTION, log=print) -> None:
        self.llm = llm
        self.history_window = history_window
        self.max_analysis_calls = max_analysis_calls
        self.log = log

    def estimate_usage(self, total_action_budget: int) -> Usage:
        reasoning = getattr(self.llm, "effort", "none") != "none"
        per_action = EST_CALLS_PER_ACTION_REASONING if reasoning else EST_CALLS_PER_ACTION
        tok_in = EST_INPUT_TOKENS_PER_CALL_REASONING if reasoning else EST_INPUT_TOKENS_PER_CALL
        tok_out = EST_OUTPUT_TOKENS_PER_CALL_REASONING if reasoning else EST_OUTPUT_TOKENS_PER_CALL
        calls = int(total_action_budget * per_action)
        return Usage(input_tokens=calls * tok_in, output_tokens=calls * tok_out, calls=calls)

    # ---- the loop -----------------------------------------------------------

    def play(self, env: ArcEnv, max_levels: int | None) -> tuple[list[AgentStep], Usage]:
        repl = Repl(env)
        load_analysis(repl)
        system = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "system", "content": load_harness_text(repl)},
        ]
        history: list[dict] = []  # rolling window of (observation, final reply) pairs
        notes = ""
        steps: list[AgentStep] = []
        used = Usage()
        bad_replies = 0

        while not env.done and (max_levels is None or env.levels_completed < max_levels):
            if env.state == "GAME_OVER":
                env.reset()
                steps.append(AgentStep(hypothesis="game over -> reset (forced)"))
                history.append({"role": "user", "content": "GAME_OVER: the level was reset for you (1 action)."})
                continue

            repl.refresh(env)
            observation = self._observation(env, repl, notes)
            messages = system + history[-2 * self.history_window:] + [{"role": "user", "content": observation}]
            step = AgentStep()
            analysis_calls = 0
            acted = False
            last_text = None
            escalated = False

            while not acted:
                stuck = bad_replies >= MAX_BAD_REPLIES or step.llm_calls >= MAX_CALLS_PER_ACTION
                if stuck and not escalated:
                    # The model will not act inside this step's context. Drop that context and ask once more,
                    # plainly, before abandoning the game: one lost step is far cheaper than a lost game.
                    escalated = True
                    bad_replies = 0
                    step.invalid_attempts.append("escalated: fresh context, action demanded")
                    demand = observation + "\n\nAnalysis for this step is over. Reply with an ACTION JSON object only."
                    messages = system + history[-2 * self.history_window:] + [{"role": "user", "content": demand}]
                elif stuck:
                    step.invalid_attempts.append(f"{step.llm_calls} calls without an action")
                    steps.append(step)
                    self.log(f"  giving up: {step.llm_calls} LLM calls without an action")
                    return steps, used
                resp = self.llm.chat(messages, temperature=RETRY_TEMPERATURE if bad_replies or escalated else None)
                used.add(resp.usage)
                step.llm_calls += 1
                step.input_tokens += resp.usage.input_tokens
                step.output_tokens += resp.usage.output_tokens
                reply = parse_reply(resp.text)
                if resp.text == last_text:
                    # Deterministic sampling + unchanged context = the same reply forever. Break the loop.
                    reply = ParsedReply(error="identical to your previous reply; it was not executed")
                elif reply.python is not None and analysis_calls >= self.max_analysis_calls:
                    reply = ParsedReply(error=f"analysis limit ({self.max_analysis_calls}) reached; code not executed, reply with an action")
                last_text = resp.text
                if reply.error or resp.truncated:
                    bad_replies += 1
                    err = "reply was cut off by the length limit; keep hypothesis short" if resp.truncated else reply.error
                    step.invalid_attempts.append(err)
                    messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": f"Invalid reply ({err}). Send one JSON object as specified."}]
                    continue
                bad_replies = 0
                if reply.hypothesis:
                    step.hypothesis = reply.hypothesis

                if reply.python is not None:
                    result = repl.run(reply.python)
                    analysis_calls += 1
                    step.analysis_code.append(reply.python)
                    step.repl_outputs.append(result.render()[:2000])
                    feedback = result.render()
                    if analysis_calls >= self.max_analysis_calls:
                        feedback += f"\n\n[{analysis_calls} analysis calls used this step; you must act now]"
                    messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": feedback}]
                    continue

                # An action.
                if reply.notes is not None:
                    notes = reply.notes
                try:
                    obs = env.step(reply.action, x=reply.x, y=reply.y)
                except InvalidAction as e:
                    bad_replies += 1
                    step.invalid_attempts.append(str(e))
                    messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": f"Invalid action: {e}"}]
                    continue
                bad_replies = 0
                acted = True
                step.notes = notes
                step.reply = resp.text
                steps.append(step)
                history += [{"role": "user", "content": observation}, {"role": "assistant", "content": resp.text}]
                last = env.steps[-1]
                self.log(f"  L{last.level + 1} a{last.actions_this_level}/{last.budget_this_level} {last.action}{last.data or ''} "
                         f"-> {last.changed_pixels}px changed{' LEVEL UP' if last.level_changed else ''} "
                         f"[{step.llm_calls} calls, {step.input_tokens}+{step.output_tokens} tok]")
        return steps, used

    def _observation(self, env: ArcEnv, repl: Repl, notes: str) -> str:
        summary = repl.run("print(analyze())").render()
        last = env.steps[-1] if env.steps else None
        last_line = (f"Last action: {last.action}{last.data or ''} -> {last.changed_pixels} pixels changed"
                     + (" (LEVEL COMPLETED, new level shown)" if last.level_changed else "")) if last else "No action taken yet."
        recent = ", ".join(f"{s.action}{s.data or ''}->{s.changed_pixels}px" for s in env.steps[-8:]) or "none"
        return (
            f"== Level {env.level + 1} of {len(env.baselines)} | actions on this level: {env.actions_this_level}/{env.budget_this_level}"
            f" | state {env.state}\nAvailable actions: {env.available_actions}\n{last_line}\n"
            f"Recent actions (oldest first, pixels changed): {recent}\n\n"
            f"Your notes:\n{notes or '(empty)'}\n\nScene summary (analyze()):\n{summary}\n"
        )
