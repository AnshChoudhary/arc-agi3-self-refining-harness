"""Outer loop: read refine-set trajectories, propose small evidence-backed harness edits.

HELD-OUT ISOLATION (CLAUDE.md rule 1). This module reads trajectories from
trajectories/refine/ only. It never imports load_split, never opens
config/split.json, and every path it opens is asserted to live under the
refine directory. tests/test_coach.py checks both the source and the runtime.

The coach only proposes. harness_state.py validates, applies, logs, snapshots.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from arc_harness.env import PROJECT_ROOT
from arc_harness.harness_state import TAGS, Edit
from arc_harness.llm import LLMClient
from arc_harness.trajectory import TRAJECTORIES_DIR

REFINE_DIR = TRAJECTORIES_DIR / "refine"


def _assert_refine(path: Path) -> Path:
    resolved = path.resolve()
    if REFINE_DIR.resolve() not in resolved.parents:
        raise PermissionError(f"coach may only read trajectories/refine/; refused {path}")
    return resolved


@dataclass
class TrajectoryDigest:
    trajectory_id: str
    text: str
    solved: bool


def load_refine_trajectories(agent: str = "student", harness_fingerprint: str | None = None) -> list[dict]:
    """Latest trajectory per game from the refine set, for one agent (and optionally one harness)."""
    latest: dict[str, dict] = {}
    for path in sorted(REFINE_DIR.glob("*.json")):
        with _assert_refine(path).open() as f:
            t = json.load(f)
        if t.get("agent") != agent:
            continue
        if harness_fingerprint and not str(t.get("harness_snapshot", "")).endswith(harness_fingerprint):
            continue
        prev = latest.get(t["game_id"])
        if prev is None or t["started_at"] > prev["started_at"]:
            latest[t["game_id"]] = t
    return [latest[k] for k in sorted(latest)]


def _compress_actions(env_steps: list[dict]) -> str:
    """'A1(52) A3(2)x7 ...' — action name with pixels changed, runs collapsed."""
    items = [(s["action"].replace("ACTION", "A"), s["changed_pixels"]) for s in env_steps]
    out: list[str] = []
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1] == items[i]:
            j += 1
        a, px = items[i]
        out.append(f"{a}({px})" + (f"x{j - i + 1}" if j > i else ""))
        i = j + 1
    return " ".join(out)


def digest(t: dict, max_chars: int = 2600) -> TrajectoryDigest:
    """Compact, game-agnostic account of one trajectory: metrics, action log, notes, hypotheses."""
    es, ags = t["env_steps"], t["agent_steps"]
    solved = t["levels_completed"] > 0
    n = max(len(es), 1)
    noop = sum(1 for s in es if s["changed_pixels"] <= 2) / n
    calls = sum(a["llm_calls"] for a in ags) / n
    invalid = sum(len(a["invalid_attempts"]) for a in ags)
    limit_hits = sum(1 for a in ags for e in a["invalid_attempts"] if "analysis limit" in e)
    escalations = sum(1 for a in ags for e in a["invalid_attempts"] if e.startswith("escalated"))
    notes = [a["notes"] for a in ags if a.get("notes")]
    avail = sorted({s["action"] for s in es}) if es else []
    lines = [
        f"### {t['trajectory_id']}",
        f"outcome={t['outcome'] or 'stopped'} levels_completed={t['levels_completed']} "
        f"actions_per_level={t['level_actions']} human_baselines={t['baselines'][:len(t['level_actions']) or 1]}",
        f"actions_used={sorted(a.replace('ACTION', 'A') for a in avail)} noop_rate={noop:.0%} llm_calls_per_action={calls:.1f} "
        f"invalid_replies={invalid} analysis_limit_hits={limit_hits} escalations={escalations}",
        f"action_log: {_compress_actions(es)}",
    ]
    if notes:
        lines.append(f"notes_after_step_{min(5, len(notes))}: {notes[min(5, len(notes)) - 1][:400]}")
        lines.append(f"final_notes: {notes[-1][:400]}")
    hyps = [a["hypothesis"] for a in ags if a.get("hypothesis")]
    label = "winning_hypotheses" if solved else "last_hypotheses"
    for h in hyps[-3:]:
        lines.append(f"{label}: {h[:220]}")
    text = "\n".join(lines)
    return TrajectoryDigest(t["trajectory_id"], text[:max_chars], solved)


COACH_SYSTEM_PROMPT = """You are the coach for an LLM agent (the student) that plays unknown 64x64 grid puzzle games.
Scoring per level is (human_actions / student_actions)^2; only environment actions cost score, analysis is free.
You never play. You improve the student's HARNESS from evidence in its past games:
  - playbook.md: an ordered list of numbered strategy rules the student reads every step.
  - analysis.py: analyze(), run in the student's Python REPL before every decision; its output is what the
    student sees about the frame (variables available: frames, curr, prev, steps, level, levels_completed,
    actions_this_level, budget_this_level, available_actions, baselines; numpy as np; helper tools).
  - tools/*.py: helper functions the student can call in the REPL (numpy only, no imports beyond np).

You will get digests of the student's most recent game on every training game, and the current harness files.
Propose a SMALL batch of edits (at most the given limit) that would most improve action efficiency across
games the student has NEVER seen. Rules:
  - Game-agnostic only. Never name or describe a specific game; never mention colours or shapes as if they
    were universal facts. Rules about *how to explore, verify, and act* transfer; facts about one game do not.
  - Every edit must cite the trajectory ids that motivated it. Copy each id in full, exactly as it appears
    after "###" in the evidence below (they look like "abcd-1234beef_student_20260101T000000Z"), not a prefix.
  - Prefer editing or removing a rule the student is ignoring over adding more rules. Keep rules short and testable.
  - Code you write must run: numpy only, no file or network access, return a string from analyze().
    Send source as a JSON array of lines (["def analyze():", "    return 'x'"]) so nothing depends on escaping.
    It must work on every state the student reaches, including before any action and when nothing changed.

Tags: procedural (how to explore/act), perceptual (what things look like), tooling (python helper),
meta (how the student reasons: hypothesis format, when to stop analysing, note-keeping).

Reply with exactly one JSON object:
{"analysis": "<what the evidence shows, 3-6 sentences>",
 "edits": [
   {"tag": "procedural|perceptual|tooling|meta", "op": "add_rule|replace_rule|remove_rule",
    "rule_number": <int, for replace/remove>, "text": "<rule text>", "evidence": ["<trajectory_id>", ...],
    "rationale": "<one sentence>"},
   {"tag": "tooling", "op": "write_tool", "name": "<snake_case>", "text": "<python source>", "evidence": [...], "rationale": "..."},
   {"tag": "meta|procedural", "op": "write_analysis", "text": "<full new analysis.py source defining analyze()>", "evidence": [...], "rationale": "..."}
 ]}
"""


def build_user_message(digests: list[TrajectoryDigest], harness_files: dict[str, str], max_edits: int) -> str:
    solved = sum(d.solved for d in digests)
    parts = [
        f"## Evidence: {len(digests)} training games, {solved} completed level 1, {len(digests) - solved} did not.\n",
        "\n\n".join(d.text for d in digests),
        "\n\n## Current harness files\n",
    ]
    for rel, text in harness_files.items():
        parts.append(f"--- {rel} ---\n{text}\n")
    parts.append(f"\nPropose at most {max_edits} edits as the JSON object described.")
    return "\n".join(parts)


def parse_proposal(text: str) -> tuple[str, list[Edit]]:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S)
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        raise ValueError("coach reply contained no JSON object")
    obj = json.loads(m.group(0))
    edits = [Edit.from_dict(d) for d in obj.get("edits", [])]
    for e in edits:
        if e.tag not in TAGS:
            e.tag = e.tag.lower()
    return str(obj.get("analysis", "")), edits


def propose(llm: LLMClient, digests: list[TrajectoryDigest], harness_files: dict[str, str],
            max_edits: int, validate=None, retries: int = 1) -> tuple[str, list[Edit], list[str]]:
    """Coach call, with up to `retries` repairs when the referee's validator rejects the batch.

    `validate(edits) -> list[str]` is passed in by the caller so this module never decides
    what is admissible; it only relays the complaints back to the model.
    Returns (analysis, edits, problems) — problems empty means the batch passed.
    """
    messages = [
        {"role": "system", "content": COACH_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(digests, harness_files, max_edits)},
    ]
    analysis, edits, problems = "", [], []
    for attempt in range(retries + 1):
        resp = llm.chat(messages, json_mode=True)
        if resp.truncated:
            # Out of room, usually from a long reasoning phase. Ask for a smaller batch rather than fail.
            if attempt == retries:
                raise ValueError("coach reply was cut off by the token limit")
            problems = ["the reply was cut off by the token limit"]
            messages += [{"role": "user", "content": "Your reply was cut off. Send fewer, shorter edits "
                          "(one is fine) and keep the analysis to three sentences."}]
            continue
        analysis, edits = parse_proposal(resp.text)
        problems = validate(edits) if validate else []
        if not problems or attempt == retries:
            return analysis, edits, problems
        messages += [
            {"role": "assistant", "content": resp.text},
            {"role": "user", "content": "The referee rejected this batch:\n- " + "\n- ".join(problems)
             + "\n\nSend the corrected batch as one JSON object. Fix only what was rejected; "
               "code must run on every state the student can reach, including before any action has been "
               "taken and when the last action changed nothing."},
        ]
    return analysis, edits, problems
