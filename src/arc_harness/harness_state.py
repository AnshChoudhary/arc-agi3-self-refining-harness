"""Referee-side management of the learned harness state: snapshots, validation, apply, rollback, log.

The coach only *proposes* edits (coach.py). Everything that touches disk under harness/
goes through here so the invariants hold regardless of what the coach outputs:
  - an edit naming a game id is rejected loudly (CLAUDE.md rule 4);
  - tool / analysis code must compile and run before it is written;
  - every applied edit is logged to harness/history/edits.jsonl with tag, diff, evidence;
  - a snapshot of harness/ is taken before each round so a batch can be rolled back.
The base system prompt lives in student.py and is never touched here (rule 2).
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arc_harness.env import ENVIRONMENTS_DIR
from arc_harness.repl import HARNESS_DIR, harness_fingerprint, load_harness_tools

HISTORY_DIR = HARNESS_DIR / "history"
SNAPSHOT_DIR = HISTORY_DIR / "snapshots"
EDIT_LOG = HISTORY_DIR / "edits.jsonl"
PLAYBOOK = HARNESS_DIR / "playbook.md"
ANALYSIS = HARNESS_DIR / "analysis.py"
TOOLS_DIR = HARNESS_DIR / "tools"

TAGS = ("procedural", "perceptual", "tooling", "meta")
OPS = ("add_rule", "replace_rule", "remove_rule", "write_tool", "write_analysis")
RULE_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def _as_source(value) -> str:
    """Code may arrive as a list of lines (preferred, unambiguous) or as one string."""
    if isinstance(value, list):
        return "\n".join(str(line) for line in value)
    return str(value)


def _source_candidates(text: str) -> list[str]:
    """Readings of a code string, best first.

    A model writing Python inside JSON sometimes double-escapes, so line breaks arrive as
    literal backslash-n. That is ambiguous: `"\\n".join(lines)` legitimately contains the same
    sequence. Rather than guess, offer each reading and let the compiler choose.
    """
    out = [text]
    if "\n" not in text and "\\n" in text:
        flat = text.replace("\\n", "\n").replace("\\t", "\t")
        out.append(flat)
        # ... and the same, with a newline between two quotes read back as an escape, which is
        # how `"\\n".join(...)` looks once every escape has been expanded.
        out.append(re.sub(r'(["\'])\n\1', lambda m: f"{m.group(1)}\\n{m.group(1)}", flat))
    return out


def choose_source(text: str) -> str:
    """The first reading that compiles; the original if none do, so errors point at what was sent."""
    for candidate in _source_candidates(text):
        try:
            compile(candidate, "<candidate>", "exec")
            return candidate
        except SyntaxError:
            continue
    return text


@dataclass
class Edit:
    tag: str
    op: str
    evidence: list[str]  # trajectory ids that motivated the edit
    rationale: str
    text: str = ""  # rule text (add/replace), tool/analysis source (write_*)
    rule_number: int | None = None  # replace_rule / remove_rule
    name: str = ""  # write_tool: module name without .py

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        return cls(
            tag=str(d.get("tag", "")),
            op=str(d.get("op", "")),
            evidence=[str(e) for e in d.get("evidence", [])],
            rationale=str(d.get("rationale", "")),
            text=_as_source(d.get("text", d.get("content", ""))),
            rule_number=int(d["rule_number"]) if d.get("rule_number") is not None else None,
            name=str(d.get("name", "")),
        )


# ---- reading -------------------------------------------------------------------

def read_files() -> dict[str, str]:
    """Current harness files as {relative path: text}; history/ excluded."""
    out: dict[str, str] = {}
    for path in sorted(HARNESS_DIR.rglob("*")):
        rel = path.relative_to(HARNESS_DIR)
        if path.is_dir() or rel.parts[0] == "history" or path.name == ".gitkeep":
            continue
        out[str(rel)] = path.read_text()
    return out


def parse_rules(playbook_text: str) -> tuple[str, list[str]]:
    """Split the playbook into its preamble and the numbered rule texts."""
    preamble: list[str] = []
    rules: list[str] = []
    for line in playbook_text.splitlines():
        m = RULE_RE.match(line.strip())
        if m:
            rules.append(m.group(2).strip())
        elif not rules:
            preamble.append(line)
    return "\n".join(preamble).rstrip() + "\n", rules


def render_rules(preamble: str, rules: list[str]) -> str:
    return preamble + "\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules)) + "\n"


# ---- validation ---------------------------------------------------------------

def known_game_ids() -> set[str]:
    """Every game id the toolkit has on disk (base and versioned). Not the split: no held-out list is read."""
    ids: set[str] = set()
    if ENVIRONMENTS_DIR.exists():
        for base in ENVIRONMENTS_DIR.iterdir():
            if base.is_dir():
                ids.add(base.name)
                for v in base.iterdir():
                    if v.is_dir():
                        ids.add(f"{base.name}-{v.name}")
    return ids


def mentions_game_id(text: str, ids: set[str]) -> str | None:
    tokens = set(re.findall(r"[a-z0-9]{4}(?:-[0-9a-f]{8})?", text.lower()))
    hit = tokens & ids
    return sorted(hit)[0] if hit else None


def _step(index: int, action: str, changed: int) -> dict:
    """A step log entry shaped exactly like env.Step, which is what the REPL exposes."""
    return {"index": index, "level": 0, "action": action, "data": {}, "state": "NOT_FINISHED",
            "levels_completed": 0, "actions_this_level": index, "budget_this_level": 30, "n_frames": 1,
            "frame_hash": "0" * 16, "changed_pixels": changed, "level_changed": False, "full_reset": False}


def _analysis_scenarios() -> list[tuple[str, dict]]:
    """States analyze() must survive. The student hits all of these in a normal episode."""
    blank = np.zeros((64, 64), dtype=np.int8)
    first = blank.copy()
    first[10:20, 10:20] = 3
    moved = blank.copy()
    moved[12:22, 10:20] = 3
    common = {"level": 0, "levels_completed": 0, "budget_this_level": 30,
              "available_actions": ["ACTION1", "ACTION6"], "baselines": [6, 9]}
    return [
        ("opening frame, nothing done yet",
         {"frames": [first], "curr": first, "prev": first, "steps": [], "actions_this_level": 0, **common}),
        ("an action that changed the frame",
         {"frames": [first, moved], "curr": moved, "prev": first, "steps": [_step(1, "ACTION1", 40)],
          "actions_this_level": 1, **common}),
        ("a no-op after earlier actions",
         {"frames": [first, moved, moved], "curr": moved, "prev": moved,
          "steps": [_step(1, "ACTION1", 40), _step(2, "ACTION6", 0)], "actions_this_level": 2, **common}),
        ("a uniform frame with no components",
         {"frames": [blank, blank], "curr": blank, "prev": blank, "steps": [_step(1, "ACTION1", 0)],
          "actions_this_level": 1, **common}),
    ]


def _check_code(source: str, kind: str) -> str | None:
    """Compile and exec tool/analysis code the way the REPL will; return an error string or None."""
    try:
        compile(source, f"<{kind}>", "exec")
    except SyntaxError as e:
        return f"{kind} does not compile: {e}"
    ns: dict = {"np": np, "__name__": f"harness.check.{kind}"}
    try:
        load_harness_tools(ns)  # analysis may call existing tools
        exec(source, ns)
    except Exception as e:  # noqa: BLE001 — any failure means the edit is unsafe to install
        return f"{kind} failed to run: {type(e).__name__}: {e}"
    if kind != "analysis":
        return None
    if "analyze" not in ns or not callable(ns["analyze"]):
        return "analysis must define analyze()"
    for label, state in _analysis_scenarios():
        ns.update(state)
        try:
            out = ns["analyze"]()
        except Exception as e:  # noqa: BLE001
            return f"analyze() raised {type(e).__name__}: {e} — on {label}"
        if not isinstance(out, str):
            return f"analyze() returned {type(out).__name__}, not a string — on {label}"
    return None


def resolve_evidence(cited: str, known: set[str]) -> str | None:
    """Map a citation to a known trajectory id: exact, else a unique prefix (a game id names one run)."""
    if cited in known:
        return cited
    matches = sorted(k for k in known if k.startswith(cited))
    return matches[0] if len(matches) == 1 else None


def validate(edits: list[Edit], known_trajectories: set[str], max_edits: int) -> list[str]:
    """Return a list of human-readable problems; empty means the batch may be applied.

    Evidence citations are normalised in place to full trajectory ids so the log records
    exactly which run motivated each edit."""
    problems: list[str] = []
    if not edits:
        problems.append("no edits proposed")
    if len(edits) > max_edits:
        problems.append(f"{len(edits)} edits exceeds the per-round limit of {max_edits}")
    ids = known_game_ids()
    _, rules = parse_rules(PLAYBOOK.read_text()) if PLAYBOOK.exists() else ("", [])
    for i, e in enumerate(edits):
        where = f"edit {i + 1} ({e.op})"
        if e.tag not in TAGS:
            problems.append(f"{where}: tag {e.tag!r} not in {TAGS}")
        if e.op not in OPS:
            problems.append(f"{where}: op {e.op!r} not in {OPS}")
            continue
        if not e.evidence:
            problems.append(f"{where}: no trajectory evidence cited")
        resolved, unknown = [], []
        for cited in e.evidence:
            hit = resolve_evidence(cited, known_trajectories)
            (resolved if hit else unknown).append(hit or cited)
        if unknown:
            problems.append(f"{where}: evidence not found in refine trajectories: {unknown}")
        else:
            e.evidence = resolved
        for label, blob in (("text", e.text), ("rationale", e.rationale), ("name", e.name)):
            hit = mentions_game_id(blob, ids)
            if hit:
                problems.append(f"{where}: {label} names game {hit!r}; harness edits must be game-agnostic")
        if e.op in ("add_rule", "replace_rule") and not e.text.strip():
            problems.append(f"{where}: empty rule text")
        if e.op in ("replace_rule", "remove_rule"):
            if e.rule_number is None or not (1 <= e.rule_number <= len(rules)):
                problems.append(f"{where}: rule_number {e.rule_number} out of range 1..{len(rules)}")
        if e.op in ("write_tool", "write_analysis"):
            e.text = choose_source(e.text)
        if e.op == "write_tool":
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", e.name or ""):
                problems.append(f"{where}: tool name {e.name!r} must be a short snake_case identifier")
            err = _check_code(e.text, "tool")
            if err:
                problems.append(f"{where}: {err}")
        if e.op == "write_analysis":
            err = _check_code(e.text, "analysis")
            if err:
                problems.append(f"{where}: {err}")
    return problems


# ---- snapshots ---------------------------------------------------------------

def snapshot(round_id: str) -> Path:
    dest = SNAPSHOT_DIR / round_id
    dest.mkdir(parents=True, exist_ok=True)
    for rel, text in read_files().items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text)
    (dest / "FINGERPRINT").write_text(harness_fingerprint())
    return dest


def restore(snap: Path) -> None:
    """Replace the live harness with a snapshot (used for rollback)."""
    for rel in read_files():
        (HARNESS_DIR / rel).unlink()
    for path in sorted(snap.rglob("*")):
        if path.is_file() and path.name != "FINGERPRINT":
            target = HARNESS_DIR / path.relative_to(snap)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text())


# ---- apply + log --------------------------------------------------------------

def _diff(rel: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), f"a/{rel}", f"b/{rel}"))


def apply(edits: list[Edit], round_id: str, coach_model: str, snapshot_path: Path) -> list[dict]:
    """Write the edits to harness/ and append one log entry per edit. Validate first."""
    entries: list[dict] = []
    for e in edits:
        if e.op in ("add_rule", "replace_rule", "remove_rule"):
            rel = "playbook.md"
            before = PLAYBOOK.read_text() if PLAYBOOK.exists() else "# Playbook\n"
            preamble, rules = parse_rules(before)
            if e.op == "add_rule":
                rules.append(e.text.strip())
            elif e.op == "replace_rule":
                rules[e.rule_number - 1] = e.text.strip()
            else:
                rules.pop(e.rule_number - 1)
            after = render_rules(preamble, rules)
            PLAYBOOK.write_text(after)
        elif e.op == "write_tool":
            rel = f"tools/{e.name}.py"
            path = TOOLS_DIR / f"{e.name}.py"
            before = path.read_text() if path.exists() else ""
            after = e.text.rstrip() + "\n"
            path.write_text(after)
        else:  # write_analysis
            rel = "analysis.py"
            before = ANALYSIS.read_text() if ANALYSIS.exists() else ""
            after = e.text.rstrip() + "\n"
            ANALYSIS.write_text(after)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round_id": round_id,
            "snapshot_before": str(snapshot_path.relative_to(HARNESS_DIR)),
            "coach_model": coach_model,
            "tag": e.tag,
            "op": e.op,
            "file": rel,
            "diff": _diff(rel, before, after),
            "evidence": e.evidence,
            "rationale": e.rationale,
            "fingerprint_after": harness_fingerprint(),
        }
        entries.append(entry)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with EDIT_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    return entries


def log_rollback(round_id: str, snapshot_path: Path, reason: str) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with EDIT_LOG.open("a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round_id": round_id,
            "op": "rollback",
            "restored_snapshot": str(snapshot_path.relative_to(HARNESS_DIR)),
            "reason": reason,
            "fingerprint_after": harness_fingerprint(),
        }) + "\n")


def edit_history() -> list[dict]:
    if not EDIT_LOG.exists():
        return []
    return [json.loads(line) for line in EDIT_LOG.read_text().splitlines() if line.strip()]
