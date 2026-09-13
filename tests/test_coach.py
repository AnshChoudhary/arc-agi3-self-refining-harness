"""Coach: held-out isolation (static + runtime), digests, proposal parsing, validation, apply/rollback."""

import builtins
import json
import re
from pathlib import Path

import pytest

from arc_harness import coach, harness_state as hs
from arc_harness.llm import LLMResponse, Usage

SRC = Path(__file__).resolve().parents[1] / "src" / "arc_harness"


def test_coach_source_never_mentions_heldout_or_split():
    for name in ("coach.py",):
        text = (SRC / name).read_text()
        body = re.sub(r'"""[\s\S]*?"""', "", text)  # docstrings may explain the rule
        assert "heldout" not in body and "held_out" not in body
        assert "split.json" not in body and "load_split" not in body


def test_loader_refuses_paths_outside_refine(tmp_path):
    with pytest.raises(PermissionError):
        coach._assert_refine(tmp_path / "x.json")


def test_runtime_never_opens_heldout(monkeypatch):
    opened = []
    real_open = builtins.open
    real_path_open = Path.open

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_path_open(self, *a, **k):
        opened.append(str(self))
        return real_path_open(self, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "open", spy_path_open)
    trajectories = coach.load_refine_trajectories(agent="student")
    assert trajectories, "expected M1 refine trajectories on disk"
    assert not [p for p in opened if "heldout" in p]
    assert all("/trajectories/refine/" in p for p in opened if p.endswith(".json"))


def test_digest_is_compact_and_informative():
    t = coach.load_refine_trajectories(agent="student")[0]
    d = coach.digest(t)
    assert d.trajectory_id == t["trajectory_id"]
    assert len(d.text) <= 2600
    for key in ("outcome=", "noop_rate=", "action_log:", "llm_calls_per_action="):
        assert key in d.text


def test_parse_proposal_and_validation_rejects_game_ids():
    reply = json.dumps({"analysis": "x", "edits": [
        {"tag": "procedural", "op": "add_rule", "text": "In ls20 always go up.", "evidence": ["t1"], "rationale": "r"},
        {"tag": "tooling", "op": "write_tool", "name": "bad tool", "text": "def f(:\n  pass", "evidence": [], "rationale": "r"},
        {"tag": "meta", "op": "replace_rule", "rule_number": 99, "text": "ok", "evidence": ["t1"], "rationale": "r"},
    ]})
    analysis, edits = coach.parse_proposal(reply)
    assert analysis == "x" and len(edits) == 3
    problems = hs.validate(edits, known_trajectories={"t1"}, max_edits=3)
    joined = "\n".join(problems)
    assert "names game 'ls20'" in joined
    assert "snake_case" in joined and "does not compile" in joined and "no trajectory evidence" in joined
    assert "out of range" in joined


def test_validation_accepts_good_batch_and_checks_code_runs():
    edits = [
        hs.Edit("procedural", "add_rule", ["t1"], "r", text="Before acting, verify the last action's effect with diff()."),
        hs.Edit("tooling", "write_tool", ["t1"], "r", name="hist",
                text="def hist(frame):\n    \"\"\"colour histogram\"\"\"\n    v, c = np.unique(frame, return_counts=True)\n    return dict(zip(v.tolist(), c.tolist()))\n"),
        hs.Edit("meta", "write_analysis", ["t1"], "r",
                text="def analyze():\n    return f'level {level}, {len(frames)} frames, colours {sorted(set(curr.flatten().tolist()))}'\n"),
    ]
    assert hs.validate(edits, {"t1"}, 3) == []
    broken = [hs.Edit("meta", "write_analysis", ["t1"], "r", text="def analyze():\n    return undefined_name\n")]
    assert any("failed to run" in p for p in hs.validate(broken, {"t1"}, 3))


def test_snapshot_apply_log_rollback_roundtrip(monkeypatch, tmp_path):
    # Work on a copy of the harness so the real one is untouched.
    live = tmp_path / "harness"
    live.mkdir()
    (live / "tools").mkdir()
    (live / "history").mkdir()
    (live / "playbook.md").write_text("# Playbook\n\n1. First rule.\n2. Second rule.\n")
    (live / "analysis.py").write_text("def analyze():\n    return 'v0'\n")
    (live / "tools" / "scene.py").write_text("def one():\n    return 1\n")
    for name, value in {
        "HARNESS_DIR": live, "HISTORY_DIR": live / "history", "SNAPSHOT_DIR": live / "history" / "snapshots",
        "EDIT_LOG": live / "history" / "edits.jsonl", "PLAYBOOK": live / "playbook.md",
        "ANALYSIS": live / "analysis.py", "TOOLS_DIR": live / "tools",
    }.items():
        monkeypatch.setattr(hs, name, value)
    monkeypatch.setattr("arc_harness.repl.HARNESS_DIR", live)
    monkeypatch.setattr(hs, "harness_fingerprint", lambda: "fp-" + str(sorted(hs.read_files().items())).__hash__().__abs__().__str__()[:6])

    before = hs.read_files()
    snap = hs.snapshot("r1")
    edits = [
        hs.Edit("procedural", "replace_rule", ["t1"], "why", text="Second rule, sharper.", rule_number=2),
        hs.Edit("procedural", "remove_rule", ["t1"], "why", rule_number=1),
        hs.Edit("tooling", "write_tool", ["t1"], "why", name="extra", text="def two():\n    return 2\n"),
    ]
    entries = hs.apply(edits, "r1", "fake-model", snap)
    assert (live / "playbook.md").read_text() == "# Playbook\n\n1. Second rule, sharper.\n"
    assert (live / "tools" / "extra.py").exists()
    assert len(entries) == 3 and all(e["diff"] and e["tag"] and e["evidence"] == ["t1"] for e in entries)
    assert len(hs.edit_history()) == 3

    hs.restore(snap)
    hs.log_rollback("r1", snap, "regressed")
    assert hs.read_files() == before
    assert hs.edit_history()[-1]["op"] == "rollback"


class FakeCoachLLM:
    def __init__(self, text):
        self.text, self.usage = text, Usage()

    def chat(self, messages, json_mode=True, temperature=None):
        self.messages = messages
        return LLMResponse(self.text, "stop", Usage(calls=1), False, 0.0)


def test_propose_builds_prompt_from_digests_and_harness():
    trajectories = coach.load_refine_trajectories(agent="student")[:3]
    digests = [coach.digest(t) for t in trajectories]
    reply = json.dumps({"analysis": "a", "edits": [{"tag": "meta", "op": "add_rule", "text": "Act after two analysis calls.",
                                                    "evidence": [digests[0].trajectory_id], "rationale": "r"}]})
    llm = FakeCoachLLM(reply)
    analysis, edits, raw = coach.propose(llm, digests, hs.read_files(), max_edits=3)
    assert analysis == "a" and edits[0].evidence == [digests[0].trajectory_id]
    user = llm.messages[1]["content"]
    assert "--- playbook.md ---" in user and digests[0].trajectory_id in user and "at most 3 edits" in user
    assert hs.validate(edits, {d.trajectory_id for d in digests}, 3) == []
