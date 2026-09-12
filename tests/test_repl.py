"""REPL: persistence, data views, output capture, error reporting, timeout, tool loading."""

import numpy as np

from arc_harness.env import ArcEnv
from arc_harness.repl import Repl, harness_fingerprint, load_harness_tools

GAME = "ls20"


def test_data_views_and_persistence():
    env = ArcEnv(GAME, seed=0)
    repl = Repl(env, load_tools=False)
    r = repl.run("n0 = len(frames); print(curr.shape, curr.dtype, level, budget_this_level, available_actions)")
    assert r.ok and r.stdout.strip() == "(64, 64) int8 0 110 ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']"

    env.step("ACTION1")
    repl.refresh(env)
    r = repl.run("print(len(frames) - n0, int((curr != prev).sum()), steps[-1]['action'], actions_this_level)")
    assert r.ok and r.stdout.split() == ["1", "52", "ACTION1", "1"]


def test_errors_are_reported_not_raised():
    env = ArcEnv(GAME, seed=0)
    repl = Repl(env, load_tools=False)
    r = repl.run("print('before'); 1 / 0")
    assert not r.ok and r.stdout.strip() == "before" and "ZeroDivisionError" in r.error
    assert "<repl>" in r.error and "repl.py" not in r.error


def test_output_truncation_and_timeout():
    env = ArcEnv(GAME, seed=0)
    repl = Repl(env, timeout_s=0.3, load_tools=False)
    r = repl.run("print('x' * 10000)")
    assert r.ok and r.truncated and len(r.stdout) == 4000
    r = repl.run("while True: pass")
    assert not r.ok and "TimeoutError" in r.error and r.elapsed_s < 2


def test_harness_tools_load_from_dir(tmp_path):
    (tmp_path / "geom.py").write_text("def changed(a, b):\n    return np.argwhere(a != b)\n\n_private = 1\n")
    ns = {"np": np}
    names = load_harness_tools(ns, tmp_path)
    assert names == ["changed"] and "_private" not in ns
    assert ns["changed"](np.zeros(3), np.array([0, 1, 0])).tolist() == [[1]]


def test_fingerprint_changes_with_content(tmp_path):
    (tmp_path / "playbook.md").write_text("rule 1")
    a = harness_fingerprint(tmp_path)
    (tmp_path / "playbook.md").write_text("rule 2")
    b = harness_fingerprint(tmp_path)
    (tmp_path / "history").mkdir()
    (tmp_path / "history" / "snap.json").write_text("{}")
    assert a != b and harness_fingerprint(tmp_path) == b  # history/ is excluded
