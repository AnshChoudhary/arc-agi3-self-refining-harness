"""Student loop with a fake LLM: reply parsing, REPL round trips, acting, notes, logging."""

import json

from arc_harness.env import ArcEnv
from arc_harness.llm import LLMResponse, Usage
from arc_harness.student import BASE_SYSTEM_PROMPT, StudentAgent, parse_reply


class FakeLLM:
    """Returns scripted replies in order; records the messages it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.usage = Usage()

    def chat(self, messages, json_mode=True, temperature=None):
        self.seen.append([dict(m) for m in messages])  # snapshot; the loop mutates its list
        text = self.replies.pop(0)
        return LLMResponse(text, "stop", Usage(input_tokens=10, output_tokens=5, calls=1), False, 0.0)


def test_parse_reply_variants():
    assert parse_reply('{"hypothesis": "h", "python": "print(1)"}').python == "print(1)"
    p = parse_reply('```json\n{"hypothesis": "h", "action": "action6", "x": "3", "y": 4, "notes": "n"}\n```')
    assert (p.action, p.x, p.y, p.notes) == ("ACTION6", 3, 4, "n")
    assert parse_reply("sure! {\"action\": \"ACTION1\"} done").action == "ACTION1"
    assert parse_reply("no json here").error
    assert parse_reply('{"hypothesis": "only"}').error
    assert parse_reply('{"python": "x", "action": "ACTION1"}').error
    assert parse_reply('{"action": "ACTION6", "x": "a"}').error


def test_loop_runs_code_then_acts_and_logs():
    replies = [
        json.dumps({"hypothesis": "look", "python": "n_before = len(frames); print(curr.shape, len(components(curr)))"}),
        json.dumps({"hypothesis": "probe", "action": "ACTION1", "notes": "A1 probed"}),
        json.dumps({"hypothesis": "bad", "action": "ACTION6", "x": 1, "y": 1}),  # not available -> invalid
        json.dumps({"hypothesis": "probe2", "python": "print(len(frames) - n_before, steps[-1]['action'])"}),
        json.dumps({"hypothesis": "act2", "action": "ACTION2", "notes": "A1, A2 probed"}),
    ]
    # A two-action budget ends the loop right after the scripted replies.
    env = ArcEnv("ls20", seed=0, offline=True, budget_multiplier=1)
    env.baselines = [2] + env.baselines[1:]  # level-1 budget = 2 actions
    llm = FakeLLM(replies)
    steps, used = StudentAgent(llm, log=lambda *_: None).play(env, max_levels=None)

    assert env.done and env.outcome == "budget_exhausted"
    assert [s.action for s in env.steps] == ["ACTION1", "ACTION2"]
    assert len(steps) == 2
    assert steps[0].analysis_code and "(64, 64)" in steps[0].repl_outputs[0]
    assert steps[0].notes == "A1 probed" and steps[0].llm_calls == 2
    assert steps[1].invalid_attempts and "ACTION6" in steps[1].invalid_attempts[0]
    assert "1 ACTION1" in steps[1].repl_outputs[0]  # REPL variables persist across steps
    assert steps[1].notes == "A1, A2 probed" and steps[1].llm_calls == 3
    assert used.calls == 5 and used.input_tokens == 50

    first = llm.seen[0]
    assert first[0]["content"] == BASE_SYSTEM_PROMPT and first[1]["content"].startswith("HARNESS")
    assert "Scene summary" in first[-1]["content"] and "components" in first[1]["content"]
    assert "Your notes:\nA1 probed" in llm.seen[2][-1]["content"]  # first call of step 2


def test_gives_up_after_repeated_bad_replies():
    llm = FakeLLM(["nonsense"] * 20)
    env = ArcEnv("ls20", seed=0, offline=True)
    steps, used = StudentAgent(llm, log=lambda *_: None).play(env, max_levels=None)
    assert not env.done and env.steps == []
    assert used.calls == 6  # 3 bad, fresh-context escalation, 3 more bad, give up
    assert any(e.startswith("escalated") for e in steps[0].invalid_attempts)
    assert "ACTION JSON object only" in llm.seen[3][-1]["content"] and len(llm.seen[3]) == 3


def test_escalation_recovers_when_model_then_acts():
    llm = FakeLLM(["nonsense"] * 3 + [json.dumps({"hypothesis": "ok", "action": "ACTION1", "notes": "n"})] + ["x"] * 20)
    env = ArcEnv("ls20", seed=0, offline=True, budget_multiplier=1)
    env.baselines = [1] + env.baselines[1:]
    steps, used = StudentAgent(llm, log=lambda *_: None).play(env, max_levels=None)
    assert [s.action for s in env.steps] == ["ACTION1"] and used.calls == 4


def test_analysis_cap_is_enforced_and_loop_abandons():
    from config.models import MAX_CALLS_PER_ACTION

    llm = FakeLLM([json.dumps({"hypothesis": "h", "python": f"print({i})"}) for i in range(40)])
    env = ArcEnv("ls20", seed=0, offline=True)
    steps, used = StudentAgent(llm, log=lambda *_: None).play(env, max_levels=None)
    assert env.steps == []  # never acted
    assert used.calls <= MAX_CALLS_PER_ACTION
    assert any("analysis limit" in e for e in steps[0].invalid_attempts)
    assert len(steps[0].analysis_code) == 6  # only the allowed calls ran, even after escalation


def test_identical_replies_are_rejected():
    same = json.dumps({"hypothesis": "h", "python": "print(1)"})
    llm = FakeLLM([same] * 10)
    env = ArcEnv("ls20", seed=0, offline=True)
    steps, used = StudentAgent(llm, log=lambda *_: None).play(env, max_levels=None)
    assert used.calls == 7  # runs once, 3 identical, escalation, 3 identical, give up
    assert len(steps[0].analysis_code) == 1
