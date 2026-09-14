# A self-refining harness for ARC-AGI-3

Can an LLM agent's harness — its playbook, tools, and analysis routine — be refined automatically from
its own gameplay, such that action efficiency improves on games the refinement process has **never seen**?

This repository is the experiment, not a leaderboard entry. The deliverable is a trustworthy measurement.
A negative result is a valid outcome, and this one is negative so far.

## The short version

| | result |
|---|---|
| Hand-written v0 harness, held-out set | **RHAE 0.0176**, level 1 solved on 6 of 10 unseen games, 4 of them faster than the human median |
| Hand-written v0 harness, refine set | RHAE 0.0098, 4 of 15 |
| Random-action baseline | 0.0002 / 0.0000 |
| Coach rounds applied | 2 — both measured, both rolled back |
| Net change to the harness after 2 rounds | **none**; the files are byte-identical to the hand-written v0 |

The coach's written diagnoses were accurate. Its edits were plausible, evidence-cited, game-agnostic,
validated — and either harmful or unmeasurable. Round 1 halved the pilot score; round 2 tied. Chasing that
produced the most useful finding: **run-to-run variance on this benchmark is as large as the effects we were
trying to detect**, so single-run comparisons cannot support a keep-or-revert decision.

Write-ups: [`docs/m1-baseline.md`](docs/m1-baseline.md), [`docs/m2-round1.md`](docs/m2-round1.md),
[`docs/m2-round2.md`](docs/m2-round2.md), [`docs/m2-round3.md`](docs/m2-round3.md),
[`docs/toolkit-notes.md`](docs/toolkit-notes.md). Design decisions live in [`SPEC.md`](SPEC.md).

## Design in one paragraph

Three roles. The **student** plays one game with a frozen harness, using a persistent Python REPL over the
frame history so it writes numpy instead of reading 4,096 integers as text. The **coach** reads trajectories
from the refine set *only* — enforced in code, with a runtime test that spies on every file it opens — and
proposes at most three tagged, evidence-cited edits per round. The **referee** evaluates both sets with a
fixed model, budget, and prompt, and reverts a batch if the refine score drops. The student's base system
prompt is immutable; only files under `harness/` change.

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
echo "NEURALWATT_API_KEY=..." > .env          # or ANTHROPIC_API_KEY, see config/models.py
.venv/bin/python -m pytest -q                 # 40 tests, no API calls

.venv/bin/python scripts/make_split.py --seed 0
.venv/bin/python scripts/random_agent.py --game ls20            # free sanity check
.venv/bin/python scripts/eval.py --set heldout --agent student --effort high \
    --max-levels 1 --max-actions-per-level 60 --repeats 3 --jobs 5
.venv/bin/python scripts/refine.py --effort high --dry-run      # one coach round, proposal only
.venv/bin/python scripts/variance.py --harness bc653ad4e040 --cap 60
```

Every run prints a projected cost first and refuses to start above the cap in `config/models.py`.
Reasoning must be requested explicitly (`--effort high`); with it off, the student never finished a level.

## Layout

```
config/     split, per-level action budget, model specs and cost caps
harness/    the learned state: playbook.md, analysis.py, tools/, history/ (snapshots + edit log)
src/arc_harness/
  env.py          arc_agi wrapper: numpy frames, action-budget enforcement, level billing
  repl.py         persistent namespace exposed to the student
  student.py      inner loop + the immutable base prompt
  coach.py        outer loop; reads trajectories/refine/ only
  harness_state.py  validation, snapshots, apply, rollback, edit log
  scoring.py      RHAE, asserted equal to the toolkit's own calculator
scripts/    eval.py, refine.py, variance.py, make_split.py, random_agent.py
results/    one JSON row per eval run: seeds, harness fingerprint, code version, tokens, dollars
trajectories/refine|heldout/   per-episode logs; the coach's only input
```

## Caveats

Level 1 only, under a 60-action ceiling, so the later levels that dominate the official weighting are
untested. Single-run figures carry the uncertainty quantified in `docs/m2-round3.md`. Two coach rounds is a
small sample of coach behaviour.
