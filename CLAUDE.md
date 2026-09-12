# CLAUDE.md — working rules for this repo

Read `SPEC.md` first. It is the source of truth for design decisions. Do not re-decide things it settles; if you think a decision is wrong, say so and stop rather than silently diverging.

## What this project is
A research-style experiment: can an LLM agent's harness be self-refined from gameplay so that action efficiency improves on ARC-AGI-3 games the refiner has never seen. The output is a trustworthy experiment, not a high score.

## Language & tooling
- Python 3.11+. `uv` or plain venv. No TypeScript.
- Dependencies: `arc-agi` (toolkit), `numpy`, `anthropic` (or whichever LLM client is configured). Keep the list short.
- Follow the SPEC repo layout. Don't create parallel structures.

## Hard rules — never violate
1. **Held-out isolation.** The coach (`coach.py`, `scripts/refine.py`) reads only `trajectories/refine/`. It must not import, glob, or open anything under `trajectories/heldout/` or read `config/split.json`'s held-out list. Enforce in code.
2. **Immutable base prompt.** The student's base system prompt lives in code and is never edited by the coach. Only files under `harness/` change.
3. **Every harness edit is logged** to `harness/history/` with: timestamp, tag (procedural / perceptual / tooling / meta), the diff, and the trajectory IDs cited as evidence. A snapshot is taken before each round so it can be rolled back.
4. **Reject edits that name a game ID.** Automated check; fail loudly.
5. **Action budget is enforced in `env.py`**, not left to the LLM: 5× human median per level. Exceeding it ends the level as failed.
6. **Cheap model by default.** `config/models.py` defaults to the cheapest available model. Switching to a stronger model is an explicit CLI flag, never a default.
7. **Every eval run records cost** (input/output tokens, $ estimate) and writes it to `results/` alongside RHAE.

## Milestone order (from SPEC §7)
M0 plumbing → M1 nonzero baseline → M2 coach → M3 analysis. Do not build the coach before M1 has a nonzero held-out RHAE. If asked to skip ahead, push back.

## How to work
- Before writing against the `arc-agi` toolkit, **read its installed source** (`python -c "import arc_agi, inspect; print(inspect.getsourcefile(arc_agi))"`) to confirm attribute names on `FrameDataRaw`, `GameAction`, `EnvironmentWrapper`. Don't rely on memory of the docs.
- Prefer numpy operations over LLM reasoning for anything mechanical (frame diffs, component labelling, counting). The LLM should form hypotheses and pick actions, not do arithmetic.
- Every experiment is reproducible: seed, model, harness snapshot ID, split file, budget — all recorded in the results row.
- Trajectory logs are the coach's only input. Log generously: frame hashes, actions, level transitions, the student's stated hypothesis per step, analysis code it ran, tokens used.
- Small commits. One milestone sub-task per PR/commit.

## Cost discipline
- Sanity-check any new loop on **one game, one level** before running the full set.
- Print projected cost before launching a full eval; abort if > configured cap in `config/models.py`.
- Cache LLM responses keyed on (prompt hash, model) during development to avoid paying twice for identical steps.

## Things to push back on
- "Just evaluate on all games for now" → no; split isolation is the point.
- "Let the coach also edit the system prompt" → no; SPEC §3.
- "Let's use the strongest model to get the baseline up" → not until plumbing is verified on the cheap one.
- Any "self-improvement" feature whose effect can't be measured on held-out games.

## Style
- Type hints, dataclasses for schemas, no clever metaprogramming.
- Short functions. A reader should be able to follow the student loop top to bottom in one screen.
- Comments explain *why* (scoring rule, isolation constraint), not *what*.
