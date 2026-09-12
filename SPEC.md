# ARC-AGI-3 Self-Improving Harness — Project Spec

## 1. Problem statement

Can an LLM agent's harness — its supplemental prompts, tool library, and exploration playbook — be **automatically refined from its own gameplay trajectories** such that action efficiency improves on ARC-AGI-3 games the refinement process has **never seen**?

This is a fun/learning project, not a leaderboard or Kaggle entry. The deliverable is a trustworthy experiment and a writeup, not a high score. A negative result ("every gain vanished on held-out games") is a valid, interesting outcome.

### Why this question
ARC Prize's technical report showed hand-built harnesses are extremely bimodal: the same model jumps from 0% to ~97% on one game with a harness, and stays at 0% on another. The official leaderboard excludes harnesses for that reason. Nobody has shown whether *learned* harness lessons generalize across games or just overfit the games they were learned on.

## 2. Benchmark facts that drive the design

- Games are interactive 2D grid environments (64×64 int grids, small palette). No instructions. The agent must discover mechanics by acting.
- Actions: `ACTION1`–`ACTION5` (simple, meaning varies per game), `ACTION6` (complex, takes x,y), `ACTION7` (undo, where supported). `env.action_space` tells you which are available.
- **Scoring is RHAE (Relative Human Action Efficiency):**
  - `level_score = (human_baseline_actions / ai_actions) ** 2`, capped at 1.15
  - game score = weighted average of level scores, weight = 1-indexed level number
  - total = mean of game scores
  - Only environment actions count. Reasoning, tool calls, retries, wall-clock are all free.
- ARC imposes an action budget of **5× human median per level** in their evals. We do the same.

Design consequences:
1. **Reason hard, act rarely.** Spend arbitrary compute between actions; never waste an action.
2. **Quadratic penalty.** 2× human actions → 25% on that level. 3× → ~11%. Random exploration is worthless.
3. **Late levels dominate.** Finishing the last level matters more than being efficient on level 1.

## 3. Architecture: two loops at two speeds

### Inner loop — the Student (plays one game; harness is frozen)
- Wraps `arc_agi` local environment.
- Maintains full frame history as numpy arrays in a **persistent Python REPL** (IPython kernel or a simple `exec` namespace). The LLM writes code against the history (`np.argwhere(prev != curr)`, connected components, color counts, etc.) instead of reading 4,096 ints as text.
- Loop per step: observe → (optionally run analysis code, any number of times) → propose hypothesis about mechanics → choose the single action that best disambiguates or progresses → act → log.
- Reads harness state at episode start. **Never writes to it.**

### Outer loop — the Coach (refines the harness between batches)
- Input: trajectories from the **refine set only** (see §4).
- Reads logs of wasted actions, level completions, action-per-level vs budget, and the student's stated hypotheses.
- Proposes **small, evidence-backed edits** to harness state: add/edit/remove a playbook rule, add a tool function, adjust the base analysis routine.
- Every edit: tagged (§5), logged with the trajectory evidence that motivated it, and snapshotted so it can be rolled back.
- Rollback rule: if refine-set RHAE regresses after an edit batch, revert that batch.

### Harness state (the thing being learned)
```
harness/
  playbook.md        # ordered list of strategy rules the student reads
  tools/             # python functions the student can call from the REPL
  analysis.py        # default per-step analysis routine
  history/           # snapshots + edit log (json)
```
The **base system prompt is immutable**. Only the files above change.

### Referee — evaluation (never touched by the Coach)
- Fixed model, fixed base prompt, fixed action budget.
- Runs the student on refine set and held-out set, reports RHAE per game/level, actions per level, and $ cost per run.
- Runs after every refinement round. Results appended to `results/`.

## 4. Data split — HARD CONSTRAINT

- Enumerate all locally available public games (`arc.list_games()` or equivalent).
- Split deterministically by seed into **refine (~60%)** and **held-out (~40%)**. Store the split in `config/split.json`. Commit it.
- The Coach process must not be able to read held-out trajectories. Enforce in code (separate output dirs; coach loads from `trajectories/refine/` only), not by convention.
- Ablation later: re-split with different seeds and repeat. If gains only appear when a game's near-twin is in the refine set, nothing general was learned.

## 5. Edit tagging

Every harness edit gets a tag:
- `procedural` — how to explore/act, game-agnostic ("diff frames before acting", "probe ACTION5 once early", "use undo to test hypotheses for free")
- `perceptual` — about what things look like ("blue tile is the goal", "walls are color 5")
- `tooling` — new/changed python helper
- `meta` — changes to how the student reasons (hypothesis format, when to stop analyzing)

Hypothesis to test: `procedural` and `tooling` edits transfer to held-out; `perceptual` edits do not. Any edit that **names a specific game ID** is a bug and gets rejected automatically.

## 6. Metrics

Primary:
- **Held-out RHAE after N refinement rounds vs. v0 baseline.**

Secondary:
- Refine-set RHAE (gap vs held-out = overfitting signal)
- Actions per level vs human baseline, per game
- Levels completed per game
- Held-out RHAE gain attributable to each edit tag (ablate by reverting tag groups)
- LLM cost per eval run and per refinement round

## 7. Milestones — in this order

**M0 — Plumbing.** Env wrapper, frame history as numpy, REPL tool, trajectory logger, split config, eval script that prints RHAE. Validate with a random agent (expect ~0%) and a scripted agent on one easy game.

**M1 — v0 Student.** Hand-written harness. Cheap model. Goal: **nonzero held-out RHAE.** If the baseline is 0% everywhere, the Coach has nothing to work with — fix the student first. Do not start M2 until M1 has a nonzero number.

**M2 — Coach.** Refinement loop with tagging, logging, snapshots, rollback. Run 3–5 rounds on the refine set. Evaluate on both sets each round.

**M3 — Analysis.** Plot held-out vs refine RHAE per round. Ablate by tag. Re-split and repeat. Write up.

Optional later: swap to a stronger model for the student only once M0–M2 are trustworthy on the cheap model.

## 8. Proposed repo layout

```
arc-harness/
  CLAUDE.md
  SPEC.md
  pyproject.toml
  config/
    split.json            # refine/held-out game IDs, seed
    budget.py             # 5x human median per level
    models.py             # model names, default = cheapest
  harness/                # learned state (see §3)
  src/arc_harness/
    env.py                # arc_agi wrapper, frame history, action budget enforcement
    repl.py               # persistent python namespace exposed to the LLM
    student.py            # inner loop
    coach.py              # outer loop; reads trajectories/refine/ only
    scoring.py            # RHAE computation
    trajectory.py         # logging schema
    llm.py                # thin client, cost tracking
  scripts/
    eval.py               # --set refine|heldout|all --harness <snapshot>
    refine.py             # one coach round
    random_agent.py       # sanity baseline
  trajectories/
    refine/
    heldout/
  results/
```

## 9. Open questions to verify against the toolkit before M0

- Exact attributes on `FrameDataRaw` (frame grid(s), `state`, `levels_completed`, action counters). Read `pip show arc-agi` source; don't guess.
- Whether local envs expose human baseline action counts per level, or whether we need to fetch them from the API / hardcode from the public game metadata. RHAE can't be computed without them.
- How `env.reset()` interacts with level progression and action counting.
- Whether `ACTION7` (undo) counts as an action in scoring. (Assume yes until verified.)

## 10. Non-goals

- Topping the leaderboard.
- Kaggle competition eligibility (offline, no API calls — different project).
- Reimplementing prime-agent. We borrow ~3 ideas: persistent REPL over state, durable harness files, evidence-backed refine step with rollback. Nothing else.
- Training or fine-tuning models.
