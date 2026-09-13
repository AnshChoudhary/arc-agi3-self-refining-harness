# M2 round 1 — proposed, applied, rolled back

Round `20260913T101720Z`. Coach: deepseek-v4-flash, effort high, $0.0016, one call.
Harness `bc653ad4e040` -> `a2139f8e5145` -> rolled back to `bc653ad4e040`.

## What the coach proposed
Three `procedural` edits, all evidence-cited and game-agnostic; all passed validation.

1. **Replace rule 3** (no-ops). Old: "changed nothing *twice in a row* is a no-op". New: "An action that
   changed nothing is a no-op in that state. Do not use it again until the state changes. After probing each
   action once, stop probing and act on your best hypothesis."
2. **Add rule 11** (budget). "If you have used more than half the budget and still don't have a concrete plan
   to finish, commit to the most promising hypothesis and stop exploring."
3. **Add rule 12** (RESET). "Only use it if you are certain you are stuck and have a new plan."

Its stated analysis was accurate: it identified the no-op waste and budget exhaustion we had found by hand.

## Pilot: 4 refine games, level 1, cap 60, same seed and settings as the v0 baseline
Two games v0 solved (regression check) and two it failed with high no-op rates (headroom). Held-out games were
deliberately not used: choosing whether to keep an edit on held-out evidence would leak the thing we measure.

| game | v0 actions | v1 actions | v0 score | v1 score | v0 no-op | v1 no-op | v0 calls/action | v1 calls/action |
|---|---|---|---|---|---|---|---|---|
| ls20 | 17 | 24 | 0.036 | 0.030 | 0% | 4% | 5.6 | 6.6 |
| tn36 | 11 | **60 (failed)** | 0.036 | **0.000** | 36% | **100%** | 5.3 | 3.2 |
| s5i5 | 60 | 60 | 0.000 | 0.000 | 73% | 93% | 5.5 | 4.2 |
| vc33 | 35 | 35 | 0.000 | 0.000 | 74% | 74% | 6.6 | 6.0 |

Mean game score **0.0179 -> 0.0075**. Two regressions, no improvements. Rolled back; the reason is recorded in
`harness/history/edits.jsonl`. Pilot cost $0.80.

## Why it backfired — the useful part
tn36 has one available action, ACTION6, a click at (x, y). Under the new rules the student used **six distinct
click positions across sixty actions**, having "probed the action once" and then committed. It settled on a
wrong theory (that a top-bar counter had to be decremented to one) and ground it out to the budget; its last
note reads "one click left". v0, with the weaker rules, kept exploring positions and finished in eleven.

The lesson is about the shape of the action space, not about any game. **"Probe each action once, then commit"
is sound for ACTION1-5, whose meaning is fixed, and wrong for ACTION6, whose meaning is its coordinates:
treating the click as one action undercounts the real choice set by a factor of 4,096.** Rule 11 compounded
it by converting "no concrete plan" into "commit to the best guess" exactly when more exploration was needed.

Both edits sounded like efficiency and were tagged `procedural`, the tag we hypothesised would transfer.
Being evidence-backed, game-agnostic, and validated was not enough to make them good.

## Consequences for the design
- The rollback rule earned its place on the first round.
- A cheap pilot (4 games, $0.80, 40 min) caught a regression the full evaluation would have charged $7 to find.
  Worth keeping as a standard gate before a full round.
- The coach now sees previously rolled-back edits so it does not re-propose them.
