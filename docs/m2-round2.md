# M2 round 2 — and the measurement problem it exposed

Round `20260913T125935Z`. Coach: deepseek-v4-flash, effort high, one call, $0.0014.
Harness `bc653ad4e040` -> `99d565924aff` -> rolled back to `bc653ad4e040`.

## Getting a different proposal
Round 2's first attempt re-proposed round 1's reverted rules almost verbatim, even though the prompt listed
them under "already tried and REVERTED" with the measured outcome. Persuasion did not work; a referee-side
check did. `validate()` now rejects an edit whose wording is too close to a reverted one
(sequence ratio >= 0.45 or Jaccard >= 0.38 over normalised words; measured re-proposals scored 0.46-0.54 /
0.39-0.49, a genuinely different rule on the same topic scored 0.34 / 0.32). The coach then produced a
different batch: state the action and its reason before acting; an endgame fallback that fires only with
fewer than 10 actions left; and an `analysis.py` that prints budget used and a no-op notice.

**A known hole:** the third edit embedded round 1's "commit past half budget, stop probing" idea as a string
inside the analysis code, where it scored only 0.33/0.34 against the reverted rule — below any threshold that
does not also block legitimate edits. Similarity checks cannot catch a paraphrase. Measurement has to.

## The pilot, and then the same pilot again
Four refine games, level 1, 60-action cap. Then **the identical configuration a second time** with the LLM
cache disabled, to see how much of the result was the harness and how much was chance.

| game | human | v0 | round 1 | round 2 run A | round 2 run B |
|---|---|---|---|---|---|
| ls20 | 22 | 17 solved | 24 solved | **60 failed** | **25 solved** |
| s5i5 | 20 | 60 failed | 60 failed | 60 failed | 60 failed |
| tn36 | 32 | 11 solved | 60 failed | 12 solved | 12 solved |
| vc33 | 7 | 35 failed | 35 failed | **6 solved** | **35 failed** |
| **mean score** | | 0.0179 | 0.0075 | 0.0179 | 0.0158 |

**Half the games flipped outcome between two runs of the same harness.** vc33 went from solving in six
actions, better than the human median of seven, to not solving at all. ls20 went from total failure to a
solve. Only the clearly-easy game (tn36: 12, 12) and the clearly-hard one (s5i5: 60, 60) were stable.

## What this means
A marginal game is close to a coin flip, and a flip is worth a full level score — the same magnitude as any
effect an edit batch could plausibly have. Every keep-or-revert decision made on single runs, including the
round 1 rollback, rests partly on luck. Round 1's verdict survives scrutiny because its regression had a
mechanism visible in the trajectory (tn36 using six click positions in sixty actions) and was large. Round 2's
does not: averaged over both runs it scores 0.0169 against v0's 0.0179, which is nothing.

Round 2 was rolled back on that basis — not a regression, but no demonstrated benefit either, and an unvalidated
batch would contaminate the baseline the whole experiment compares against.

## The fix
`eval.py --repeats N` runs each game N times and averages the scores before computing RHAE, so one lucky or
unlucky run cannot move the set. Repeats bypass the reply cache, or every repeat would replay the first.
Cost scales linearly: the level-1 refine scan at N=3 projects to about $19 against $6.4 at N=1.

Until a round is judged at N>=3, no rollback decision from this project should be treated as evidence.
