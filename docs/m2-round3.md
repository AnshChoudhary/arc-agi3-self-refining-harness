# M2 round 3 — not run; what the budget bought instead

Round 3 was not completed. The coach call stalled for 28 minutes against provider gateway timeouts and was
killed, and the monthly key allowance ($18) was exhausted part-way through the control arm. It resets
2026-10-01. What the spend did buy is the measurement the project most needed.

## Repeated runs of one unchanged harness (v0, `bc653ad4e040`, level 1, 60-action cap)

| game | n | solved | actions per run (F = failed) | mean score | sd | sem |
|---|---|---|---|---|---|---|
| ls20 | 5 | 3/5 | 17, 41F, 28F, 23, 19 | 0.0208 | 0.0190 | 0.0085 |
| s5i5 | 5 | 0/5 | 60F, 57F, 2F, 59F, 53F | 0.0000 | 0.0000 | 0.0000 |
| tn36 | 5 | 3/5 | 11, 14, 12, 16F, 7F | 0.0214 | 0.0196 | 0.0087 |
| vc33 | 5 | 4/5 | 35F, 15, 9, 8, 11 | 0.0142 | 0.0108 | 0.0049 |

**A game's per-run standard deviation is about as large as its mean.** Three of the four games are not
"solvable" or "unsolvable" by this student; they are solvable with some probability. vc33 solves four times
out of five and the single run in the M1 baseline scan happened to be the failure. s5i5 never solves.

Two consequences for every number this project has produced:

- **The M1 baseline is one draw, not a fixed quantity.** On these four games the single-run scan recorded
  0.0179; the mean over five runs each is 0.0141. The published held-out figure of 0.0176 carries the same
  kind of uncertainty and should be read as approximate.
- **Effect sizes must clear the noise.** `scripts/variance.py` reports the standard error of a set mean from
  the repeats available. On the four-game pilot set a difference below roughly 0.008 is indistinguishable
  from chance, which means round 2's measured 0.002 was never interpretable, and round 1's -0.010 was only
  marginally so (its rollback rests on the mechanism in the trajectory, not on the number alone).

## Cost of doing this properly
Detecting a change of about 0.01 needs roughly five runs per game per arm. On the four-game pilot set that is
about $5 per round for control and treatment together. On the full fifteen-game refine set at three repeats it
is about $19 per arm, plus about $13 for held-out. A serious M2 with three rounds needs roughly $100.

## Fixes made while blocked
- Transient 5xx and timeouts are retried like rate limits; a 524 had killed a run two actions in.
- `effort` is recorded in the trajectory. It was only in the results row, and pooling runs across effort
  settings produced a wrong first version of the table above.
- `scripts/variance.py` reports per-game spread and the detectable-effect floor for any harness.
