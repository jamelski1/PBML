# Findings: is there any learnable signal in Powerball draws?

**Short answer: no — and we tested it properly.** Two independent hypotheses,
each with the correct instrument, both returning results fully consistent with a
fair, memoryless random process. This documents what we ran and what we found.

## The data

- **1,387 draws**, the current-rules era (2015-10-07 → 2026-08-10; white 1–69,
  Powerball 1–26). Source: NY State Open Data.
- This is the entire population, not a sample — there is one Powerball draw
  nationwide per drawing, so no more data exists for this era. It grows ~150
  draws/year.

## Methodology: a permutation-test referee

The central risk in "try many models until one looks good" is false positives
from multiple comparisons. Our guard: every temporal experiment is judged against
a **permutation null** — the same pipeline trained on copies with the past→future
pairing shuffled away (marginals preserved). A result counts as signal only if it
beats that null with **p < 0.01** on a held-out target. Validated on
known-random synthetic data: the referee produced **no false positives**.

## Hypothesis 1 — temporal predictability (does the past predict the next draw?)

6 models (`freq`, `mlp`, `lstm`, `gru`, `tcn`, `transformer`) × 5 target framings
(exact whites, Powerball, sum-bucket, odd-count, high-count), plus feature
engineering (recency/gap, calendar). Promising candidates went through a 30-shuffle
permutation test.

**Result: null.** Best outcome was `lstm/sum_bucket` at z = +0.66 (p = 0.167) —
squarely inside the noise. Several models scored *below* their shuffled null.
Notably, targets that topped the raw-gap leaderboard (`sum_bucket`, `high_count`,
gaps up to +0.08) collapsed under the permutation test: those gains were the
models learning a fixed marginal distribution, not predicting anything.

## Hypothesis 2 — physical bias (is any ball worn/heavy?)

The correct instrument for a physical defect is a direct frequency test over all
draws (no windowing/split → maximum power): chi-square goodness-of-fit plus a
per-number Bonferroni-corrected binomial test.

**Result: null.**

| | chi² p-value | hottest | coldest | biased after correction |
|---|---|---|---|---|
| White balls | 0.167 | #21 (123×, exp. 100.5) | #13 (75×) | **NONE** |
| Powerball | 0.522 | #4 (66×, exp. 53.3) | #16 (40×) | **NONE** |

The "hot" #21 is +2.3σ — and across 69 numbers you expect ~1.4 numbers that
extreme by chance, so one is exactly what randomness predicts. It does not
survive multiple-comparison correction.

## Conclusion

Both the temporal and marginal routes — which between them cover essentially every
way a lottery could be exploitable from historical numbers alone — return results
indistinguishable from a fair random process. **No model in this study can predict
Powerball better than random guessing, and none should be expected to.** The
value here is the negative result and the rigor used to reach it.

## The one untested avenue

A bias confined to a single physical **ball set / machine** (Powerball rotates
several) would be diluted below detection when draws are pooled, as they were
here. Testing it requires draw-level ball-set IDs, which the NY feed does not
include. This remains open — as a data-sourcing problem, not a modeling one — and
is a long shot even with the data in hand.

## Reproduce

```python
import sys; sys.path.insert(0, "src")
import experiments as E, runner as R
df = E.load_draws()
E.physical_bias_test(df)                                    # Hypothesis 2
R.permutation_test(df, {"model": "transformer",
                        "target": "white_multihot", "window": 24})  # Hypothesis 1
```
