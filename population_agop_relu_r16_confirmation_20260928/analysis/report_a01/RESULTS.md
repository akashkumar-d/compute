# Four fresh-seed confirmation: ReLU h2, rank 16

All four prespecified seeds 9351–9354 are retained. Development seeds 641/642 are not pooled. The selected recipe and criteria were frozen before these outcomes; this is confirmation of that recipe only.

| Initial window | Original candidates | Numerically qualified | Qualified with later loss release |
|---|---:|---:|---:|
| 1% | 4/4 | 4/4 | 4/4 |
| 5% | 4/4 | 4/4 | 4/4 |

## Every planned outcome

| Seed | Process | Saved stop | Final raw MSE | Diagnostics complete | 1% / 5% qualified assessment |
|---|---|---|---:|---|---|
| 9351 | exit_zero | force_time_horizon | 0.0193 | True | qualified_candidate_observed / qualified_candidate_observed |
| 9352 | exit_zero | force_time_horizon | 0.0194 | True | qualified_candidate_observed / qualified_candidate_observed |
| 9353 | exit_zero | force_time_horizon | 0.0194 | True | qualified_candidate_observed / qualified_candidate_observed |
| 9354 | exit_zero | force_time_horizon | 0.0193 | True | qualified_candidate_observed / qualified_candidate_observed |

## First numerically qualified saved checkpoint

The two windows are reported separately even when their first qualified checkpoint is the same.

| Seed | Window | Step | Force time | ΔA_min | Refit lower gain / Var(Y) | Later raw-loss drop |
|---|---:|---:|---:|---:|---:|---:|
| 9351 | 1% | 63 | 31.5 | 0.897 | 0.684 | 0.981 |
| 9351 | 5% | 63 | 31.5 | 0.897 | 0.684 | 0.981 |
| 9352 | 1% | 46 | 23 | 0.852 | 0.551 | 0.981 |
| 9352 | 5% | 46 | 23 | 0.852 | 0.551 | 0.981 |
| 9353 | 1% | 46 | 23 | 0.71 | 0.575 | 0.981 |
| 9353 | 5% | 46 | 23 | 0.71 | 0.575 | 0.981 |
| 9354 | 1% | 46 | 23 | 0.849 | 0.573 | 0.981 |
| 9354 | 5% | 46 | 23 | 0.849 | 0.573 | 0.981 |

## Figures and interpretation

- [Four-seed median and seed range: full / 1% / 5%](four_seed_median.png). All displayed medians require 4/4 finite support. Bands are seed ranges, not confidence intervals.
- Individual seeds: [full](seeds_full.png), [1%](seeds_initial_1pct.png), [5%](seeds_initial_5pct.png). Each seed uses its own loss-only zoom; the median uses the smallest common loss-only window.
- Matching SVGs, a four-page PDF, raw plotted per-seed arrays, four-seed aggregate arrays and complete arm records are saved alongside these figures.

The exact all-update initial max/min loss windows are 1.01 and 1.05. Each candidate uses one saved checkpoint with ΔA_min ≥ 0.5 and bounded-refit initial-lower minus current-upper gain ≥ 0.1 Var(Y), plus the original and additional numerical screens. Later release requires a subsequent raw-loss decrease ≥ 0.1 Var(Y) from a qualified candidate. Cell-level canonical `complete_sequences` tracks original candidates; this report explicitly uses `first_qualified_complete_sequence` for qualified later release.

Raw MSE equals variance-normalized MSE here because the fixed h2 target has Var(Y)=1. The raw teacher and profiled output intercept are unchanged. Bounded refits retain l2 ≤ 32 and l1 ≤ 256. The force-time horizon is 1500 and the raw-loss stop is 0.01; horizon completion does not imply reaching the loss target.

Curves use saved data only. Missing diagnostics remain gaps, unresolved finite values retain warnings, and no interpolation determines a candidate. Four-seed aggregation is a separately named presentation adapter using unchanged canonical metric/interpolation functions; the existing two-seed aggregate is untouched.

This is numerical confirmation of one selected recipe with four fresh seeds. It does not establish a high-probability, architecture-wide or formal population claim. No new model evaluations, quadrature tests, Jacobian audits or training were performed. All saved numerical warnings, completion and censoring fields remain in the records.
