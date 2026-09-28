# Actual-update quadrature audit

Completed 2026-09-28T20:51:07.609159+00:00. Four preselected fixed states, no training, 40.68 seconds on one CPU thread. Prior data and source hashes unchanged.

Unlike the earlier unweighted-gradient comparison, this check uses the actual block-scaled direction m*(gP,gV,0.001*ga), the same maximum per-neuron relative update cap, and the resulting adaptive Euler step from the unchanged training engine. No state is updated.

| Seed / step | Orders | Relative update-direction difference | Relative actual-step difference | Learned-projector operator difference |
|---|---|---:|---:|---:|
| 641 / 0 | 1_vs_2 | 1.431908e-15 | 1.3390165e-15 | 3.689601e-15 |
| 641 / 0 | 2_vs_4 | 6.0634699e-15 | 5.9908251e-15 | 3.8864251e-15 |
| 641 / 102 | 1_vs_2 | 6.0426983e-16 | 4.3153803e-16 | 4.9154377e-14 |
| 641 / 102 | 2_vs_4 | 1.9064738e-15 | 1.6661195e-15 | 2.8887772e-14 |
| 642 / 0 | 1_vs_2 | 1.466702e-15 | 1.387243e-15 | 3.893159e-15 |
| 642 / 0 | 2_vs_4 | 6.7615734e-15 | 6.6898162e-15 | 3.0484779e-15 |
| 642 / 380 | 1_vs_2 | 0.10900544 | 0.10883243 | 0.053864619 |
| 642 / 380 | 2_vs_4 | 0.00067199388 | 0.00067729936 | 0.026875353 |

The sensitive seed642 checkpoint has a10.9% update-direction discrepancy at the original-to-doubled order comparison, despite a1.70% unweighted-gradient discrepancy. The head gradient dominates the unweighted norm and is downweighted during training. The original trajectory therefore cannot be called numerically confirmed.

Doubled-to-quadrupled orders reduce the update-direction discrepancy to0.0672%, and actual-step discrepancy to0.0677%. The top-eight learned projectors still differ by0.02688 in operator norm; agreement is improving but this is not a subspace error certificate. Both initialization states and the seed641 candidate are stable to floating-point precision.

All-order multipliers1/2/4 mean pair16/32/64, diagonal48/96/192, conditional24/48/96, teacher12/24/48. Canonical teacher normalization, student initialization, head rate and training scheme are unchanged.

Next: predeclared two-order seed642 trajectories from the same initialization, recomputing every-update loss windows and same-checkpoint alignment/refit criteria. These are numerical development checks, not fresh-seed confirmation. A full trajectory can still differ from these fixed-state comparisons. No manuscript or primary success criterion changes.
