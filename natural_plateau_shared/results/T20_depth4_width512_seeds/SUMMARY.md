# T20 depth 4 at width 512, seeds 1-2: summary (A, 2026-09-28 03:40 UTC)

Run on the user's Delta allocation, 60k steps (relaunched from checkpoints at 00:18 UTC). As in T19: grokking starts at 28k-40k steps and is unfinished at 60k (final test 0.49-0.66); in all four runs the longest <=3% CE window found from the loss alone starts at 20.4k-23.4k steps and runs to the last step at 0.82-0.83 log p, holding the whole rise so far; CE range over grokking so far 1.9-2.7%.

| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T20_D4_w512_r1.0_1.0_0.45_s1_T60k | 20480-59980 (39500) | 2.825 (0.82) | 0.01→0.66 | 0.45→1.00 | 0.17→0.17 | 0.17→0.51 | 0.03→0.07 / 0.11→0.20 / 0.27→0.39 / 0.58→0.65 | 0.42→0.68 | 29040→None | 0.66 | 2.7% | -2.6% |
| T20_D4_w512_r1.0_1.0_0.45_s2_T60k | 20400-59980 (39580) | 2.837 (0.83) | 0.00→0.49 | 0.31→1.00 | 0.16→0.22 | 0.16→0.46 | 0.03→0.06 / 0.10→0.19 / 0.25→0.37 / 0.57→0.64 | 0.39→0.65 | 39860→None | 0.49 | 2.0% | -2.0% |
| T20_D4_w512_r1.0_1.0_0.4_s1_T60k | 23380-59980 (36600) | 2.816 (0.82) | 0.04→0.65 | 0.76→1.00 | 0.15→0.17 | 0.20→0.51 | 0.04→0.07 / 0.12→0.20 / 0.29→0.40 / 0.61→0.66 | 0.47→0.68 | 28120→None | 0.65 | 2.0% | -1.4% |
| T20_D4_w512_r1.0_1.0_0.4_s2_T60k | 23200-59980 (36780) | 2.829 (0.82) | 0.00→0.50 | 0.50→1.00 | 0.16→0.22 | 0.17→0.46 | 0.03→0.06 / 0.11→0.19 / 0.27→0.38 / 0.59→0.65 | 0.42→0.66 | 37980→None | 0.50 | 1.9% | -1.9% |
