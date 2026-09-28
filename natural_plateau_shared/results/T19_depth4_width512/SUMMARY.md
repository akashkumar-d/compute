# T19 depth 4 at width 512: summary (A, 2026-09-28 02:30 UTC)

Run on the user's Lightning Studio, 60k steps. Grokking starts at 26k-34k steps and is unfinished at 60k (final test 0.55-0.87). In all four runs the longest <=3% CE window found from the loss alone runs from 20.5k-27.1k steps to the last step at 0.82 log p, so the whole rise so far is inside it, at both rates on each seed. T24 (account B) repeats the tuning seed and seed 0 at 0.4x for 120k steps.

| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T19_D4_w512_r1.0_1.0_0.35_tune_T60k | 27060-59980 (32920) | 2.804 (0.82) | 0.05→0.56 | 0.93→1.00 | 0.23→0.33 | 0.23→0.51 | 0.04→0.07 / 0.14→0.20 / 0.32→0.41 / 0.65→0.69 | 0.49→0.68 | 32820→None | 0.56 | 1.3% | -0.7% |
| T19_D4_w512_r1.0_1.0_0.45_s0_T60k | 20520-59980 (39460) | 2.832 (0.82) | 0.01→0.87 | 0.41→1.00 | 0.21→0.29 | 0.19→0.51 | 0.03→0.07 / 0.11→0.19 / 0.27→0.39 / 0.58→0.64 | 0.42→0.67 | 27700→None | 0.87 | 2.6% | -2.5% |
| T19_D4_w512_r1.0_1.0_0.4_s0_T60k | 23420-59980 (36560) | 2.824 (0.82) | 0.05→0.86 | 0.64→1.00 | 0.26→0.32 | 0.22→0.52 | 0.04→0.07 / 0.12→0.20 / 0.29→0.40 / 0.61→0.66 | 0.47→0.68 | 26380→None | 0.86 | 2.0% | -0.8% |
| T19_D4_w512_r1.0_1.0_0.4_tune_T60k | 23360-59980 (36620) | 2.817 (0.82) | 0.01→0.55 | 0.65→1.00 | 0.18→0.27 | 0.19→0.50 | 0.04→0.07 / 0.12→0.20 / 0.29→0.40 / 0.62→0.67 | 0.45→0.68 | 34460→None | 0.55 | 2.0% | -2.0% |
