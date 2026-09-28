# T10 depth 3 at width 512: summary (A, 2026-09-28 02:30 UTC)

All three runs here came from the user's Lightning Studio (the cascade tuning-seed and seed-0 runs are Lightning copies of runs A also did in its cloud session; the report's tables use A's cloud copies, kept in the report's runs tarball). The two copies of each run hold 87% / 87% and 93% / 92% of the test 0.1 -> 0.9 rise in the loss-alone window (README §4). Columns from `natplat_server.py summary`; `flat` is the longest <=3% CE window found from the loss alone, which at width 512 is the grokking plateau.

| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T10_D3_w512_cascade_r0.9_0.45_s0_T60k | 21880-49520 (27640) | 2.486 (0.72) | 0.16→0.96 | 0.97→1.00 | 0.25→0.26 | 0.25→0.50 | 0.06→0.08 / 0.17→0.23 / 0.43→0.48 | 0.61→0.77 | 20440→41180 | 1.00 | 4.7% | +3.1% |
| T10_D3_w512_cascade_r0.9_0.45_tune_T60k | 22200-50540 (28340) | 2.481 (0.72) | 0.11→0.81 | 0.97→1.00 | 0.21→0.29 | 0.24→0.49 | 0.06→0.08 / 0.17→0.24 / 0.45→0.49 | 0.60→0.76 | 21900→None | 0.84 | 4.3% | -1.0% |
| T10_D3_w512_lastslow_r1.0_0.5_tune_T60k | 19760-44240 (24480) | 2.491 (0.73) | 0.05→0.74 | 0.93→1.00 | 0.18→0.31 | 0.22→0.47 | 0.05→0.08 / 0.17→0.23 / 0.43→0.49 | 0.56→0.75 | 22020→None | 0.85 | 5.4% | -4.3% |
