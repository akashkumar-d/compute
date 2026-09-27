| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T7_D3_cascade_r0.8_0.4_tune_T100k | 79460-99960 (20500) | 1.351 (0.39) | 1.00→1.00 | 1.00→1.00 | 0.44→0.40 | 0.68→0.71 | 0.09→0.09 / 0.25→0.26 / 0.50→0.51 | 0.87→0.87 | 19720→38120 | 1.00 | 9.7% | +5.5% |
| T7_D3_cascade_r0.9_0.45_tune_T100k | 76480-99960 (23480) | 1.333 (0.39) | 1.00→1.00 | 1.00→1.00 | 0.37→0.33 | 0.67→0.70 | 0.09→0.09 / 0.25→0.25 / 0.50→0.50 | 0.86→0.85 | 19940→36240 | 1.00 | 7.0% | -3.5% |

Note (A): the table's "flat" column is the longest window with CE range ≤ 3% anywhere after the knee. Below is the flat window (≤ 3%) that holds the most of the test 0.1 → 0.9 rise. Depth 3, width 256, tuning seed, every setting:

| Setting (layer 2, layer 3 shrink rate × layer 1's) | Flat window | Test inside | Share of rise | CE range over grokking | First-layer AGOP (top-4) |
|---|---|---|---|---|---|
| uniform 0.4 | 4.8k | 0.23 → 0.53 | 38% | 13.4% | 0.21 → 0.24 |
| uniform 0.35 | 7.9k | 0.09 → 0.38 | 35% | 13.4% | 0.20 → 0.24 |
| cascade (0.5, 0.25) | 1.1k | 0.56 → 0.68 | 14% | 45% (CE rises) | 0.34 → 0.36 |
| cascade (0.7, 0.35) | 14.9k | 0.36 → 0.87 | 63% | 13.3% | 0.25 → 0.34 |
| cascade (0.8, 0.4) | 13.1k | 0.20 → 0.84 | 80% | 9.7% | 0.25 → 0.38 |
| cascade (0.9, 0.45) | 11.0k | 0.10 → 0.76 | 82% | 7.0% | 0.21 → 0.32 |
