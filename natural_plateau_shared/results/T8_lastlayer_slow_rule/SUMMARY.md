| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T8_D3_lastslow_r1.0_0.5_tune_T100k | 73720-99960 (26240) | 1.314 (0.38) | 1.00→1.00 | 1.00→1.00 | 0.27→0.24 | 0.69→0.72 | 0.09→0.09 / 0.25→0.25 / 0.49→0.50 | 0.86→0.84 | 18560→34320 | 1.00 | 8.1% | -6.0% |
| T8_D4_lastslow_r1.0_1.0_0.5_tune_T120k | 96420-119980 (23560) | 1.406 (0.41) | 1.00→1.00 | 1.00→1.00 | 0.26→0.22 | 0.75→0.78 | 0.05→0.04 / 0.19→0.18 / 0.39→0.38 / 0.64→0.64 | 0.68→0.61 | 28200→49280 | 1.00 | 15.9% | -15.9% |

Note (A): the table's "flat" column is the longest ≤3% window anywhere after the knee. At depths 3 and 4 that is the slow approach to the final level after grokking. The window that matters here is the ≤3% window holding the most of the test 0.1 → 0.9 rise (`best_grok_window`, chosen knowing test accuracy):

| Run (tuning seed) | Best window | CE there | Test inside | Share of the rise | CE range, drift over grokking | Hidden-layer norms over grokking |
|---|---|---|---|---|---|---|
| depth 3, (1.0, 0.5) | 17.9k–28.2k | 0.50·log p | 0.06 → 0.72 | 77% | 8.1%, −6.0% | ×1.02 / ×1.03 / ×0.94 |
| depth 4, (1.0, 1.0, 0.5) | 32.7k–36.4k | 0.55·log p | 0.32 → 0.61 | 36% | 15.9%, −15.9% | ×1.02 / ×1.01 / ×1.00 / ×1.02 |

- The rule works at depth 3: it is close to the tuned cascade (0.9, 0.45) at 82% and 7.0%.
- It fails at depth 4. Grokking starts later (test 0.1 at 28.2k steps against 18.6k at depth 3), after layer 4 has already settled, so no layer shrinks during grokking and the CE falls 16%.
- T11 (local) and T14 (Lightning) try a slower last layer at depth 4.
