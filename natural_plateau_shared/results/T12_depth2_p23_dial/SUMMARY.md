| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T12_D2_p23_lw0.015_r0.35_tune | 27780-39760 (11980) | 0.743 (0.24) | 0.25→0.69 | 0.99→1.00 | 0.20→0.29 | 0.29→0.39 | 0.08→0.09 / 0.25→0.28 | 0.76→0.81 | 22400→None | 0.70 | 19.3% | +16.1% |
| T12_D2_p23_lw0.015_r0.4_tune | 23980-33600 (9620) | 0.753 (0.24) | 0.17→0.64 | 0.99→1.00 | 0.19→0.24 | 0.28→0.37 | 0.08→0.09 / 0.23→0.27 | 0.73→0.79 | 21200→None | 0.81 | 10.4% | +1.9% |

Note (A): p = 23, depth 2, layer-1 decay 0.015, tuning seed, 40k steps. The layer-2 rate r is the dial (T9 has r = 0.5):
- r = 0.5: the CE falls 16.5% through grokking; layer 2 has already settled (×0.99).
- r = 0.4: balanced. The ≤3% window holding the most of the rise holds 59% of it (test 0.17 → 0.64). Over grokking to 40k the CE range is 10.4% and the drift +1.9%; layer 2 shrinks ×0.92.
- r = 0.35: the CE rises 16% (layer 2 ×0.87).
- Grokking at p = 23 is slow: none of these reaches test 0.9 by 40k steps. T16 reruns r = 0.45 and 0.4 for 80k steps on three seeds.
