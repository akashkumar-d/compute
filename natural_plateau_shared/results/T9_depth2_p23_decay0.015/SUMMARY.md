| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T9_D2_p23_lw0.015_r0.5_s0 | 18680-26240 (7560) | 0.811 (0.26) | 0.01→0.45 | 0.93→0.98 | 0.24→0.30 | 0.25→0.33 | 0.07→0.08 / 0.21→0.25 | 0.68→0.76 | 21280→35640 | 0.95 | 14.2% | -14.2% |
| T9_D2_p23_lw0.015_r0.5_tune | 18260-25920 (7660) | 0.789 (0.25) | 0.08→0.36 | 0.95→1.00 | 0.16→0.18 | 0.25→0.32 | 0.07→0.08 / 0.20→0.24 | 0.67→0.75 | 18840→None | 0.88 | 18.5% | -16.5% |

Note (A): p = 23 at depth 2 with layer-1 decay 0.015 (as at p = 31 and 47) instead of 0.01 (T3):
- The plateau rises from 0.10·log p to 0.25–0.26·log p.
- The ≤3% window holding the most of the test 0.1 → 0.9 rise holds 32% (tuning seed) and 44% (seed 0), against 13% and 5% with decay 0.01.
- The CE still falls 14–19% through grokking. The reason: layer 2 has finished shrinking before grokking starts (norm ×0.99 over grokking, against ×0.88 at p = 31 and ×0.70 at p = 47), so nothing offsets the growing margins.
- Grokking starts later at p = 23 (test 0.1 at 18.8k–21.3k steps, against about 16k at p = 31 and 13–14k at p = 47).
- T12 tries a slower layer 2 (r = 0.4 and 0.35) at p = 23.
