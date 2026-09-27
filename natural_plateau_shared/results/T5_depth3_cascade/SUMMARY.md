| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T5_D3_cascade_r0.5_0.25_tune_T100k | 39520-63100 (23580) | 1.602 (0.47) | 0.82→0.99 | 1.00→1.00 | 0.38→0.42 | 0.38→0.53 | 0.09→0.10 / 0.21→0.25 / 0.49→0.51 | 0.80→0.87 | 23400→46080 | 1.00 | 45.1% | +45.1% |
| T5_D3_cascade_r0.7_0.35_tune_T100k | 81800-99980 (18180) | 1.375 (0.40) | 1.00→1.00 | 1.00→1.00 | 0.38→0.41 | 0.66→0.70 | 0.09→0.09 / 0.25→0.26 / 0.50→0.51 | 0.87→0.87 | 21700→44800 | 1.00 | 13.3% | +8.6% |

Note (A): the table's "flat" column is the longest window with CE range ≤ 3% anywhere after the knee. For (0.7, 0.35) that window is the slow post-grokking stretch. The flat window that holds the most grokking is 26,880–41,820 (14.9k steps) at CE 1.664 (0.48·log p). Inside it:
- test accuracy goes 0.36 → 0.87 (63% of the 0.1 → 0.9 rise);
- first-layer AGOP (top-4) goes 0.25 → 0.34, and last-layer AGOP (top-30) 0.27 → 0.43;
- the layer-1 weight Fourier share goes 0.68 → 0.81.

For comparison, uniform r = 0.35 manages 43% (test 0.02 → 0.36), and depth 2 at r = 0.5, seed 0, manages 91%.
