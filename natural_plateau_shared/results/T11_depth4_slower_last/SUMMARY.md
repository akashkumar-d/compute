| run | flat | CE | test | refit | agop_h1_top4 | agop_last_top30 | AH | W1four | grok | final | grok_CE_range | grok_CE_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11_D4_lastslow_r1.0_1.0_0.25_tune_T120k | 91740-119980 (28240) | 1.482 (0.43) | 1.00→1.00 | 1.00→1.00 | 0.37→0.35 | 0.76→0.78 | 0.08→0.08 / 0.26→0.25 / 0.49→0.48 / 0.76→0.74 | 0.83→0.80 | 23120→42160 | 1.00 | 50.3% | +50.3% |
| T11_D4_lastslow_r1.0_1.0_0.35_tune_T120k | 94280-119980 (25700) | 1.451 (0.42) | 1.00→1.00 | 1.00→1.00 | 0.32→0.27 | 0.74→0.77 | 0.07→0.06 / 0.23→0.22 / 0.44→0.43 / 0.69→0.68 | 0.77→0.73 | 26260→46760 | 1.00 | 5.8% | -2.6% |

Note (A): the table's "flat" column is the longest ≤3% window after the knee. At depth 4 that is the slow approach to the final level after grokking. The ≤3% window holding the most of the test 0.1 → 0.9 rise (`best_grok_window`, chosen knowing test accuracy):

| Run (tuning seed) | Best window | CE there | Test inside | Share of the rise | CE range, drift over grokking | Layer-4 norm over grokking |
|---|---|---|---|---|---|---|
| depth 4, (1.0, 1.0, 0.35) | 26.5k–41.7k | 0.53·log p | 0.10 → 0.84 | 93% | 5.8%, −2.6% | ×0.91 |
| depth 4, (1.0, 1.0, 0.25) | 33.4k–35.0k | 0.43·log p | 0.70 → 0.79 | 11% | 50.3%, +50.3% | ×0.47 |
| depth 4, (1.0, 1.0, 0.5) (T8) | 32.7k–36.4k | 0.55·log p | 0.32 → 0.61 | 36% | 15.9%, −15.9% | ×1.02 |

Inside the (1.0, 1.0, 0.35) window:
- first-layer AGOP (top-4) rises 0.22 → 0.34, against random frames at 0.14 → 0.15;
- last-layer AGOP (top-30) rises 0.23 → 0.43, against random frames at 0.21 → 0.32;
- A_H rises in all four layers;
- the layer-1 weight Fourier share rises 0.52 → 0.70.
