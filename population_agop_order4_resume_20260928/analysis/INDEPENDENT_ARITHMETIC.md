# Independent continuation arithmetic review

PASS for the saved arithmetic and diagnostic coverage. Order 4 retains the same 5% material sequence as order 2. Neither order has a qualifying saved checkpoint in the 1% initial window.

This review only reads saved JSON and NPZ arrays. It performs no model evaluation, training or network access. Source provenance and the strict continuation anchor are reviewed separately.

| Order | 1% end / exit | 5% end / exit | 5% first candidate | ΔA_min | Cutoff-envelope gain | Later loss drop |
|---|---|---|---|---:|---:|---:|
| 2 | 257 / 258 | 381 / 382 | 358 | 0.5654931626 | 0.7876448924 | 0.1570800229 |
| 4 | 257 / 258 | 381 / 382 | 358 | 0.5480117692 | 0.7876379312 | 0.1573121576 |

Both histories contain 620 evaluated states / 619 applied updates. Exact prefixes use all recorded losses and strict max/min crossings, with no tolerance or interpolation. Both 1% grids have 33 diagnosed states and both 5% grids have 41. The same saved candidate steps 358, 380 and 381 qualify in each 5% window. All saved original 1e-9 spectral guards and additional cutoff-spread/residual screens pass.

The continuation diagnoses all 54 saved checkpoints: all 45 original required states are retained, including extra interruption state 437, and all 53 order-2 checkpoint step indices are present. No missing or duplicate diagnostic steps were found.

| Quantity | Order 2 | Order 4 |
|---|---:|---:|
| 1% end time | 24.155476038951 | 24.155476038201 |
| 5% end time | 34.043930717847 | 34.043813571945 |
| First candidate time | 31.951493447244 | 31.951379027807 |
| First candidate minimum alignment | 0.570649621988 | 0.553168228583 |
| First candidate raw loss | 0.955899183996 | 0.955897477180 |
| Terminal raw loss | 0.798819161101 | 0.798585319542 |
| Terminal time | 53.826758017469 | 53.828390875386 |

Both first candidates achieve a later loss decrease of at least 0.1 variance at step 586 and reach their lowest recorded loss at step 619. The target variance is 1. The maximum 1% minimum-alignment gain is 0.0104784203 for order 4, far below 0.5.

Across 53 common checkpoint indices, the largest relative concatenated parameter difference is 0.69538764% at step 619. The maximum absolute minimum-alignment difference is 0.01748139341; the maximum raw-loss difference over all updates is 0.0002338415582. Per-state values and prefix-restricted maxima are preserved in the JSON receipt.

The comparison uses corresponding update indices on two trajectories. It is numerical corroboration on reused seed 642, not independent-seed evidence or a formal quadrature/eigenspace certificate. The unrestricted pseudoinverse cutoff envelope is a sensitivity comparison, not a bound on optimal-risk improvement.

No arithmetic or required-diagnostic coverage bugs were found. Every input retained its SHA-256 digest through the review.
