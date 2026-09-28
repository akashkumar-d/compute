# Independent review: saved-state weighted updates

Verdict: PASS for the implementation, saved arithmetic, and stated fixed-state scope. The audit exposes a material base-order update discrepancy at seed 642, step 380. Higher orders agree much better at that state, but neither trajectory convergence nor an eigenspace error certificate follows.

Reviewed `weighted_update_audit.py`, `h2_weighted_updates.json`, the unchanged v8 `swsmall_periodic.py`, `swpop.py`, and `diagnostics.py`, and both v8 h2 candidate audits. This review performed source inspection, hashing, and arithmetic on saved JSON only: no model evaluations, training, remote access, or manuscript edits.

## Implementation and provenance

- The audit calls the actual training helper, producing `D = m*(gP, gV, 0.001*ga)` with `m=64`. Training subtracts `dt*D`; the common minus sign does not affect comparisons. The cap matches training exactly: per-neuron `||D_j|| / sqrt(||P_j||²+||V_j||²+a_j²)`, including both biases, followed by `dt=min(dt_max,h/max_j rate_j)`. The head multiplier belongs in the numerator only. Saved direction norms, block-error combination, cap maxima/indices, `dt`, and step norms satisfy these formulas to roundoff.
- Each vector relative difference uses the **higher-order vector norm**. Step comparisons use each order's own capped `dt`; `dt` relative differences use the higher-order `dt`. Multipliers 1, 2, 4 scale all four quadrature orders: `(16,48,24,12)`, `(32,96,48,24)`, `(64,192,96,48)` for `(pair,diag,z,x)`.
- All 7 recorded input hashes, all 9 source hashes, and the script hash independently match current files. Relative input paths resolve from `goal_followup_v9/`. The 9 source hashes also match the v8 higher-order audit and both original run receipts; each receipt's raw-file hashes match. The four records match the frozen selection and saved row indices/times.
- Across both v8 candidate audits, raw loss matches exactly; reconstructed raw-gradient differences agree within `3.5e-18`; shared `Amin`, mean alignment, rank-8/9 eigenvalues, and relative gaps agree to roundoff (largest absolute difference `2.2e-15`). The new Gram-asymmetry denominator is `||G||F`, whereas v8 diagnostics use `||sym(G)||F`; these are slightly different definitions, with negligible numerical difference here.

## Material results

Seed 642, step 380:

| Comparison | Raw gradient relative difference | Actual direction relative difference | Actual step relative difference | Direction angle |
|---|---:|---:|---:|---:|
| 1 versus 2 | 1.699685% | 10.900544% | 10.883243% | 4.925647° |
| 2 versus 4 | 0.010325% | 0.067199% | 0.067730% | 0.030928° |
| 1 versus 4 | — | 10.846030% | 10.828190% | 4.906294° |

The amplification is expected from the optimizer: at order 2, raw `||ga||=0.0278901` dominates `||gP||=0.00304302` and `||gV||=0.00307608`, but the actual head rate suppresses that block by 1000. The 1-versus-2 `gP` error is 15.3256%. The common factor `m` cancels in relative direction error. The cap remains neuron 62; `dt` changes only 0.026441% (1 versus 2) and 0.000885% (2 versus 4), so it does not remove the direction discrepancy. Both initialization states and seed 641 step 102 agree near machine precision.

The top-eight eigenspace calculations are correct: `Q[:8]` measures alignment against the teacher coordinate subspace, and singular values of `Q_low.T @ Q_high` measure angles between the learned subspaces. For seed 642, 2 versus 4 gives projector operator difference `0.02687535`, equal to the sine of the largest principal angle (1.54003°), and minimum squared principal cosine `0.99927772`. The squared projector/angle identity agrees within `3.2e-15` across all comparisons. This is a dimensionless operator distance, not a relative AGOP-matrix error.

At that state, `Amin` is `0.78417831 / 0.81190591 / 0.79673172` for orders 1/2/4. The 2-versus-4 shift is 0.01517419 despite a small matrix perturbation: the rank-eight gap is tiny, and `||M4-M2||op/gap4=114.32679`. This ratio is a conditioning diagnostic, not a useful upper bound below one or a certificate against the exact population operator.

These results support close order-2/order-4 **local updates at four selected saved states** and retention of the candidate endpoint alignment. They do not validate the base-order trajectory, all intermediate states, plateau timing, later release, or convergence to exact population flow. A fresh higher-order trajectory comparison remains necessary; endpoint order agreement cannot repair accumulated base-order integration error.

Reviewed report SHA-256: `687675653bede1fa80a41af4529ade843774c645ab35e47965599027cec5819f`.
Reviewed script SHA-256: `db6b0c886a2fc0c985ef767dc9a4350338db34fa482ed011c376f058c2f11152`.
