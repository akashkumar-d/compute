# Independent confirmation review

PASS: all four predeclared fresh-seed arms have recorded zero exits, complete training/diagnostics, and original plus numerically qualified complete sequences inside both the 1% and 5% initial windows. Checked 596 raw diagnostic rows, all 12,000 updates, eight prefix endpoints, and strictly later loss-release arithmetic. No discrepancies.

| Seed | 1% end step/time | 5% end step/time | First qualified step/time | ΔAmin | Numerical lower refit gain | Later loss drop | Final raw MSE |
|---:|---|---|---|---:|---:|---:|---:|
| 9351 | 146 / 73 | 169 / 84.5 | 63 / 31.5 | 0.897249097 | 0.683613499 | 0.980644669 | 0.019326635 |
| 9352 | 146 / 73 | 169 / 84.5 | 46 / 23 | 0.851775431 | 0.550787036 | 0.980577987 | 0.019413882 |
| 9353 | 146 / 73 | 169 / 84.5 | 46 / 23 | 0.710397972 | 0.575287991 | 0.980619234 | 0.019372352 |
| 9354 | 145 / 72.5 | 169 / 84.5 | 46 / 23 | 0.849106173 | 0.572692061 | 0.980684284 | 0.019307059 |

The denominator is four fresh seeds (9351–9354), separate from development seeds 641/642. Each seed remains individually reported. Every run stopped at the declared force-time horizon 1500 after 3,000 accepted updates; final raw MSE remains above the 0.01 loss target. All later minima in the table occur at step 3000, strictly after the qualifying state.

The first qualified state is the same for both windows in each run. Arithmetic uses current minus initial minimum alignment, initial refit lower value minus max(0, current feasible refit), and candidate raw loss minus the strictly later dense-history minimum. Here Var(Y)=1, so the two material-loss thresholds are 0.1. Numerical screens were rebuilt from raw saved diagnostics; native legacy >1e-5 flags were not used.

Transfer completeness relies on the parent’s 657-file/90,050,223-byte audit. This review separately hashes its read inputs and checks their stability; it does not repeat the bulk checkpoint hash audit. The canonical optional COMPLETE.json is absent, but the completed STATUS.json contains all four zero exits and each arm has DONE.json.

These results confirm a selected recipe on four fresh seeds. They do not establish a general success probability, rank scaling law, architecture-wide claim, or formal population certificate. No model evaluations, training, quadrature checks or independent Jacobian tests were performed. Numerical screens remain evidence, not certification.

A second read-only raw-receipt review independently agrees on all IDs, zero exits, configurations, 149 diagnostics per arm and exact loss/accepted-clock histories. Use cumulative accepted_h for accepted-force time; gd_cumulative_time.npy records a different clock ending at 192000.
