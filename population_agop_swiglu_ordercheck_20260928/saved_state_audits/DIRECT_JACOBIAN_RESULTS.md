# Independent sampled Jacobian check

Four predeclared states, two fixed independent Gaussian replicates, nested8192/32768samples per replicate. Completed in6.71seconds on one CPU thread. No training. All input hashes unchanged.

The full input derivative includes both branches of the SwiGLU product rule. Geometry comes from the direct Jacobian SVD, avoiding the pair-quadrature calculation. The audited helper is unchanged from v6. Sampled AGOP is not exact population AGOP; ranges across two replicates are not confidence intervals.

| Seed / state | Minimum alignment at32768samples, replicate1 | Replicate2 |
|---|---:|---:|
| 641 / step0 | 0.00288536 | 0.00283685 |
| 641 / step102 | 0.70647633 | 0.57780480 |
| 642 / step0 | 0.00458119 | 0.00485800 |
| 642 / step380 | 0.78892653 | 0.79876466 |

Both candidate endpoints retain minimum-alignment gain above0.5 in both32768-sample replicates. The seed641 candidate varies more across sampled replicates (about0.58–0.71) than seed642 (about0.789–0.799), despite deterministic quadrature agreement for seed641. This is a reminder that sampling variability and deterministic integration error are different checks. Neither replaces trajectory validation or proves a population confidence bound.

Original all-update loss-window membership and refit scores are not recomputed by this Jacobian-only check. Retain the existing numerical caveat on the original trajectory.
