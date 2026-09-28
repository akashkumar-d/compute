# Higher-rank SwiGLU production driver

This isolated copy starts from `population_agop_scaling_1h_v1/swiglu/code`.
All four numerical engine files and requirements are byte-identical to that source.
The wrapper changes only execution/manifest validation and provenance recording.
Corrected prediction-risk formulas remain unchanged. Added geometry is posthoc.

Production launch, from this study's root:

```sh
AGOP_EXECUTION_SITE=SERVER python swiglu/code/run_one.py \
  --config swiglu/code/configs.json --tag EXACT_CONFIG_TAG \
  --out-dir results/swiglu/EXACT_CONFIG_TAG \
  --wall-seconds 1200 --diagnostic-seconds 90 \
  --deadline-utc FUTURE_TIME_WITH_Z_OR_UTC_OFFSET
```

The wrapper requires Linux and `AGOP_EXECUTION_SITE=SERVER` before importing the
numerical engine. Local training is prohibited. The unchanged engine modules are
implementation dependencies; launch through `run_one.py`, not their historical
standalone interfaces. CLI help and the pure synthetic geometry check are local-safe.

The config is a nonempty list with unique safe `tag` values. Each row has a
nonempty `cell`, integer `rank`, and `args` including `d`, `m`, and `c`. Require
`rank=len(c)>=1`, `d>=2*rank`, and `m>=rank`. The parent owns the frozen config.
Every launch records its actual configuration hash, current Python source hashes,
source-manifest hash, diagnostic schema and numerical resolution threshold.
No source-manifest hash is claimed for configurations not yet created.

The numerical experiment remains adaptive Euler approximating population gradient
flow with a profiled intercept, Gaussian population quadrature, and the existing
initialization/update rule. The wrapper's wall deadline is soft: an in-flight
population evaluation finishes before graceful stop. Existing completed results
are skipped; `completed=true` records an attempted outcome, including censoring.
Preserve original files and use distinct output directories for new attempts.

## Added geometry

The teacher subspace is `U=span(e_1,...,e_r)` and `Q_r` comprises the top-r AGOP
eigenvectors. All values are numerical; flags are not rigorous certificates.

- `agop_principal_cosines_squared_descending`: squared singular values of
  `U^T Q_r`, descending. Existing `agop_A` is their mean; `agop_Amin` their minimum.
- `agop_teacher_axis_captures`: diagonal of `U^T Q_r Q_r^T U`, one entry per axis.
- `agop_teacher_trace_energy_fraction`: `trace(U^T M U)/trace(M)`.
- `agop_balanced_weakest_direction_energy`: `r*lambda_min(U^T M U)/trace(M)`;
  equals one when M is isotropic on U and zero on its orthogonal complement.
- `agop_teacher_axis_energy_fractions`: `diag(U^T M U)/trace(M)`.
- `agop_teacher_block_eigenvalues_descending`: spectrum of `U^T M U`.
- `agop_eigenvalues_descending`, `agop_rank_r_eigenvalue_relative`,
  `agop_rank_r_relative_gap`: retain full M spectrum and the existing frozen-state
  audit formulas `lambda_r/scale`, `(lambda_r-lambda_(r+1))/scale`, where
  `scale=max(abs(lambda_1),1e-300)`.

For threshold `g=1e-9`, `agop_psd_resolved` requires positive lambda_1 and
lambda_min/scale >= -g. `agop_leading_resolved` additionally requires the top
relative gap >g. `agop_full_rank_resolved` instead requires lambda_r/scale >g
and the rank-r relative gap >g. A tied leading eigenvalue can leave top-one
unresolved while the top-r subspace is resolved. Zero/nonpositive trace leaves
energy fractions null. Values are not clipped; PSD and resolution flags remain
available to decide whether a displayed score is interpretable.

Use corrected `pinv['1e-12']['actual_mse']` for the primary unrestricted numerical
refit. Keep all four cutoffs and actual ridge risk. The source `refit` is a
penalized objective. Prefix endpoints come from all-update losses and the last
saved state inside the .001 initial absolute-change prefix. These definitions
remain separate from the ReLU driver.

Validation: run `python -B review/SWIGLU_DRIVER_CHECK.py` from the study root.
It parses source, verifies engine hashes, and checks tiny synthetic matrices and
configuration/server guards. It evaluates no population moments or training steps.
