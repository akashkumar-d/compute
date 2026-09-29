# Transition-resolving quadrature prototype

This is a numerical diagnostic for the weak pure-cubic SwiGLU experiments. It is not a new training recipe, a successful learning result, or permission to overwrite old kernels.

Why: the earlier fixed-state checks found that student-moment quadrature accounts for most of the loss/gradient slope discrepancy at the saved plateau state and at the late state under doubled order. Correcting only the teacher integral is therefore insufficient.

## Scope

- Student: SiLU gate times linear value, with trainable gates, values, biases and heads.
- Teacher: only normalized additive pure h3 with the original coefficients and normalization. No other teacher link is admitted by this prototype.
- Original r=8,d=64,m=64 states: step472 (last saved valid 5% initial plateau) and step2260 (before the old non-descent step).
- First validation uses four neurons selected by the largest saved gate norms at each state. Alpha is multiplied by 4/64 to retain the original per-neuron output prefactor. Removing the other neurons changes the model; passing this test does not validate the full-width gradient or trajectory.
- A third case modifies the late subset to contain zero and collinear gate vectors. It is a numerical edge-case fixture, not a learning experiment.

## Files

- transition_rule.py: composite Gauss-Legendre nodes at Gaussian base intervals and gate transition levels; stable SiLU derivatives.
- teacher_h3.py: division-free one-dimensional pure cubic teacher identities.
- student_transition.py: conditional one/two-dimensional student moments; explicit residual covariance.
- probe.py: server-only fixed-state probe, no parameter updates.
- PROBE_MANIFEST.json: exact input/source hashes and bounds.
- STUDENT_SOURCE_REVIEW.md: independent formula/API review.
- PROBE_REVIEW.json: exact-manifest launch review, when ready.
- TEACHER_README.md and test_scalar_rule.py: scalar method details and test.

All original engine and saved-state bytes are copied unchanged. Local checks compile source or test scalar mathematics only. No local models or training are run.

## Numerical limitations

Integration omits Gaussian tails outside the stated limit without weight normalization. This is an approximation, not a rigorous error certificate. Near-collinear gate residuals below 64 machine eps times the larger gate norm are dropped and the largest dropped residual is reported. Ordered pair quadratures need not be exactly symmetric: the probe reports Gram and p11 asymmetry. Analytical population gradients need not exactly differentiate a finite parameter-dependent quadrature.

## Planned bounded server check

Exactly 33 evaluate calls if complete: three cases, orders8 and12, one base and four central-difference evaluations at each order; plus one limit10 base per case compared with limit12 at order12. One CPU, single-thread libraries, nice10, 170-second soft deadline and external175-second TERM/180-second kill. Initial reference test is deliberately small. Outcomes and any capped calls must be retained.

Record finite-difference slope mismatch (screen <=0.001), order refinement, pair asymmetry, tail sensitivity, unchanged input hashes, and all calls. A successful subset diagnostic only permits considering full-width validation; it does not authorize a training sweep. The broad scientific goal remains incomplete.
