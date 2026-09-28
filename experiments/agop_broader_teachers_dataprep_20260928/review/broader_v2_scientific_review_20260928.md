# Broader-teacher v2: scientific configuration and preflight review

2026-09-28, codex/review-1hl/agop-checkpoint-review.

**Static verdict: no scientific configuration or teacher-factory blocker found. Numerical preflight remains mandatory on the server before training.** This review has not run the scientific checks locally, trained any model, or changed a numerical kernel, config, manifest, previous pilot, experiment record, or paper.

## Actual configuration review

The new package contains exactly 28 distinct arms and matching config files:

- ReLU: `h3`, `damped_2_3`, `three_atom`, `h3_plus_h5`, `h3_plus_sine`, `h2`, `h4`, `sine`, `relu`, `abs` × seeds 501/502. All use rank 8, dimension 64, width 128, initialization scale 1e-4, force step 0.05 and 20,000 intended updates.
- SwiGLU: `he3`, `he2`, `relu`, `tanh` × seeds 501/502. All use rank 8, dimension 64, width 64, initialization scale 0.05 and eight unit teacher coefficients. Remaining non-link/non-seed arguments match across these eight arms.

The ReLU runner differs from the original `W/code/scaling_run.py` only in its teacher-name allowlist and corresponding error message. All five numerical/support modules are byte-identical to the original copies. All six SwiGLU Python modules are likewise unchanged. Each source parses, and every config file equals its manifest entry. The old eight-arm `followup_width_scale_v1/` experiment remains separate and unchanged.

The actual SwiGLU training parser is `swsmall_periodic.make_link`; diagnostic construction uses `swsmall.make_link`. Both support the four declared links with matching definitions, including normalized Hermites and the ReLU kink at zero. `run_one.validate_jobs` introduces no link whitelist restriction. The separate unnormalized Hermites in `vlab.ACTS` are not the teacher construction path here.

## Scientific qualifications

The three ReLU links `h3`, `damped_2_3`, and `three_atom` are exact members of the central positive Gaussian-cubic mixture teacher family. The remaining seven are exploratory controls. The 0.1 perturbations are not certified by the small-residual theorem, and the accessible pilot settings do not satisfy the theorem's complete sufficient parameter prescription.

Every declared additive nonlinear teacher has true input rank eight. Hermite degree is not input rank; in particular the additive h2 teacher is radial within its eight-dimensional teacher subspace. Its individual axes need not be identifiable. Existing product teachers remain rank-two implementations; none is relabeled as rank eight in this manifest.

ReLU training preserves unit **second moment** and the target mean. In particular ReLU and abs targets can be dominated by mean/linear energy as rank grows; sine has about 85.1% linear energy. Their legacy absolute refit-gain threshold may be ineligible given the initial comparator. Retain continuous refit gains, mean/linear/nonlinear loss decomposition, and same-run same-checkpoint timing relative to each run's own original plateau. Do not redefine the plateau from geometry or count comparator ineligibility as evidence against feature learning.

SwiGLU profiles an intercept and reports loss divided by the teacher variance. This makes its nonzero-mean ReLU target's loss convention different from the uncentered raw-MSE ReLU student arm. The two architectures' raw clocks, plateau rules and refit budgets also remain different; do not pool their event counts as if they were matched procedures.

## New server-only preflight

Created `broader_teachers_v2/checks/server_preflight.py`. Its CLI is:

    python checks/server_preflight.py --out NEW_RESULT.json --seconds 90

The root defaults to the package containing the script; `--root` can override it. The output must not already exist. Both the CLI and the scientific-entry function require Linux and `AGOP_EXECUTION_SITE=SERVER` before NumPy, SciPy or model imports. Top-level imports are standard-library only; bytecode writing is disabled and numerical threading is capped at one before scientific imports.

The preflight requires finalized nonempty manifest source hashes, including its own file and all twelve relevant engine/source files. It independently pins the five unmodified ReLU numerical/support kernel hashes, verifies every declared source/config, checks the exact 28-arm teacher/seed design and calls the production config validators. It rechecks source/config bytes before returning success. A pass requires exit code zero, JSON status `PASS`, and `inputs_unchanged=true`.

Scientific checks are fixed-state diagnostics, never training:

1. Independent scalar formulas and one-dimensional Gaussian quadrature verify teacher norms, means, second moments, the first Hermite component, and positive derivative variance. Pointwise teacher values and reference axes are also checked.
2. A bank of four fixed finite neurons in d=64 checks every one of the ten ReLU teacher links at rank 8, and `h3`, `three_atom`, `h3_plus_h5`, `h2`, `relu`, `abs` at rank 16. These rank-16 checks are kernel checks, not extra training arms.
3. Central finite differences separately verify each neuron's teacher cross-moment derivatives with respect to W, b, and their joint direction. Two declared finite-difference steps, 1e-4 and 2.5e-5, must both satisfy absolute tolerance 2e-8 plus relative tolerance 2e-5; every measured error and scale is retained.
4. Full population MSE directional derivatives are checked separately for A, W, b, and a joint direction against `-(2/m) * loss_force` contracted with that direction. This verifies the force-to-gradient factor used by the averaged-output model.
5. The alpha=1 linear-student identities check cross moments and derivatives against independently obtained target mean/first-Hermite coefficients. This is the same identity used to form the production nonlinear-loss decomposition.
6. SwiGLU checks only the four scalar teacher factories and their normalization at 12 versus 24 nodes per integration interval, comparing the actual training and diagnostic parsers. No full SwiGLU feature bank, loss/gradient, AGOP, refit or trajectory is evaluated.
7. The existing bivariate `quadratic_product` factory must reject an attempted rank-eight label.

The four-neuron state has moderate amplitudes so finite differences of full MSE are not erased by subtraction from the unit teacher energy. It is a fixed diagnostic state, not a production initialization. This check cannot establish numerical accuracy for every near-collinear or late-training state, high-probability feature recovery, or sufficiency of the run horizon. Independent scalar quadrature on [-12,12] is a consistency check, not a formal quadrature certificate.

The nominal preflight budget defaults to 90 seconds and is capped at 180. SIGALRM is a userspace alarm and can be delayed by a native numerical routine. The parent launcher should impose its own process deadline with cleanup margin and treat any failure, timeout, missing report, hash mismatch, or non-PASS report as a batch launch blocker.

## Runtime qualifications for the supervising task

With 28 arms, four workers and 480 seconds per arm, the sum of full per-arm allocations divided by worker count is 3,360 seconds, exceeding the 3,300-second global limit before overhead. Global censoring is therefore possible even if every arm remains within its individual limit. Keep the global cap and account explicitly for unfinished/unstarted arms.

The unchanged SwiGLU driver reserves only `min(diagnostic_seconds,60)+10` near the global deadline. To aim for a full 140-second diagnostic allocation, the external wrapper must independently limit the training wall allocation; merely passing `--diagnostic-seconds 140` does not reserve all 140 seconds near a deadline.

## Review provenance

This reviewer and an independent static reviewer checked the allowlist-only delta, config design and teacher paths. Only standard-library AST/hash/config operations were run locally. The preflight's numerical outcome will be added after the coordinating task executes it remotely against finalized sources. Static approval is not numerical preflight approval.

Final statically reviewed preflight source SHA256: `01c5700cd67eabaa0ea7063d50119b5f18b068d3b8aa661c08e4cd8dcaa31948`. Independent code and mathematical rereview found no blocker. In particular, the derivative checks establish internal consistency of cross moments and full loss/forces; they do not independently establish every nonlinear cross-moment value, production-width accuracy, or accuracy of learned AGOP/refit quantities. The numerical preflight is still pending server execution.
