# Targeted SwiGLU follow-up, v8

This30-run paired development study follows the completed v7 screen. Existing data and all failed/censored outcomes remain unchanged. It is not a preregistered independent confirmation and it does not change the original success criterion.

## Fixed grid

Rank8,dimension64,width64; reused Gaussian seeds641/642; uncentered raw additive targets normalized to E[Y²]=1. Every learner profiles its output intercept. All gate,value and output parameters train. Every arm starts from initialization; these are not resumed trajectories.

All30 arms use declared flow horizon3000,15000update cap and loss-stop0.01Var(Y), compared with600/5000/0.1Var(Y) in v7. Relative Euler cap0.01,dtmax50 and all four quadrature orders remain unchanged.

- Eight long-stop arms: softplus,tanh,erf,sine, scale0.1 and head multiplier0.01. The first three v7 loss thresholds exceeded nonlinear teacher energy; the sine horizon/seed disparity also needs a longer observation.
- Twelve scale arms: h3,h4,h5 at scales0.3 and0.5, head multiplier0.01. These compare a stronger higher-degree signal while retaining actual raw initial loss, with no rescaling of the plotted training loss.
- Two h3 long-controls: scale0.1, head multiplier0.01, isolating the horizon/stop change against v7 before attributing effects to scale.
- Six slower-head arms: h2,absolute value,Gaussian RBF at scale0.1 and head multiplier0.001, testing whether additional feature-learning time precedes loss release in both paired seeds.
- Two h3 scale0.3/head0.001 arms compare directly with the scale0.3/head0.01 arms.

No Gaussian seed is screened out, and no teacher is silently centered. Source mathematics is byte-identical to the v7 frozen bundle. CONFIG_COMPARISONS.json enumerates all changes. This is an experiment search; high-probability success and general-r,d,m claims are not supported by two reused seeds at one size.

## Compute and evidence

Require32effectiveCPUs,24GiBavailableRAM,10GiBfree disk. At most30singlethreadworkers, leaving2free. Each arm710s training,180s diagnostics,10s cleanup; global1100s plus separately bounded120s setup. A wall cap is censoring. User's≤2%weeklyallowance checkpoint/pause instruction takes precedence.

Keep the exact initial1%/5% all-update max/min loss windows, same-checkpoint minimum-alignment gain≥0.5, cutoff-sensitivity refit gain≥0.1Var(Y), unchanged numerical guards, and later loss decrease≥0.1Var(Y). The SwiGLU cutoff envelope is not a formal optimal-risk bound. Sparse diagnostics and unchecked doubled quadrature stay explicit. Promising states require separate numerical audits; no integration certificate is implied.

Also apply the existing secondary loss-only later-window rule without replacing initial-window failures. Select longest windows in the predetermined variance/nonlinear-energy bands, independent of alignment/refit; reuse neither optimized window endpoints nor selected seeds. All outcomes stay in the report.

Readout feasibility and nonlinear residual size are reported. For softplus/tanh/erf, even perfect reduction of a small nonlinear residual need not give a0.1Var improvement relative to a already-good refit comparator; do not lower that primary threshold after seeing outcomes. Report raw and variance-relative continuous effects as secondary measurements.

## Preservation

New named remote run and output directory only. No old source/data/plots/manuscripts are edited. Publish this folder to the shared compute repository before remote execution. Resume means observe the same live process; a failed observation is not permission to relaunch.
