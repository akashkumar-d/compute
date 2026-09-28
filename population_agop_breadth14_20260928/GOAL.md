# Fourteen teacher links for each student

User steering on 2026-09-28: cover at least fourteen teacher links for **both ReLU and SwiGLU**. This extends the active higher-rank experiment goal; it does not claim fourteen positive theorems or fourteen successful configurations.

## Canonical set

The same fourteen scalar functions are used by both student implementations:

1. `h2`: `(z^2-1)/sqrt(2)`.
2. `h3`: `(z^3-3z)/sqrt(6)`.
3. `h4`: `(z^4-6z^2+3)/sqrt(24)`.
4. `h5`: `(z^5-10z^3+15z)/sqrt(120)`.
5. `relu`: `max(z,0)`.
6. `leaky_relu`: `max(z,0)+0.1 min(z,0)`.
7. `abs`: `|z|`.
8. `softplus`: `log(1+exp(z))`, evaluated stably.
9. `silu`: `z/(1+exp(-z))`.
10. `gelu`: `z Phi(z)` (exact Gaussian-CDF GELU).
11. `sine`: `sin(z)`.
12. `tanh`: `tanh(z)`.
13. `erf`: `erf(z)`.
14. `gaussian_rbf`: `exp(-z^2/2)`.

All belong to Gaussian `H^1`. Every polynomial has finite Gaussian derivative energy; the other functions have bounded or at-most-polynomial function and weak-derivative growth. This regularity alone gives no favorable-dynamics guarantee.

These are distinct scalar functions, not fourteen copies obtained by changing centering or multiplying by a constant. ReLU, leaky ReLU and absolute value are algebraically related, and the odd saturating links have related leading Hermite content; this is intentionally not a count of independent mechanisms. Previously studied mixtures and two-index products remain additional evidence outside this canonical list.

## Target and comparison

For each scalar function `q`, use the uncentered additive target

`Y = sum_i q(u_i^T X) / sqrt(r [E[q(Z)^2]+(r-1)E[q(Z)]^2])`,

where `X~N(0,I_d)`, `Z~N(0,1)`, and the columns of `U` are orthonormal. Both implementations must use the same fixed scalar moments, so `E[Y^2]=1`. Nonzero means are retained. A profiled output intercept is an explicitly identified learner, not a plotting normalization.

The first common screen is rank8/dimension64 and two fresh seeds per student/link: 28 cells and 56 runs. Student widths, parameter scales and update rules are declared separately. This is a broad exploratory screen, not an equal-compute student ranking or a high-probability confirmation study. Short horizons, unresolved AGOP/refit diagnostics, failures and unstarted runs remain visible.

## What counts as progress

At one saved checkpoint inside an all-update initial max/min loss ratio of at most1.01 or1.05 (reported separately), require minimum principal-angle alignment gain at least0.5, passing initial/current AGOP screens, and a numerical lower same-representation refit gain of at least0.1 Var(Y). Report a subsequent raw population MSE reduction of at least0.1 Var(Y) from that same candidate. A later loss-only window may be explored separately; it cannot retroactively turn an initial-window failure into success.

Plots report raw MSE, minimum/mean AGOP alignment and refit MSE. Seed medians use common observed support; missing or unresolved diagnostics are not interpolated across. Numerical observations do not constitute population certificates.

## Resources and preservation

The user subsequently authorized a64-core Lightning CPU with four cores left free. The prepared expanded batch uses at most60 single-thread workers after verifying the actual allocation and memory; it refuses to launch at this concurrency on the old4-core machine. The56 common-grid runs and four previously planned pure-sine scale follow-ups can run in one wave. Per-arm cap900s, including180s diagnostics and10s cleanup; global compute cap1200s. No GPU switch or heavy local training. Preserve all older data and every outcome. Current paper sources remain untouched. Save reproducible source/configuration hashes and exact restart commands. At2% or less weekly allowance remaining, checkpoint and pause task-owned work as previously instructed.

The focused pure-sine scale follow-ups occupy the additional four slots under the user's60-core allowance, with their longer resource limits redeclared in the new manifest. They are extra runs and do not inflate fourteen-link coverage. The cubic SwiGLU horizon proposal remains saved; it is not launched automatically or substituted for the fourteen-link coverage requirement.
