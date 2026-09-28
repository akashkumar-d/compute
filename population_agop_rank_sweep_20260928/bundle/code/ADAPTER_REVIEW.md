# ReLU canonical fourteen-link adapter

Status: **implemented and numerically verified, ready for bounded exploratory runs**. No training was run for this adapter review. This is a new source copy; all earlier cohorts, scientific kernels, plots, and manuscripts are preserved.

## Exact target definitions

The fourteen raw scalar links are the normalized probabilists' Hermites `h2`, `h3`, `h4`, `h5`; `relu(z)=max(z,0)`; `leaky_relu(z)=max(z,0)-0.1 max(-z,0)`; `abs(z)`; `softplus(z)=log(1+exp(z))`; `silu(z)=z sigmoid(z)`; exact `gelu(z)=z Phi(z)`; `sine(z)=sin(z)`; `tanh(z)`; `erf(z)`; and `gaussian_rbf(z)=exp(-z²/2)`.

For orthogonal coordinate directions and equal coefficients, every target is

    Y = [sum_{i=1}^r q(X_i)] / [sqrt(r) sqrt(E[q(Z)^2] + (r-1) E[q(Z)]^2)].

Thus E[Y²]=1 and the raw mean is retained. `canonical14_moments.json` is a fixed copy of the SwiGLU registry's scalar moments, independently checked here by full-line adaptive Gaussian integration. A profiled intercept is a separately declared learner option; normalization does not silently center the teacher. All these functions belong to H¹(γ); that regularity assertion does not claim favorable dynamics for every function.

## What changed

- Existing exact paths for h2, h3, h4, sine, ReLU, absolute value, and the two existing cubic mixtures are bitwise unchanged at the verification states.
- h5 uses the existing exact Hermite formula. Leaky-ReLU teachers use the signed combination ReLU(z)-0.1 ReLU(-z) in the existing exact Gaussian kernel.
- Six additional smooth links use `teachers_smooth.py`. Their target normalization is fixed independently of the integration order. They support both ReLU and leaky-ReLU cross moments; the unchanged experiment runner still trains ReLU students only.
- The new runner allowlist covers all fourteen. Provenance includes the smooth adapter, scalar-moment JSON, and diagnostic scheduling helper. The copied runner locates its kernel hashes relative to its own code directory, so it can be placed in a server bundle's `code/` directory.
- The optional `diagnostic_priority_fractions` field affects only diagnostic evaluation order. It does not change a training state, remove a checkpoint, change a stopping rule, or choose points from feature values. Omission reproduces the old scheduling order exactly.

## Smooth-link numerical method

For one teacher coordinate Z and student preactivation S=W·X+b, let n=||W||, beta=b/n, rho=W_i/n and v=||W with coordinate i removed||/n. The direct residual norm is used instead of subtracting nearly equal squared norms.

Conditioning on Z=z gives

    H(z) = n [alpha (beta+rho z)
              + (1-alpha){v phi((beta+rho z)/v)
                           +(beta+rho z) Phi((beta+rho z)/v)}],
    p(z) = alpha + (1-alpha) Phi((beta+rho z)/v).

The v=0 branch uses the exact hard-threshold limit. Integrating q(z)H(z) and q(z)p(z) gives the cross moment and bias derivative. Spatial gradients use Gaussian integration by parts:

    E[X q(Z) sigma'(S)]
      = u E[q'(Z) sigma'(S)]
        + W (1-alpha) phi(beta)/n E[q(Z) | W·X=-b].

This avoids a division by the residual standard deviation in the gradient. The remaining conditional expectation uses q(-rho beta+vN), which is smooth even at exact parallelism. Alpha=1 uses the scalar moments directly and is exact up to their recorded floating-point values, including the linear-energy diagnostic.

Integration is deterministic, with 16-point Gauss-Legendre rules on subintervals of [-12,12]. Fixed cuts are -12,-8,-6,-4,-2,0,2,4,6,8,12; additional cuts follow the conditional threshold at offsets -12,-8,-4,-2,-1,0,1,2,4,8,12 residual standard deviations. They are clipped and sorted; zero-length intervals carry zero weight. These moving cuts resolve narrow transitions as a neuron aligns. Conditional expectations use the fixed piecewise rule. This is not unsafe fixed-node integration across an unresolved moving kink.

The adapter is deterministic numerical population integration, not an exact symbolic kernel and not a Monte Carlo training estimate. Finite-domain and quadrature errors remain numerical errors; the review does not provide a formal uniform error theorem. At the tested scales/order and near-collinear cases they are much smaller than the experiment's stopping/refit tolerances. Selected saved-state rechecks at doubled order remain appropriate before reporting new positives.

## Verification

Run `verify_adapter.py` with one numerical-library thread. It performs no training. `VERIFICATION.json` records all cases and source hashes; `VERIFICATION_SUMMARY.json` provides maxima.

- Independent full-line moments for all fourteen match the shared scalar registry.
- Eight old teacher paths match bitwise for student leaks 0, 0.1, and 1.
- 216 independent conditional-quadrature comparisons cover six new smooth links plus exact h5/leaky additions, leaks 0/0.1/1, parallel and antiparallel directions, residual standard deviations down to 1e-10, zero correlation, and biases from -9 to 7 student standard deviations.
- Finite differences check every teacher's cross gradients and the full MSE gradient with and without profiled intercept. The force-to-MSE-gradient conversion is -2/m.
- Orders 16 and 32 agree at the recorded stress states; homogeneity checks cover scales 1e-9, 1e-4 and 1e4.
- A separate 60-row order-12 stress check is saved, but production remains order 16 for margin.
- Source files compile. Timings are bounded single cross evaluations at r=8,d=64,m=128, not experiment trajectories.

## Diagnostic scheduling option

`diagnostic_priority_fractions=[0.25,0.5,0.75]` keeps the historical boundary order first: initialization, exact 1% last-valid and first-crossing states when present, exact 5% last-valid and first-crossing states when present, and terminal state. It then takes the three fractions of the all-update initial 1% endpoint and the three fractions of the initial 5% endpoint. Each is snapped to the nearest already-saved step inside its prefix; ties go to the earlier step. If the 1% prefix is right-censored, its observed endpoint is used. All other saved states follow in the old ascending order. Metadata and final receipts store the actual scheduling order; incomplete diagnostics remain censored.

`verify_diagnostic_priority.py` checks 300 default-order comparisons, preservation of every saved state, boundary priority, tie handling, and censored prefixes. It performs no training.

## Assembly and limitations

Copy these new code files into the remote bundle's `code/` directory and copy the unchanged v6 `diagnostics/` directory alongside it. Preserve the canonical moments JSON. The launcher must hash all files before running and record the new quadrature method metadata. Do not use the earlier frozen cohort manifest for this code.

The added smooth kernels cost more than the exact Hermite kernels. A bounded screen can therefore stop at different achieved training horizons even with equal wall-time budgets. Censoring is not failure, and tested coverage is not proof of the desired plateau/alignment/refit behavior. The student Gram, stable AGOP diagnostic, refit program, training gradient, Armijo rule, and all earlier data are unchanged.
