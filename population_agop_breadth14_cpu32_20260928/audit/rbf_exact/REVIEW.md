# Exact Gaussian-RBF check at the original ReLU candidate

**PASS.** The reviewed states are initialization and step 250 of `relu_gaussian_rbf_r8_d64_m256_seed641`. Step 250 is the original first numerically qualified candidate inside the initial 1% loss window in the frozen partial summary. No alternative state was selected, no training was run, and all input hashes remain unchanged.

## Exact cross kernel

For standard Gaussian X and a teacher coordinate i,

    exp(-X_i²/2) phi_d(X) = (1/sqrt(2)) phi_{Sigma_i}(X),
    Sigma_i = I - (1/2) e_i e_i^T.

The density identity follows directly by combining the exponents: coordinate i then has variance 1/2, while the normalizing mass is 1/sqrt(2). Thus, with s_i² = W^T Sigma_i W = ||W||²-W_i²/2 and beta_i=b/s_i,

    E[exp(-X_i²/2) ReLU(W·X+b)]
      = (1/sqrt(2)) [s_i phi(beta_i)+b Phi(beta_i)].

Differentiating this closed form gives

    d/db = (1/sqrt(2)) Phi(beta_i),
    grad_W = (1/sqrt(2)) phi(beta_i) Sigma_i W / s_i.

For a leaky student with leak alpha, add alpha*b to the mean term and multiply the ReLU part by 1-alpha; the bias derivative adds alpha. The linear spatial derivative vanishes because the tilted Gaussian has zero mean. Summing these terms with the original `lam_i/norm` produces the exact normalized additive teacher cross kernel. This audit implements it only as a read-only override; production files were not changed.

## Saved-state results

At both states, the frozen order-16 and doubled order-32 kernels agree with the exact formula to at most 2.61e-18 in cross moments and 3.34e-16 in cross gradients. Exact-kernel raw population losses and feasible refit risks reproduce the saved values to the displayed precision. Recomputed AGOP screens remain valid.

| Quantity | Initial | Original candidate, step 250 |
|---|---:|---:|
| Raw profiled-intercept MSE | 0.01897071895818203 | 0.018970298399180185 |
| Minimum AGOP principal-angle alignment | 0.000802758365858 | 0.525539118721322 |
| Same-budget feasible refit MSE | 0.016380727909558468 | 0.007042638667535739 |
| Exact-kernel numerical refit lower bound | 0.016380727841311615 | 0.007042638578001248 |

The original all-update loss max/min ratio through step 250 is 1.0000221693403548. The alignment gain is 0.5247363603554636. Refit improvement using the independently recomputed initial numerical lower bound and candidate feasible risk is at least 0.009338089173775876, or 0.4922369476578137 times target variance. The unchanged material threshold is 0.0018970719727986363. **The original candidate still passes.**

The target second moment is one, but its mean is approximately 0.99046922 and variance is 0.018970719727986363. This learner explicitly profiles its output intercept; its plateau loss should therefore be described relative to residual target variance, not as raw loss near one.

## Scope and reproducibility

`audit_saved.py` reads the frozen source, source receipt, original summary, original two saved states, original diagnostics and full saved scalar loss history. `AUDIT.json` records exact errors, input hashes and checks. It used one numerical-library thread, 9.83 seconds wall time and 4.79 seconds CPU time, with a 60-second audit limit. There were no Monte Carlo evaluations, training updates, or network calls.

This supports the numerical interpretation of these two states and the original candidate. It is not an exact reconstruction of every training gradient, a proof of the full trajectory, or a formal floating-point/population certificate. The all-update loss ratio uses the original stored loss history; only its initial and candidate endpoints were independently recomputed here.
