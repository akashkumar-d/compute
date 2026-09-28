# Coupled Gaussian-head scaling: focused cubic comparison

This is a proposed six-arm development comparison, not a successful result or a launchable training bundle. It preserves the raw additive cubic teacher, Gaussian initialization shapes, student architecture and the original success criteria. No manuscript is changed. The numerical trajectory validation prepared in v9 remains the first compute priority.

The v8 cubic trials changed inner initialization scale or the head learning rate. Larger inner scale gave eventual all-direction recovery after substantial loss reduction; merely slowing the head with unchanged head initialization did not repair that timing. The present comparison changes both the initial head amplitude and its learning rate in a coupled way. It tests a distinct mechanism, with failure allowed and retained.

Let each centered feature be psi_j = feature_j - E[feature_j], let t_j = E[(Y-E[Y]) psi_j], and let G_jk = E[psi_j psi_k]. The existing fitted intercept gives the loss

    L(theta,a) = Var(Y) - 2(alpha/m) a^T t + (alpha/m)^2 a^T G a.

At a fixed representation and rescaled head b, write a=q b. Set the head learning-rate multiplier to q^2 lambda and use the normalized clock tau=q t_physical. The exact continuous-time block-rate equations are

    d theta/d tau = 2 alpha grad_theta(b^T t)
                     - q (alpha^2/m) grad_theta(b^T G b),
    d b/d tau = 2 alpha lambda [t - q(alpha/m) G b].

Thus q reduces the model's self-interaction relative to its teacher-correlation drift in normalized time. It also reduces prediction amplitude at a fixed (theta,b). This is a change of dynamics, not an exact equivalence to the q=1 trajectory. Feature-refit risk depends on the features and is independent of the current trained head. The AGOP at a fixed (theta,b) scales by q^2, so its eigenspaces do not change under this head scaling alone. Any claimed AGOP alignment gain must come from actual dynamics of (theta,b), not the common q^2 multiplier. Relative changes in b can affect AGOP even at fixed hidden features; a material feature-refit gain is checked separately to establish improved predictive features.

The exact profiled loss identity still holds for nonzero teacher means: centering here describes the fitted output intercept already in the learner, not a modification of the raw teacher. The proposed pilot uses h3, whose mean is zero. Inner weights and biases continue training.

## Fixed comparisons

- Raw additive h3; r=8, d=64, m=64; inner scale s=0.3; reused seeds641/642.
- Initial head ratio q in {1,0.3,0.1}; head multiplier0.01*q^2.
- All-order quadrature tuple(32,96,48,24), unchanged for all six arms. The q=1 controls are rerun at that same order; low-order historical controls cannot isolate this comparison.
- Physical horizon3000/q and dt_max50/q; checkpoint time floor0.5/q. The relative update cap h=0.01 and maximum15000updates remain unchanged. Loss stop0.01Var(Y) remains unchanged.
- Record and plot both physical time t and normalized time tau=q*t. Report raw loss and loss/Var(Y); never divide loss by q or redefine the plateau to make curves look flatter.

The adaptive cap uses the full per-neuron parameter norm including the output head. Scaling a changes that denominator and the capped direction; therefore the discrete numerical trajectories are not exact rescalings even when dt_max is adjusted. This comparison is not an Euler step-size-convergence test. It makes no claim of a longer plateau merely because the physical clock is expanded.

## What would count as evidence

Use the same all-update initial1% and5% max/min loss windows and the same-checkpoint requirements: minimum AGOP alignment gain>=0.5, numerical cutoff-envelope refit improvement>=0.1Var(Y), and a subsequent loss decrease>=0.1Var(Y). Keep absolute effects and all numerical/censoring flags. The criterion concerns the minimum over teacher directions; a top-one or mean-alignment gain alone cannot replace it.

Any promising new trajectory needs higher-order checks of the actual block-weighted updates and relevant AGOP/refit states, followed by full trajectory order comparison if those checks reveal sensitivity. The v9 finding shows why unweighted gradient error is insufficient. A finite-order screen is not a population error certificate. Two reused seeds are not independent confirmation.

Smaller q need not work. It may permit a fast neuron to grow further before output feedback slows it, increasing rather than repairing imbalance among teacher directions. It may also make later loss release slower, worsen quadrature accuracy at large gate norms, or leave the weakest directions unlearned. All such outcomes are part of the experiment. The fixed six configurations and their changes are recorded in DESIGN.json; no result is inferred from this scaling argument.
