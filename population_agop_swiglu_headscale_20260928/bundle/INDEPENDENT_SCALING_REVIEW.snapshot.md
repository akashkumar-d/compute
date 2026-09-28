# Independent review of coupled head scaling

**Verdict: algebraically sound as a proposed comparison, with no positive plateau guarantee.** For fixed positive lambda, use `head_ratio=q`, `head_lr=q^2*lambda`, and normalized time tau=q*t_phys, with q>0. On the normalized clock, the teacher contribution to both inner and normalized-head motion has no explicit q, whereas model self-interaction has a factor q. At a fixed normalized state, centered prediction scales by q and AGOP by q^2. This is a different family of flows, not an exact time reparameterization of the q=1 trajectory.

Scope: independent source/algebra review only. No training, model evaluation, new empirical result, network access, or manuscript change was performed. The motivating observation supplied for this review is that existing repeated-cubic r=8, d=64, m=64 high-scale runs learned all directions only after loss decreased, and lowering head learning rate alone with `head_ratio=1` did not fix that timing. That observation is motivation, not evidence that the proposed family succeeds.

## Source contract

Reviewed source directory, relative to the higher-rank workstream:

`goal_followup_v8/targeted30/bundle/swiglu/code/engine/`

- `swsmall_periodic.py`, SHA256 `4ba79a4401cc89309f431c5e2f7fc54372174e19241f563eaf729366573fe9ae`: rates at lines 93–98; teacher/intercept/init at 104–117; adaptive step at 138–146; checkpoint and stopping rules at 161–180.
- `swpop.py`, SHA256 `fa11dd787587be30b8427f6638ee71b164104e159f78b1fe81c19504c84714e7`: fixed raw teacher moments at 55–84; profiled loss at 179–206; inner gradients including mean derivatives at 225–242; AGOP at 245–262; frozen-feature refit at 264–269.
- Supporting read-only checks: `canonical_links.py` declares raw normalized Hermite links and fixed moments; `vlab.py:516–525` defines the reported AGOP eigengap as a ratio.

The intended population equations below are exact. The engine evaluates analytic expectation formulas with finite-order quadrature and floating-point arithmetic; the review does not certify those approximations or any finite-step trajectory. At fixed P,V, the stated polynomial dependence on a is also explicit in the implemented arithmetic, up to floating-point error.

## Exact centered equations

Write theta=(P,V) for all inner parameters, including gate and value biases. Use beta for the scalar output intercept, to avoid confusing it with the normalized head b. Define

\[
\phi_j(x;\theta)=\operatorname{SiLU}(P_j^T\widetilde x)\,V_j^T\widetilde x,
\quad \widetilde x=(x,1),\quad
\mu=\mathbb E\phi,\quad \psi=\phi-\mu,
\]
\[
y_c=Y-\mathbb EY,\quad V_Y=\operatorname{Var}(Y),\quad
\mathbf t(\theta)=\mathbb E[y_c\psi],\quad
G(\theta)=\mathbb E[\psi\psi^T],\quad k=\alpha/m.
\]

Bold t is the teacher-feature moment, not physical time. Profiling the intercept gives

\[
\beta_*(\theta,a)=\mathbb EY-k a^T\mu(\theta),\qquad
f(x)=\mathbb EY+k a^T\psi(x),
\]
\[
L(\theta,a)=V_Y-2k a^T\mathbf t+k^2a^TG a.
\]

The source trains with the **raw** loss L: the reported division by `T.V` is only a diagnostic. There is no extra 1/V_Y factor in the update. Rates are m for both inner blocks and m*head_lr for the head.

Set a=q*b and head_lr=q^2*lambda, keeping q>0 and lambda>0 constant within an arm. Let a prime denote d/dtau with tau=q*t_phys. Direct chain rule gives

\[
\boxed{\theta'=2\alpha\nabla_\theta(b^T\mathbf t)
 -\frac{\alpha^2q}{m}\nabla_\theta(b^TG b)},
\]
\[
\boxed{b'=2\alpha\lambda\left(\mathbf t-
 \frac{\alpha q}{m}G b\right)}.
\]

In particular, da/dt_phys=q^2*b', whereas dtheta/dt_phys=q*theta'. Forgetting either of these two q factors in the normalized-head change would give the wrong coupling.

An equivalent residual form, with h_theta,b(x)=k*b^T*psi(x) and R=y_c-q*h_theta,b, is

\[
\theta_j'=2\alpha b_j\,\mathbb E[R\,\partial_{\theta_j}\psi_j],
\qquad b_j'=2\alpha\lambda\,\mathbb E[R\psi_j].
\]

This makes the comparison's rationale precise: the teacher terms remain at order one on the normalized clock, while prediction feedback enters with q. Lowering `head_lr` alone at `head_ratio=1` retains order-one initial prediction and model feedback in the inner flow. It therefore does not implement this experiment.

The normalized equations are block gradient flow of

\[
J_q=(L-V_Y)/q=-2k b^T\mathbf t+q k^2b^TG b
\]

with rates m and m*lambda in theta and b. Accordingly,

\[
\frac{dL}{d\tau}=-\frac{q}{m}\left(
 \|\theta'\|^2+\frac{\|b'\|^2}{\lambda}\right).
\]

This identity applies to exact population continuous flow. It explains a possible small loss change over an interval of bounded normalized motion, but does not bound the normalized motion or ensure useful feature learning.

## What scales, and what remains invariant at a fixed normalized state

Define C=E[y_c*h_theta,b]=k*b^T*t and H=E[h_theta,b^2]=k^2*b^T*G*b. Then

\[
f_q-\mathbb EY=q h_{\theta,b},\quad
\operatorname{Var}(f_q)=q^2H,\quad
\mathbb E[y_c(f_q-\mathbb EY)]=qC,
\]
\[
L_q=V_Y-2qC+q^2H,\qquad
\frac{L_q}{V_Y}=1-\frac{2qC}{V_Y}+\frac{q^2H}{V_Y}.
\]

Thus loss does not scale simply by q or q^2. For uniformly bounded C,H over a common normalized-time interval it differs from V_Y by O(q). Such bounds are conditions, not results of this review. Centered output scales by q; the full output retains its fixed teacher mean. For the canonical pure cubic teacher the fixed raw mean is zero and V_Y=1, but the derivation also handles nonzero means.

Since centering changes no input derivative,

\[
M_q=\mathbb E[\nabla_x f_q\nabla_x f_q^T]=q^2M_1
\]

at fixed theta,b. Hence:

| Quantity | Fixed-state effect of q>0 |
|---|---|
| Raw feature moments mu, t, G | Unchanged |
| Centered prediction and its input gradient | Multiply by q |
| AGOP, its eigenvalues, trace, absolute spectral gap | Multiply by q^2 |
| AGOP top-r subspace and alignment | Unchanged when the relevant subspace is uniquely defined |
| Reported gap lambda_r/lambda_(r+1) | Unchanged in exact arithmetic |
| Frozen-feature head/intercept refit | Unchanged: source refit uses only G,t,V_Y |
| P/V directions, masses, specialization counts | Unchanged |
| Output-weighted P alignment | Unchanged, because every weight acquires the same positive factor q |
| Total neuron norms and the norm-defined winner | Generally changed: these include q^2*b_j^2 |
| Profiled intercept beta_* | Generally changed |

A spectral degeneracy remains a degeneracy; rescaling cannot make a nonunique subspace meaningful. Very small q also lowers the absolute AGOP signal and may expose floating-point/eigensolver limitations. At q=0 the AGOP is zero and its top-r alignment is not identified; q=0 is only a formal limiting vector field, not an admissible scaled run with these clock and rate definitions.

The frozen-feature refit diagnostic allows arbitrary output coefficients. Consequently its invariance does not say that the actual small-head trajectory attains the refitted loss. At identical theta, the diagnostic is unchanged even when the current network prediction is much smaller.

Across trajectories theta_q(tau),b_q(tau), these fixed-state relations do **not** imply identical AGOP alignment, refit, feature directions, loss histories, or time to learn. The q-dependent feedback changes the path. The teacher-only limiting flow may describe finite intervals only if appropriate boundedness and regularity hold; no such theorem is claimed here.

## Initialization, biases, and balancedness

Keep d,m,s,alpha, seed, teacher link/coefficients, and all quadrature settings matched. The actual source draws P then V then a from the same generator. Changing only `head_ratio` and `head_lr` leaves the sampled P,V arrays identical; with `head_ratio=q`, b(0)=a(0)/q has the same s-scaled Gaussian draws as the q=1 head (up to floating-point representation). The inner marginal variances remain s^2/(d+1), including each bias. The raw head variance becomes q^2*s^2.

This preserves Gaussian marginal shape, independence structure, head signs and relative coordinates, and the normalized initial state. It does not preserve the complete raw parameter distribution or ordinary head/value balance. In particular, the full parameter vector is not undergoing a common isotropic rescaling.

For every augmented inner coordinate, centering must be differentiated:

\[
\partial_\theta\mathbf t
=\partial_\theta\mathbb E[Y\phi]
 -\mathbb EY\,\partial_\theta\mu,
\quad
\partial_\theta G
=\partial_\theta\mathbb E[\phi\phi^T]
 -\partial_\theta(\mu\mu^T).
\]

The source implements both mean corrections in `swpop.py:229–237`. Gate-bias and value-bias feature derivatives are respectively SiLU'(P_j^T*x_tilde)*(V_j^T*x_tilde) and SiLU(P_j^T*x_tilde); they are not zero, and their centered versions subtract their expectations. Since E[R]=0 under exact intercept profiling, the residual expression may equivalently use uncentered feature derivatives. That equivalence does not license omitting the mean corrections from a moment-based calculation.

The scalar output intercept is profiled, not independently trained. Its total normalized-time derivative is

\[
\beta_*'=-kq\left(b'^T\mu+b^T D_\theta\mu[\theta']\right).
\]

There is no separate intercept learning-rate choice. The envelope theorem removes an extra d(beta_*)/dtheta term from the profiled-loss gradient, because the raw loss is stationary with respect to the intercept at beta_*. It does not remove the feature-centering derivatives already displayed above.

The exact continuous-flow per-neuron invariant, including the value bias, is

\[
\frac{a_j^2}{\mathrm{head\_lr}}-\|V_j\|^2
=\frac{b_j^2}{\lambda}-\|V_j\|^2.
\]

It follows from linear homogeneity of the feature in the entire V_j block: V_j dot grad_(V_j)L = a_j*partial_(a_j)L. The initial invariant distribution is q-independent at fixed lambda; its expectation is s^2*(1/lambda-1), zero only when lambda=1. There is no corresponding generic gate/head invariant because SiLU is not homogeneous. The source correctly marks `a^2-sum(V^2)` as legacy, non-invariant when head_lr differs from one. Adaptive Euler and quadrature error can produce drift even in the weighted quantity, so its recorded drift is a numerical diagnostic, not an exactly preserved discrete law.

## Clock, step cap, and horizon matching

To target common normalized values D_tau, T_tau and C_tau, the configuration conversion is

| Setting | Scaled arm |
|---|---|
| `head_ratio` | q |
| `head_lr` | q^2*lambda |
| `dt_max` | D_tau/q |
| `t_max` | T_tau/q |
| `cp_min` | C_tau/q, if matching the minimum clock-based checkpoint spacing |
| `h`, `cp_ratio` | Same declared values; this does not ensure the same realized step/checkpoint grid |

Without the 1/q horizon conversion, a smaller-q arm is stopped at a shorter normalized time. Without the 1/q `dt_max` conversion, it uses a smaller normalized step ceiling. Preserve raw physical times and also report tau=q*t_phys; retain loss-triggered checkpoints and all censoring records.

The implemented adaptive cap depends on raw head magnitude. To see the obstruction to exact step matching, define for neuron j

\[
A_j=\|P_j\|^2+\|V_j\|^2,\qquad
U_j=\|P_j'\|^2+\|V_j'\|^2.
\]

The source cap, expressed on the normalized clock, is

\[
\Delta\tau=
\min\left(q\,\mathrm{dt\_max},
\frac{h}{\max_j\sqrt{U_j+q^2(b_j')^2}/
                         \sqrt{A_j+q^2b_j^2}}\right),
\]

apart from its tiny denominator floors. Both the metric and the vector field depend on q. Scaling `dt_max` by 1/q only matches the first term. Even a hypothetical common normalized vector field would not generally have identical accepted steps under this cap. Tiny heads also mean the cap does not directly control relative changes in normalized b by a q-independent bound.

The update is forward Euler; no exact discrete trajectory rescaling or exact loss monotonicity follows. The final time may overshoot `t_max` by a step because stopping is checked at the next evaluated state. Matching `max_steps` alone does not imply matched attained tau or numerical accuracy, and wall-time caps have no guaranteed 1/q conversion. Record the attained tau, steps, termination reason, quadrature orders, and any unresolved numerical sensitivity; use independent step/order checks before promoting a promising trajectory. Loss-based checkpoint triggers will intentionally differ as q changes.

## Interpretation limits for the proposed experiment

The comparison tests whether reducing prediction feedback relative to normalized teacher-driven motion creates a useful interval of feature change with small loss change. All existing material all-direction alignment, refit, spectral-separation, later-loss-release, and censoring requirements should remain explicit. A small head by itself can make loss nearly flat without learning useful features.

Small q does not guarantee bounded normalized heads or inner features. It may allow the fastest neurons to grow much larger before residual feedback restrains them; teacher-driven nonlinear growth can concentrate on a few directions and can approach a fast-growth or blow-up regime. Then C,H and normalized velocity can grow with 1/q or faster, defeating the fixed-state suppression argument. Biased gates, feature interactions, unequal initial directional correlations, and finite-width competition remain present. The balancedness invariant gives no bound preventing such behavior. Saturation and loss release may also be delayed beyond the observed horizon, so failure to see release must remain unresolved or censored.

**Approved rationale only:** q provides an independent intervention beyond lowering head learning rate alone while keeping the raw teacher and inner Gaussian draws fixed. No algebraic identity here proves an all-direction plateau, a favorable learning order, success at m=64, bounded dynamics, or eventual release. Those remain empirical questions subject to numerical validation.

## Review of the concrete prepared comparison

Read `PROTOCOL.md`, `DESIGN.json`, and all six referenced configuration files. A read-only standard-library check confirmed all six configuration hashes, all six base-configuration hashes, and exact agreement of each declared argument change list with the actual argument changes. All six arms match the intended raw h3 teacher with eight unit coefficients, alpha=1, d=64, m=64, s=0.3, h=0.01, reused seed641 or642, and common quadrature orders `(n_pair,n_diag,n_z,n_x)=(32,96,48,24)`.

The q values 1,0.3,0.1 produce `head_lr` values 0.01,0.0009,0.0001 at fixed lambda=0.01. The files have `head_ratio=q`, `t_max=3000/q`, `dt_max=50/q`, and `cp_min=0.5/q`, with common 15000-step cap and `L_stop=0.01`. These implement the intended scaling. The q=1 controls receive the same quadrature refinement, so historical lower-order trajectories are not being treated as the matched controls. No training or model evaluation was needed for these static checks.

One concrete wording correction was sent to the parent: the initial protocol sentence that any AGOP alignment gain must come from “actual representation dynamics” should say **actual dynamics of (theta,b), not the common q^2 multiplier**. Relative normalized head weights can change the AGOP eigenspace even at fixed theta. The required material frozen-feature refit improvement separately checks change in the feature representation. This nuance does not change any configuration or the scaling rationale.

The protocol correctly retains all-direction, refit, later-release and censoring criteria, warns that runaway by fast neurons may worsen balance among teacher directions, disclaims exact trajectory equivalence, and treats two reused seeds as development evidence. No algebraic/configuration correction is required. The remaining step-size, quadrature, spectral-definition and boundedness limitations above still apply.

Final reviewed status: **prepared only; waiting for compute and a separate concrete bundle/runtime review. The v9 higher-order trajectory check remains the first compute priority. No launch or successful new result is implied.**
