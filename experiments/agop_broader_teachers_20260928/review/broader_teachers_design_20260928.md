# Broader higher-rank teacher design review

2026-09-28 — codex/review-1hl/agop-checkpoint-review. Read-only scientific/code review; no model evaluation, training, remote command, manuscript edit, or original-source/data edit. The old `followup_width_scale_v1/` pilot remains unchanged.

## Recommendation

For a first bounded breadth pilot, use the eight implemented additive teachers below at a fixed higher rank, width, initialization convention and matched development seeds. This varies the teacher while retaining the existing exact population kernels. Use a new manifest and copied runner with an explicit expanded teacher allowlist. Keep the prior one-hour global ceiling; teacher breadth should replace, rather than multiply, the old width/scale factorial. Censoring and unstarted arms remain explicit if the budget expires.

The current higher-rank `scaling_run.py:39` accepts only `h3`, `damped_2_3`, and `relu`, although its `Teacher` factory already implements every additive candidate below for arbitrary valid r. Enabling these additional names requires an isolated runner change; merely writing new configs will currently fail validation.

## Eight concrete additive links

Let h_k be the orthonormal probabilists' Hermite polynomial. All eight use orthonormal teacher axes `U=eye(d)[:,:r]`, with `y=(1/sqrt(r))*sum_i q(x_i)`, mean zero and E[y²]=1 after the stated scalar normalization. Each has genuine input rank r, not just Hermite degree r.

| Existing factory name | Exact normalized scalar link q | Role and theorem scope |
|---|---|---|
| `h3` | h3 | Baseline; exact central teacher family |
| `damped_2_3` | g_(2/3) / ||g_(2/3)||_2 | Existing infinite-Hermite baseline; exact central family |
| `three_atom` | (0.2 g_0.4 + 0.3 g_0.7 + 0.5 g_1) / its L2 norm | New positive multiscale mixture; exact central family |
| `h3_plus_h5` | (h3 + 0.1 h5) / sqrt(1.01) | Higher-degree perturbation; exploratory, coefficient 0.1 is not theorem-certified |
| `h3_plus_sine` | (h3 + 0.1 sin) / its L2 norm | Small first-Hermite component plus full nonpolynomial link; exploratory |
| `h2` | (z²−1)/sqrt(2) | Degree-two control; exploratory |
| `h4` | (z⁴−6z²+3)/sqrt(24) | Degree-four control; exploratory |
| `sine` | sin(z) / sqrt((1−exp(−2))/2) | Strong-linear-component control; exploratory, interpret legacy refit threshold separately |

The family generator is exactly

    g_v(z) = v^(-7/2) (z^3 - 3 v z) exp(-(1/v - 1) z^2 / 2).

It is not simply h3 multiplied by a damping envelope: the linear polynomial coefficient changes with v. `h3_plus_sine` has squared normalization

    1 + 0.01*(1-exp(-2))/2 - 0.2*exp(-1/2)/sqrt(6),

because h3 and sine are correlated. Normalizing it by sqrt(1.01) would be incorrect. The existing implementation includes this cross term.

If preserving the absence of degrees 0–2 is more important than a strong-linear control, replace `sine` before launch by `damped_1_3` or `mixture`. Also supported in the central family are `damped_1_2` and `mixture=(g_1+g_(1/3))/2` followed by normalization. The six `MixtureTeacher` names are all exact positive-family links, but neither the accessible pilot nor the earlier illustrations thereby satisfy the theorem's much stronger width/dimension/scale/mesh prescription.

## Rank and normalization cautions

For a generic uncentered additive raw link q with mean mu and second moment s2, `ExtraTeacher` uses

    c_r^2 = s2 + (r-1)*mu^2,
    y = sum_i q(x_i) / (sqrt(r)*c_r),
    E[y] = sqrt(r)*mu/c_r,   E[y^2] = 1.

It retains the mean; it does not silently standardize variance. The teacher AGOP in its r-dimensional reference space is

    [Var(q'(Z))*I + (E[q'(Z)])^2*11^T] / (r*c_r^2).

All recommended nonlinear links have Var(q')>0, hence true rank r. The additive h2 target is radial within U; its coordinate axes are not individually identifiable, although U is. Neuron-axis specialization should therefore not be imposed as its success criterion. A linear sum, or a function of one sum, would have true rank at most one even if r coordinates appeared in its formula; none of the eight recommended links makes that substitution.

`relu` and `abs` are additional supported controls but are best separated from the primary eight. At rank 8, the normalized abs target has about 93.3% of its energy in its mean. The normalized ReLU target has about 94.4% in mean plus linear projection. Their plateau can end through fitting these simple components before nonlinear directional recovery, and their initial bounded refit may already be too good to allow an absolute 0.399 improvement. This is not a failure of the cubic mechanism. Centering these teachers would define new targets and requires new, explicit names and corresponding cross-moment corrections.

Pure sine is mean zero but has linear-energy fraction

    exp(-1) / ((1-exp(-2))/2) ≈ 0.851,

independent of r. It is intentionally a control for easy linear learning. In contrast, `h3_plus_sine` has only about 0.004 of its normalized target energy in the linear projection. Report the exact computed teacher mean and first-Hermite energy for every new arm rather than inferring them from a plot.

## Interaction options

The existing `ProductTeacher` rejects r other than 2. Its six genuine products are `relu_product`, `silu_product`, `gelu_product`, `tanh_product`, `quadratic_product`, and `bilinear_product`. The names `silu_additive` and `silu_centered_additive` are two-coordinate additive controls, not interactions. All are already in the main illustrative engine, but changing their manifest r cannot turn them into higher-rank experiments.

For the normalized product `q(S)*T/||q||_2`, with independent S,T, the target has mean zero, second moment one, and teacher AGOP

    diag(||q'||_2^2 / ||q||_2^2, 1).

Every implemented nonconstant gate has true rank two. A nonzero gate mean creates a linear component along T despite the target's zero mean. Its presence must be retained in the loss decomposition.

The simplest genuine higher-rank extension would use disjoint pairs:

    r=2k,
    y = (1/sqrt(k)) * sum_(i=1)^k h2(x_(2i-1))*x_(2i).

This has mean zero, E[y²]=1, no linear component, and teacher AGOP `blockdiag(diag(2,1),...,diag(2,1))/k`, so it has true rank r. A disjoint bilinear-block sum is a useful degree-two interaction control with teacher AGOP `I_r/k`. These are new adapters, not current higher-rank capabilities. An adapter could reuse the existing rank-two kernel on coordinate-permuted W and sum C, gb, and inverse-permuted gW divided by sqrt(k). It needs bounded server checks of normalization, gradient permutation and directional derivatives before training. Reusing the same two coordinates in every summand would remain rank two.

For the first one-hour breadth pilot, existing additive kernels are the lowest-risk choice. If interactions are included immediately, either label existing products as rank-two controls or validate a separately hashed disjoint-block adapter first. Do not claim a pure product is theorem-covered just because a theorem allows a small interaction residual.

## Proven family versus exploratory cases

The central ReLU/leaky theorem uses a probability measure on v in [1/3,1], positive damping weights, normalized scalar links and equal positive coordinate coefficients. The broader theorem permits coordinate-dependent links and normalized positive coefficients only with comparable optimized signals, `min K_i* >= (5/6) max K_i*`; for a common link the coefficient ratio is at most 6/5. Because these links are odd, an axis sign can absorb a coordinate-coefficient sign. Signed damping measures are still outside the stated family.

The broader theorem's interaction residual must satisfy every explicit small H1 bound in its parameter prescription. An arbitrary product, h2/h4 teacher, sine teacher, or a 0.1 perturbation does not inherit that guarantee. The practical Armijo experiment remains illustrative even for exact-family links; it does not establish the literal fixed-step theorem hypotheses.

## Same-run measurements and launch preparation

Retain the original raw all-update population MSE plateau definition and never select its boundary from geometry. At each saved state retain the full predictor AGOP, minimum/mean principal-angle scores, all principal cosines, teacher-axis captures, leading score, full spectrum/boundary gap/residual/old warning, teacher gradient-energy mass and balanced weakest-direction energy. Report the raw score plus warning; a large leading score or teacher mass cannot certify recovery of all r directions.

Keep the teacher's own spectral shape as an interpretation baseline. For an exact additive teacher the balanced weakest-direction energy is `Var(q')/E[(q')²]`, which is about 0.352 for sine and 0.5 for ReLU, rather than one. Intrinsic teacher anisotropy must not be misreported as missing student directions. The known teacher subspace remains full rank and principal-angle recovery still has its usual meaning.

The runner already records teacher mean, first-Hermite energy, mean-error squared, linear-error squared, residual nonlinear loss and residual nonlinear loss divided by the target nonlinear variance. These are particularly important for sine/ReLU/abs. They supplement raw MSE; they do not redefine the original plateau after seeing results.

Keep bounded refit caps and initial/endpoint numerical lower–upper risk bounds fixed within each arm. Show continuous absolute and relative refit gain. When an initial feasible refit risk is already below the legacy required absolute gain, flag that comparator as ineligible for that event threshold rather than treating a missed threshold as missing representation learning. Preserve legacy threshold outcomes as labeled historical-comparison diagnostics, with no post hoc threshold change.

Report, per seed and at the same checkpoint, first minimum-alignment gain, first refit gain and first joint gain relative to that run's own initialization, and whether each occurs inside or after its own initial plateau. Include all declared arms, censoring and missing diagnostics. Candidate within-plateau events require saved-state numerical checks; separately averaged or pointwise median curves cannot supply the within-run joint event.

Before training, the new package should record the teacher formula/mean/variance/true rank and source hashes, pass structural config checks locally without evaluating a model, and run any numerical smoke/derivative/normalization checks only on the authorized server inside the bounded compute plan. Do not edit the original eight-arm width/scale protocol or infer sufficient compute from an old Studio-status record.

## Source evidence

Inspected `code/population.py`, `teachers_extra.py`, `teachers_product.py`, `scaling_run.py`, `frames.py`, and the main illustrative `population_agop_50seed_release_v1/code/{protocol.py,run.py}`. The higher-rank copies of population, extra teachers, product teachers, refit and frames are byte-identical to that release. Also inspected current arXiv `paper/sections/{setup.tex,theory.tex}`; independent review checked the explicit residual prescription and experiment appendix. No paper was edited.

Scientific-source hashes: population `dfb863648efaccac9f4be3d103c6d920dcb1f59cd9458bff4a000c7dbbbea488`; extra teachers `a0e9d12035a3fbe57b47a292a95a88fbe19355aee8b06ffa75e55a99a4b7cb11`; product teachers `b36f03bbb8f7f3a1b24b420131ec882f95e13da224cdda63c131041e747f25d1`.
