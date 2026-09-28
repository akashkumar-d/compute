# Weak raw links: evidence and the next two-arm comparison

**Recommendation: two new raw-ReLU-teacher SwiGLU runs at rank8, changing only `head_lr: 0.01 → 0.001`, retaining `head_ratio=1` and seeds641/642.** Use the completed v10 rank8/order2 runs as matched controls. This is a bounded test of whether slower readout adaptation allows the weakest feature directions to develop before the initial loss band closes. It is not a prediction of success, and it does not duplicate the six cubic coupled-head-scaling runs currently owned by the parent.

## What the completed evidence changes

The mean is already fitted by the profiled output intercept at initialization. Consequently **rapid mean fitting is not a later dynamical explanation** for these losses. Raw means still matter through normalization: at rank8, raw ReLU has Var(Y)=0.211169264 despite E[Y²]=1. Its linear component occupies one aggregate teacher direction; nonlinear components distinguish all directions. High top-direction capture and a large refit improvement are consistent with fitting that linear component first, but the saved top-direction statistic is projection into the whole teacher subspace, not a direct measurement of projection onto the aggregate linear direction. Do not equate the two.

The fixed-attempt completed rank report adds a useful positive and a persistent timing failure:

| Raw ReLU → SwiGLU rank | Maximum screened ΔA_min inside 5%, seeds641 /642 | 5% endpoint refit gain / Var | Initial joint + later release |
|---:|---:|---:|---:|
| 2 | 0.743954 / 0.0436351 | 0.797036 / 0.813345 | 1/2 |
| 4 | 0.00008738 / 0.0992052 | 0.710040 / 0.759819 | 0/2 |
| 8 | 0.0176665 / 0 | 0.702765 / 0.738368 | 0/2 |
| 16 | 0.00132446 / 0.00003124 | 0.705697 / 0.644476 | 0/2 |

The alignment maxima and refit endpoints above need not be the same checkpoint; they are not combined into new successes. All eight raw-ReLU/SwiGLU initial windows are closed, fully diagnosed, and have no unresolved initial-window rows. Rank2 seed641 has an actual same-state candidate at t=14.30195: loss/Var=0.99929694, ΔA_min=0.53277671, envelope refit gain/Var=0.71108712, followed by the required loss release. It qualifies in both the original1% and5% windows. This is one reused development seed, not a probability estimate or a reason to drop seed642.

Ranks4/8/16 subsequently reach a screened half-gain only after loss/Var falls to approximately0.155–0.208. The rank8 first such saved points are at t=131.57487/143.46133 and loss/Var=0.19936635/0.17572006; their5% windows already ended at t=42.36210/42.43379. More physical time alone cannot repair those closed initial windows. Rank4 seed642 has missing later diagnostics; several runs are wall-censored. Those limits affect eventual-learning claims, not the fully diagnosed initial-window verdicts.

In the same rank study, h2/ReLU passes at all eight rank/seed cells; h2/SwiGLU passes6/8. Raw-ReLU/ReLU passes0/8, with rank8/16 initial numerical ambiguity retained. Thus the weak raw link is a distinct obstacle, not a generic inability of the study to observe the sequence. Different student refit classes prevent a controlled architectural comparison.

## Signal and effect-size budget

The frozen rank8 teacher energies give the following raw budgets. The affine residual fraction is independent of rank under the equal-weight raw normalization, although raw variance and per-axis signal are not.

| Teacher | Var(Y) | Nonlinear residual / Var | Raw nonlinear energy | Required raw refit improvement0.1Var |
|---|---:|---:|---:|---:|
| relu | 0.211169 | 0.266529 | 0.0562827 | 0.0211169 |
| leaky_relu | 0.267218 | 0.195659 | 0.0522837 | 0.0267218 |
| softplus | 0.0496428 | 0.0792389 | 0.00393364 | 0.00496428 |
| silu | 0.478266 | 0.201490 | 0.0963661 | 0.0478266 |
| gelu | 0.351885 | 0.276712 | 0.0973710 | 0.0351885 |
| tanh | 1 | 0.0695301 | 0.0695301 | 0.1 |
| erf | 1 | 0.0864172 | 0.0864172 | 0.1 |
| sine | 1 | 0.149082 | 0.149082 | 0.1 |

Raw ReLU is the best first diagnostic teacher: its nonlinear budget is2.665 times the primary improvement threshold and its degree2 energy alone is0.233471 Var, while it retains the dominant linear component0.733471 Var. At rank8 each axis has nonlinear energy0.00703534 raw, or0.0333161 Var. The favorable h2 mechanism therefore has a substantial analogue here, without subtracting the linear component or changing the teacher.

The completed rank8 initial refit errors are0.96117279/0.96885826 Var, so these saved numerical initializations are not excluded by the0.1Var improvement threshold. By the5% endpoint, refit errors are0.25840770/0.23049068 Var. Compared with the ideal affine residual0.26652890 Var, they improve on affine risk by only0.00812119/0.03603822 Var, although their total initial-reference refit improvements exceed0.7Var. This is an **affine-comparator arithmetic difference, not an identified decomposition of learned features**. It shows why large refit gain alone gives little evidence that every nonlinear direction is learned. The required minimum-direction check remains essential.

Softplus/tanh/erf have less than0.1Var residual after ideal affine fitting; this does not exclude improvement from their actual initial random representation. However, it makes a post-affine reanchoring especially unsuitable. In v8, softplus's first strong all-direction states already have loss0.03966/0.05493 Var, so another0.1Var loss decrease from those late anchors is unavailable under nonnegative MSE. Keep the initial reference and all original thresholds.

## Exact next experiment and interpretation

Clone the completed rank8 raw-ReLU/SwiGLU control configurations, retaining their scientific sources and all listed settings:

| Field | Both new arms |
|---|---|
| Teacher / geometry | raw `relu`, eight unit coefficients, r8,d64,m64, E[Y²]=1, existing profiled intercept |
| Seeds | **641 and642**, both retained |
| Initialization | `s=0.1`, `head_ratio=1` |
| Only dynamical change | `head_lr=0.001` instead of0.01 |
| Integration | `h=0.01`, `dt_max=50`; quadrature32/96/48/24 |
| Observation bounds | `t_max=3000`, `max_steps=20000`, `L_stop=0.01` of variance |
| Saved diagnostics | `cp_ratio=1.06`, `cp_min=0.5`, `dl_ratio=1.5`, priority fractions0.25/0.5/0.75; exact loss boundaries |
| Suggested bounds | at most2 workers;1800s training+180s diagnostics+10s cleanup per arm;2150s global including bounded setup |

These settings use existing knobs and preserve the original learning architecture. Give both new arms distinct output IDs; no source or launcher change is requested here. The controls are `v10_rank_swiglu_relu_r8_d64_m64_seed641/642` from the outcome-independent fixed-attempt combined report. Reuse them only after verifying the source/config lineage and identical initial arrays; if a required numerical correction changes the recipe, rerun both controls as well, producing four matched arms rather than mixing precision settings. The controls reached t≈1042/951 under wall caps; compare curves on common observed support and retain each arm's full per-run verdict.

This intervention leaves the initial features, trained head, AGOP and refit unchanged. At a fixed state the continuous inner gradient is unchanged, while the output-head contribution to the loss derivative is reduced by ten. Subsequent features and heads follow a different trajectory; the adaptive step can also change, so this is not an exact clock rescaling. It tests whether earlier output adaptation closes the loss window before hidden weak directions develop. Head-only slowing does **not** reduce the relative linear-versus-nonlinear teacher force and need not prevent concentration on a few directions. V8 cubic head-only slowing failed to repair its timing, so failure here remains plausible; the proposed comparison is deliberately only two new arms.

Use the unchanged initial whole-update ratios≤1.01/1.05, same-checkpoint ΔA_min≥0.5 and refit gain≥0.1Var, then later loss decrease≥0.1Var. Read the full saved principal-cosine spectrum, teacher-axis captures, relative eigengaps and refit at those exact checkpoints. A longer nearly flat interval with top alignment high but weakest gain still small supports a missing-direction/concentration explanation; it is not a success. Earlier weakest-direction gain meeting the joint test would support the output-timescale hypothesis, subject to numerical validation. No result should be inferred from a rescaled physical clock.

Widening is a reasonable **later** test if this intervention only delays loss while leaving weak directions absent, but it changes finite-width averaging, initial representation/refit and pairwise cost at once. Larger inner scales have already produced late, rather than initial-window, gains in cubic studies. Neither is the cleanest first intervention here. Do not add a width/scale grid, select favorable seeds, change the teacher, or duplicate the ongoing cubic q-scaling pilot. Review that pilot before proposing any separate transfer of coupled head scaling to raw links.

## Provenance and limits

This note uses only saved summaries and scalar-energy arithmetic: [v7/v8 weak-link triage](../weak_teacher_triage/TRIAGE.md), [ReLU weak-link triage](../relu_triage/TRIAGE.md), [fixed-attempt rank report](../rank_retry_a02/analysis/report_a02/REPORT.md), [combined32 summary](../rank_retry_a02/analysis/report_a02/COMBINED32_SUMMARY.json), [frozen teacher energies](../../goal_followup_v7/breadth14/audit/TEACHER_ENERGIES.json), and [ongoing cubic comparison protocol](../headscale/PROTOCOL.md). The combined summary read here has SHA256 `6360be42bc1a07dfe276045ad09059b900193f6e8fe396a258dfaf498902aaeb`; its separate rank2-h2 diagnostic sidecar does not change the raw-ReLU evidence used above.

SwiGLU cutoff min/max envelopes are sensitivity samples, not formal oracle-risk bounds. No order4 check for these raw-ReLU trajectories is claimed. Any promising candidate needs actual block-weighted-update and AGOP/refit order checks, with whole-trajectory comparison if sensitive. Two reused seeds are development evidence. No training, model evaluation, network action, source edit or manuscript edit was performed for this note.
