# Bounded SwiGLU teacher expansion: independent design review

2026-09-28. Author: codex/review-1hl/agop-svd-diagnostics.

This is a read-only source and saved-report review. No training, population-moment evaluation, remote operation, engine modification, or manuscript edit was performed. The proposed cohort is a new exploratory screen, separate from the old 200-run study. Configuration and launch decisions belong to the coordinating task.

## Recommended minimum

Alongside the broader ReLU teacher screen, add **eight SwiGLU arms**: four links (`he3`, `he2`, `relu`, `tanh`) at **r=8, d=64, m=64**, with two fresh, predeclared paired seeds per link. This adds three qualitatively different teacher links plus a fresh cubic control without changing the numerical engine. Use the coordinator's fresh seed pair; do not silently pool these outcomes with seeds 301–310.

| SwiGLU link string | Scalar link before whole-target normalization | Role |
|---|---|---|
| `he3` | (z³−3z)/√6 | Historical cubic baseline on the new server |
| `he2` | (z²−1)/√2 | Even quadratic; different lowest Hermite degree |
| `relu` | max(z,0) | Nonsmooth, nonzero-mean link with degree-one and degree-two components |
| `tanh` | tanh(z) | Smooth bounded nonpolynomial link with a degree-one component |

Retain `alpha=1`, `s=.05`, `head_ratio=1`, `h=.01`, `dt_max=50`, `t_max=1e6`, `L_stop=.5`, `max_steps=1000`, `cp_ratio=1.15`, `cp_min=1`, `dl_ratio=2`, and quadrature orders `n_pair=16`, `n_diag=48`, `n_z=24`, `n_x=12`. Set `c=[1]*8`. These are the historical rank-eight configuration values with link and seed changed. An explicitly declared longer update horizon is a separate design choice; it must not be introduced conditionally after looking for favorable alignments.

This is the smallest proposed screen with a matched baseline, even polynomial, nonsmooth, and smooth nonpolynomial targets and at least two starts per condition. Two seeds are descriptive replication, not a success-probability estimate. If the bounded server budget cannot support eight attempts, preserve unstarted/censored arms and the frozen manifest; do not report the fastest completed subset as the complete comparison. Schedule links in an interleaved order so the global deadline does not systematically omit the later link classes.

`abs` is already available and would add a useful even nonsmooth comparison, but is second priority after this minimum. Mixtures such as `he3+0.3*he2` are also supported; hold their coefficients fixed in advance and label them separately rather than treating them as new independent base-link families. Changing width or scale simultaneously would confound the immediate question about teacher links; the separate ReLU width/scale pilot can address those knobs.

## What the current factory actually supports

Training imports `swsmall_periodic` in [run_one.py](../swiglu/code/run_one.py:109). That engine calls its own `make_link` in [swsmall_periodic.py](../swiglu/code/engine/swsmall_periodic.py:96), whose registry is exactly `he3`, `he2`, `relu`, `abs`, and `tanh`, with sums parsed from strings ([factory](../swiglu/code/engine/swsmall_periodic.py:56)). The posthoc diagnostic factory independently imports `swsmall.make_link` ([diagnostics.py](../swiglu/code/diagnostics.py:78)); that file has the same five-link registry ([swsmall.py](../swiglu/code/engine/swsmall.py:49)). The proposed four choices work through both routes unchanged.

The larger `vlab.ACTS` registry is **not** the factory used by these runs. In particular, `sine`, `silu`, `gelu`, the damped cubic family, and the ReLU engine's `h4`, `h3_plus_h5`, and `h3_plus_sine` are not automatically available in this SwiGLU launcher. The external label `h3` corresponds to the SwiGLU config string `he3`, and ReLU's `h2` corresponds to SwiGLU's `he2`. Do not copy ReLU config strings blindly into the SwiGLU manifest.

The launcher validator currently checks safe tags, integer dimensions/rank, `rank=len(c)>=1`, `d>=2*rank`, and `m>=rank` ([run_one.py](../swiglu/code/run_one.py:33)). It does not validate link syntax or reject degenerate coefficient combinations before launch. For a new manifest, source/config validation should explicitly check every link against both registries, all coefficients finite and nonzero for the intended rank, and the retained settings. This can be done without training or evaluating population moments.

## Genuine higher-rank scope and interaction restriction

`swpop.Teacher` forms an additive target

    Y = gamma * sum_{i=1}^r c_i phi(X_i),  X ~ N(0,I_d),  U = span(e_1,...,e_r).

Its `teacher_terms` loops over every declared coordinate ([swpop.py](../swiglu/code/engine/swpop.py:119)); it is not a bivariate teacher relabeled with a larger r. Each proposed scalar link has a nonconstant derivative with positive variance under a Gaussian. With all c_i nonzero, the teacher's gradient second-moment matrix on U contains a strictly positive diagonal term:

    E[∇Y ∇Yᵀ]|U = gamma² [ Var(phi'(Z)) diag(c_i²)
                                  + E[phi'(Z)]² c cᵀ ].

Thus the intended teacher input-gradient span has dimension r for these additive links. For the normalized Hermites, derivative variances are 3 for `he3` and 2 for `he2`; for `relu` they are 1/4; for `tanh`, nonconstant sech² gives positive variance. This is a mathematical property of the targets, not a claim that the student learns all r directions.

The pure quadratic target is rotationally invariant within U. Its teacher subspace is meaningful, but recovery of a particular listed orthonormal coordinate basis is not an intrinsic specialization claim. Principal-angle/subspace diagnostics are appropriate; coordinate captures are explicitly basis-relative.

Preserve the separate ReLU `ProductTeacher` restriction: [teachers_product.py](../code/teachers_product.py:61) explicitly requires **r=2** for all entries in that registry, including its two-coordinate additive SiLU controls. SiLU/ReLU/GELU/tanh/quadratic/bilinear product targets must not be promoted to rank 8 by changing a config field or padding U. A higher-rank interaction construction would need a new target definition, its true gradient span, normalization, and audited cross moments. None is part of this recommended expansion.

## Normalization and comparable quantities

The SwiGLU teacher factory uses piecewise Gaussian quadrature to estimate μ=E[phi(Z)] and q=E[phi(Z)²], then sets

    gamma⁻² = sum_i c_i² (q−μ²) + (sum_i c_i μ)²,
    EY = gamma sum_i c_i μ,
    V = 1−EY².

See [swpop.py](../swiglu/code/engine/swpop.py:41). This normalizes **E[Y²]=1**, not necessarily Var(Y)=1. For balanced c_i=1, V=(q−μ²)/(q−μ²+rμ²). Consequently:

| Link | Mean | V under balanced rank r |
|---|---|---|
| `he3`, `he2`, `tanh` | Zero | 1, up to quadrature rounding |
| `relu` | Positive | (π−1)/(π+r−1) |
| `abs` | Positive | (π−2)/(π+2r−2) |

`SwiGLUPop.evaluate` profiles the free intercept by centering teacher/feature cross moments and the feature Gram, then computes the unnormalized prediction MSE ([swpop.py](../swiglu/code/engine/swpop.py:168)). The training loop records `L=ev['L']/T.V` but takes its gradients from the unnormalized `ev` ([swsmall_periodic.py](../swiglu/code/engine/swsmall_periodic.py:111)). All parameter blocks use the same mean-field Euler step, adapted by the largest relative per-neuron change. These conventions must stay explicit: equal normalized loss, update count, or reported physical flow time does not imply equal raw target scale or equal cross-link dynamics.

Preserve and report `gamma`, `EY`, `V`, raw MSE (= recorded normalized MSE times V), and variance-normalized MSE. Keep the original initial-prefix rule `abs(L_t−L_0)<=.001` in the normalized units, plus the secondary .0001 prefix. Do not replace the target by a centered/variance-one target without declaring a new experiment. Profiling an output intercept does not mean centering input gradients: the AGOP remains the uncentered full-predictor input-gradient second moment.

The new SwiGLU arm and ReLU arm with the same named link share a target form under balanced weights, but their training conventions, loss units, plateau windows, and readouts differ. ReLU bounded refit and SwiGLU unrestricted intercept refit are separate observables, not pooled replicates or matched performance measures.

## Cost and censoring

The dominant population evaluation uses two-projection pair quadrature for all neuron pairs and a teacher-coordinate loop. From [swpop.py](../swiglu/code/engine/swpop.py:119) and [evaluate](../swiglu/code/engine/swpop.py:152), the principal quadrature workloads scale roughly as

    O(m² n_pair²) + O(r m N_x n_z),

in addition to dimension-dependent matrix work. `N_x` is the total teacher quadrature nodes. The fixed splits −10,−6,−4,−2,0,2,4,6,10 give eight intervals, so `n_x=12` means `N_x=96`; the ReLU/absolute-value kink at zero is already a split. At m=64,r=8, default orders give 1,048,576 pair-node combinations and 1,179,648 teacher-node combinations per evaluation. These are operation-count indicators, not measured seconds.

All proposed links use the same grids and loop counts. Their phi values are precomputed in the teacher object, so the new scalar-link formulas should not by themselves multiply per-step cost. ReLU/tanh can nevertheless change checkpoint density, adaptive step sizes, and stopping times. Increasing rank mostly increases teacher terms linearly at fixed width; doubling width roughly quadruples pair quadrature and increases dense refit costs. Keeping m=64 is a deliberate bounded-screen choice, not an argument that it suffices for full-r recovery.

Posthoc `metrics` reruns a complete `evaluate(...,need_grad=True)`, constructs AGOP, performs dense refit solves/eigendecomposition, and recomputes geometry ([diagnostics.py](../swiglu/code/diagnostics.py:63)). Therefore diagnostics are substantial work, not only reading cached scalars. The default 90-second diagnostic ceiling is soft: the driver checks time before each evaluation, and an evaluation already underway may finish after the deadline. Baseline priorities are initialization, last saved points in the .001 and .0001 prefixes, terminal state, then remaining saved states. Doubled-order checks occur **after** baseline traversal ([run_one.py](../swiglu/code/run_one.py:219)); merely passing `--double-quadrature` does not ensure any doubled result is produced. Doubling all orders multiplies the pair and teacher quadrature-node products by four and diagonal quadrature by two, before other overhead.

Keep the old 1200-second training and 90-second frozen-diagnostic limits as planning references, not new-server runtime predictions. Predeclare actual new-server worker count, per-arm cap, diagnostic reserve, and global cap. Reserve enough budget to attempt the priority diagnostics on every arm. Preserve explicit `not_evaluated_runtime_budget` rows and durable snapshots if later frozen-state checking is needed; never treat their absence as alignment zero or success. The training timer is also soft and only takes effect after an in-flight evaluation completes ([run_one.py](../swiglu/code/run_one.py:130), [swsmall_periodic.py](../swiglu/code/engine/swsmall_periodic.py:120)).

An independent read of the ten saved final rank-eight records provides a more concrete historical reference. The eight h3 runs reaching 1000 updates took **122.17–145.73 seconds training, median 127.88**, and **4.73–6.37 seconds diagnostics, median 5.00**. Median total time was **132.74 seconds**, and every retained checkpoint was evaluated. Eight arms at that total median would be about **1062 serial worker-seconds (17.7 minutes)**; this is neither a measured new-link runtime nor a prediction under the new server's concurrent load. The two other rank-eight outcomes ended at 859 and 79 updates with effective training allocations 117.71 and 13.02 seconds. Those short times reflect deadline censoring, not unusually fast completion. These numbers come only from the archived `result.json` records in [the final SwiGLU data folder](../remote/lightning_final_completed/extracted/data/swiglu/), not a new model evaluation.

Historical evidence reinforces the censoring requirement: the final 40 SwiGLU attempts comprise 35 update caps, one loss stop, and four wall limits ([final audit](FINAL_BATCH_AUDIT.md)). The wrapper labels every non-loss-stop outcome `censored_or_failed`; distinguish the exact stored termination reason instead of interpreting that label as an execution failure. Reaching 1000 updates is a completed planned attempt, not scientific convergence. A prefix ending only because observation stopped is right-censored; extending it can add evidence, whereas extending a trajectory after an observed initial-prefix exit cannot move later recovery into that original prefix.

## Scientific interpretation and promotion checks

For each declared seed, retain all-update loss, both time axes (update count and physical flow time), full spectrum, top/r boundary gaps, raw leading score, complete principal-cosine vector, minimum/mean top-r scores, teacher-axis captures, teacher energy fraction, weakest-direction energy, and corrected refit risk. Existing 1e-9 spectral/PSD flags are warnings, not certificates; retain flagged raw diagnostics and their uncertainty separately. Use actual equilibrated-pseudoinverse prediction MSE at cutoff 1e-12 and preserve cutoff sensitivity, ridge actual risk, and source penalized refit separately ([diagnostics.py](../swiglu/code/diagnostics.py:63)).

The old higher-rank results already show why scope matters: the saved-record review reports median best in-prefix leading scores of .996–.998 across ranks, while median best minimum-alignment gains shrink to about .00371 at rank 8 and .000680 at rank 16 ([earlier scope review](higher_rank_next_plan_20260928.md)). These are descriptive historical diagnostics. A nearly teacher-contained leading eigenvector or high teacher mass is compatible with only one learned direction.

Use this screen to ask whether **leading-direction and partial/subspace behavior generalize across specified additive teachers**. Full-r recovery is a separate exploratory endpoint requiring a stable r/(r+1) cutoff and satisfactory minimum principal angle at the relevant checkpoint. An all-directions-plus-refit event must occur at the same saved checkpoint inside that seed's own initial loss prefix; medians in separate panels cannot establish it. Same-J SVD/Gram agreement does not establish quadrature accuracy or population identification. Recheck promising or unresolved frozen states with the existing numerical audit and independent quadrature/sampling diagnostics on the server before making a stronger claim.

No theorem for arbitrary teacher links, high-probability conclusion, rank-scaling law, or full-r recovery conclusion follows from this eight-arm screen. The renewed cohort should report all planned outcomes, uncertainty, missing diagnostics, and censoring even if none reaches an all-directions threshold.
