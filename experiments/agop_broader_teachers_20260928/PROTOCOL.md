# Broader-teacher higher-rank development screen

Declared before new outcomes. This extends the existing higher-rank study in an isolated folder, preserves all old results, and makes no manuscript changes. The current user request authorizes reconsidering training with more links. The earlier one-hour compute limit is retained for this first screen; a larger study requires a later explicit batch decision.

## Fixed first cohort

All cases have input dimension64 and teacher rank8. Seeds501 and502 are fresh development initializations, paired within architecture and dimensions. This is28 arms, not28 independent teacher families.

| Student | Teacher links | Width | Initial scale | Arms |
|---|---|---:|---:|---:|
| ReLU | h3, damped cubic g_(2/3), positive three-atom cubic mixture |128|0.0001|6|
| ReLU | h3+0.1h5, h3+0.1sin(z), h2, h4, sin(z) |128|0.0001|10|
| ReLU | raw ReLU(z), abs(z), with their means retained |128|0.0001|4|
| SwiGLU | h3, h2, raw ReLU(z), tanh(z) |64|0.05|8|

The ReLU positive-mixture family is covered qualitatively by the paper's family assumptions, but these numerical size/scale choices do not certify its quantitative theorem prescriptions. Other links are exploratory. All chosen links are in Gaussian Sobolev space. Every teacher is a normalized additive function of eight independent Gaussian coordinates; no rank-two product link is mislabeled rank8. Their genuine index rank is8 because the derivative has positive variance in every coordinate direction. The positive three-atom mixture uses the existing fixed scales(.4,.7,1) and weights(.2,.3,.5).

ReLU target: Y=sum_i q(X_i)/sqrt(r*(E[q(Z)^2]+(r-1)E[q(Z)]^2)). Thus E[Y^2]=1 and a nonzero mean stays in the task. Damped/mixture links use the existing exact L2 normalization. SwiGLU uses the existing normalization E[Y^2]=1 and profiles an output intercept; its loss plots divide by Var(Y). Preserve target means, second moments, variances and raw losses. Do not quietly center teachers or equate raw clocks across those conventions.

## Optimization unchanged

ReLU uses all trainable IID Gaussian heads, weights and biases of variance scale^2/d, averaged network output, the existing force step0.05/persistent Armijo rule, and20,000-update maximum. Raw full-MSE step is m*h/2 before backtracking. Refit uses the original normalized feature bank, l2 cap32, l1 cap64sqrt(r), no extra intercept. The sole scientific runner edit broadens the allowed teacher names to existing analytic kernels; the five numerical kernel/support modules remain byte-identical.

SwiGLU uses the unmodified population adaptive Euler engine, output scale1, profiled intercept, head_ratio1, relative-change step cap0.01, dt_max50, t_max1e6, max_steps1000, and original loss stop0.5. The same quadrature orders and unrestricted refit are retained. This is an approximation to gradient flow, not fixed-step GD; it is not pooled with ReLU's bounded refit.

## Outcomes and timing

Keep the original loss-only maximal initial plateau: ReLU MSE in[.95,1.05] and running max/min<=1.05 on every update; SwiGLU variance-normalized change from initial loss<=.001. Use each run's own plateau endpoint. The .0001 SwiGLU threshold remains a labeled secondary diagnostic.

At the same saved checkpoint report minimum and mean AGOP principal-angle scores, gain from that run's initialization, top-direction score, full spectrum/gap/residual and warning flags, teacher-axis captures, teacher gradient-energy coverage and bounded/unrestricted refit as appropriate. The legacy ReLU joint event keeps minimum gain>=.5 and numerical lower refit improvement minus1e-5>=.399. Keep the existing SwiGLU numerical conventions separately. Report first observed crossings before/after plateau and all-update trained-loss drop independently. Medians do not establish within-run co-occurrence. No interpolation certifies an event.

For raw ReLU/abs and strongly linear sine, the initial bounded-refit error can already be too small for an absolute.399 gain. Record initial error and threshold eligibility; report continuous absolute and relative-to-initial gains where initial error is positive. Retain the legacy absolute threshold without presenting an ineligible control as evidence against learning. Mean/linear/residual loss decomposition is already recorded by the ReLU runner. Balanced weakest-direction energy need not approach1 for intrinsically anisotropic targets.

All28 declared outcomes stay in tables, including not-started, failed, wall-censored, incomplete diagnostics and numerically unresolved geometry. Two development seeds are not high-probability evidence. Any apparent new success requires saved-state numerical checking, then a separately fixed confirmation cohort; no automatic postselection or retries.

## Capacity, budgets and queue

Four single-thread CPU workers,3300seconds maximum batch compute,480seconds hard per arm with a normal/full-budget140second diagnostic reserve and cleanup margin. A shortened tail arm can have a smaller effective reserve in the unchanged engine. If every arm consumes its cap,28*480/4=3360seconds exceeds3300; some arms must therefore be censored, and all remain listed. SwiGLU's soft training budget is330seconds because its diagnostic budget is separate; ReLU receives a combined470second soft budget. Some runs may reach caps before completing. Earlier comparable timings are planning evidence, not a guarantee on this four-core machine.

The named existing dependency is modarith-p61-decay-grid-20260928-v2. A queued launcher waits at most7200seconds for its exit record and non-live PID, polling every30seconds; this waiting is distinct from the55-minute computation ceiling. It never stops another task or changes hardware. Named-dependency completion is not proof of exclusive server access; inspect live status at dispatch. If the dependency remains active at queue timeout, record no training and return. No local training and no GPU use.

The frozen source is versioned on the user's shared compute repository on a new branch. Server preflight checks configuration hashes, teacher factories and small finite-difference population gradients before training. All experimental output lives in a fresh execution/ directory that cannot be overwritten. Freeze configuration and source hashes after independent review and before remote execution.

## Next stages, not launched by this screen

After reviewing this complete development record, predeclare matched rank4/rank16 comparisons and width128/256 or initialization-scale ablations. Preserve controls and all outcomes. Stronger paper figures require fresh-seed confirmation, same-run plateau alignment/refit timing, and numerical checks of candidate states. Higher-rank interaction teachers need an explicitly validated disjoint-pair adapter; current ProductTeacher remains rank2 only. The old eight-arm h3 width/scale pilot remains preserved and unlaunched.
