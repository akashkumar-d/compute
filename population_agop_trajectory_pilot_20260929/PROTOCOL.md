# Matched-scale safeguarded trajectory pilot — prepared, not launched

The four complete configurations are copied from MATCHED_SCALE_PROTOCOL.json:
pure additive h3, rank8/d64/width64, reused development seeds641/642, inner
scale0.3 or1.0, matched actual initial heads of scale0.3, head_lr0.01. Original
Gaussian draw order and arithmetic are preserved. All configurations, numerical
settings and this protocol are pinned before outcomes. These are four arms,
two paired development seeds; no fresh confirmation claim.

**Fixed evaluator.** All four arms use unchanged Hermite degree128/scalar
transition order24, inherited diagonal/teacher transition order16, Gaussian
limit12. Thirteen reviewed diagnostic-transfer sources are byte-identical.
hermite_audit_only.py is a separate explicit clone allowing only degree256/
order32; HERMITE_AUDIT.diff contains every change. No adaptive masks, order
switches, hidden normalization or teacher changes. The inherited population
gradient approximates the intended Gaussian objective; it is not the exact
derivative of the finite-series loss. The frozen kernel's diagnostic-only label
remains unchanged: conditional pilot authorization belongs to this reviewed
harness and a separately reviewed smoke, not to the kernel itself.

**Update.** D=m(gP,gV,head_lr*ga), with the exact frozen arithmetic order.
Proposed dt=min(dt_max,h/max_j(||D_j||/||theta_j||),remaining physical time),
h=.01. Trial theta-dt D must have strictly lower evaluated raw MSE and satisfy
Ltrial <= L-dt*1e-4*(g dot D)+tol. Halve dt at most20 times (21 trials).
tol=8*float64-eps*max(1,abs(L),VarY). If requested Armijo decrease <=tol, record
numerical_stall and accept nothing. Every trial uses the same base evaluator
and full gradient; every rejected trial is retained. Nonfinite moments or
gradient, materially negative scalar residual energy, raw MSE<-1e-10 VarY,
or failed audit halts and saves the failing state. Tiny signed negative MSE
is retained and produces numerical_stall, not successful loss-stop completion.
Diagnostic failure before a state commit saves its uncommitted candidate;
parameters are never paired with another state's moments.

**Diagnostics and sampled audits.** Canonical diagnostics.metrics is unchanged
and receives exactly one cached evaluation; AGOP/refit consume the cached
moments. Every accepted update saves raw/normalized loss, clock, state hash,
metrics and validity flags. Natural checkpoints use original cp_ratio/cp_min/
dl_ratio and .02 loss-change rule. Additional initial, both previous/current
first-window boundary states, first candidate, first later release and final
states are saved without advancing the natural schedule. Snapshots include
P/V/a, gradients, G/t, Cp/D, teacher variance, AGOP, cutoff weights/masks,
Gram eigenvalues and signed scalar energies/per-pair tail estimates.

Higher-order checks occur initially, every25 accepted steps, at both boundary
states, first candidate/release and final. Full unchanged transition reference
is required before the first update and at first candidate/release/final;
duplicate state/method audits are reused. Loss tolerance1e-7 VarY, weighted
direction/common-slope relative1e-4 with declared1e-10 absolute scale terms,
AGOP Frobenius relative1e-6, error/gap1e-3, alignments1e-4, and cutoff self/
cross risk1e-7 are engineering screens. The existing AGOP1e-9 and refit
cutoff1e-8/1e-10/1e-12/1e-14 validity/sensitivity screens are unchanged.
Resolution/stability disagreement between evaluators halts; both invalid may
continue with invalid metrics and cannot support a qualified candidate.
Every event remains explicitly unverified until independent saved-output
review, including reference validity at candidate/release. Failed/capped audits
remain visible and cannot certify an event. Sampled checks and finite scalar
tail estimates are not rigorous integral or whole-trajectory certificates.

**Scientific criteria unchanged.** Initial prefixes use all accepted-update
losses and max/min<=1.01 or1.05. Same-checkpoint delta Amin>=.5 and initial
minimum cutoff-risk minus max(0,current maximum cutoff-risk)>=.1 VarY require
valid initial/current metrics. Subsequent raw loss drop>=.1 VarY is separate.
Invalid initial refit prevents qualification; the arm remains in the denominator.
Negative, failed, unresolved and wall-censored outcomes are all retained.

**Smoke first; no automatic dispatch.** One CPU, hard180s (TERM175/KILL5),
admission170s, maximum80 model evaluations. Four old pinned states472/473/
2260/2261 compare base and high with existing saved reference moments. Old
state comparisons retain their original .0009 head rate. No fresh reference
calls in smoke. Each of four matched initials gets base/high checks, central
directional differences at relative1e-4 and1e-5 (slope tolerance1e-3), and two
Armijo updates, each audited at high order. All four arm placeholders and
initialization hashes survive a cap. This is engineering only. A complete
passing smoke plus independent SMOKE_REVIEW.json bound to its exact result
and manifest is mandatory for any pilot arm. Full reference initial gate is
then performed independently within each pilot budget before its first update.

**Resource/output contract.** Linux SERVER only, allocated>=32CPU, available
memory>=8GiB, disk>=5GiB, nice>=10, exactly one chosen affinity CPU and all
BLAS/OpenMP pools1; CUDA hidden. Parent verifies account, actual live concurrency
(this worker plus at most29 others), and distinct CPUs before each dispatch.
One arm only per invocation; no scheduler or automatic retry. Training admission
ends1800s, final diagnostic admission1980s, TERM1985/KILL5 gives1990s maximum
including reference/degree checks. t_max3000, max_steps15000 and L_stop.01 are
physical/scientific limits, separate from wall censoring. No extensions.
Fresh direct-child output and exclusive persistent attempt journal prevent
overwrite/restart. Any interrupted attempt requires explicit recovery review.
RESULTS.json is atomic; append journals retain started/returned calls and trials.
Publish source only. Four reference NPZ files are exclusively copied by pinned
SHA256 from the completed same-Studio diagnostic-transfer run; never publish
or regenerate those states. Source approval and input hashes are checked before
scientific imports. Local tests use synthetic arrays/polynomials only.
