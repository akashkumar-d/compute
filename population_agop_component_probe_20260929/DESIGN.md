# Frozen-state component probe — prepared, not launched

This diagnostic separates the finite-order evaluated loss and nominal slopes into teacher and student terms at the two pinned q=.3, seed641 states from the completed cubic pilot. It does not advance a trajectory, test an optimizer, or authorize training. The integrator probe and its a01/a02 artifacts remain unchanged.

All four slots are required: saved update472 (last-valid 5% endpoint) and update2260 (base of the old bad step), each at native orders `(32,96,48,24)` and doubled orders `(64,192,96,48)` in `(n_pair,n_diag,n_z,n_x)` order. Saved source planes, configuration and provenance are copied byte-for-byte from `integrator_quality/probe_inputs`; unused next-state planes remain in that unchanged archive. Each slot forms its own full-gradient direction at that order. The physical epsilon scales use the same saved native proposed dt at both orders: `.2441632677378478` for update472 and `.46444123148065264` for update2260, times `1e-3` and `1e-4`.

The scope gate requires exactly the pure canonical h3 link, fixed raw moments `(0,1)`, teacher `EY=0` and `VarY=1`. The engine uses `intercept='refit'` and `bias=True`. With `al=alpha/m`, the evaluator returns centered `t` and `G`, and we record

```
F = L/VarY = 1 + T + S
T = -2*al*(a @ ev['t'])/VarY
S = al**2*(a @ ev['G'] @ a)/VarY
```

The intercept correction is inside the centered Gram matrix. Since EY is exactly zero in this restricted teacher, the centered teacher term is the raw `teacher_terms` term and its derivative needs no mean correction. This formula is not generalized to nonzero-mean teachers. The reconstructed loss residual is retained for the base and both perturbed evaluations.

The unchanged engine function supplies `D=(m*gP,m*gV,m*head_lr*ga)` from the full base evaluation; `head_lr` is applied once. Every reported directional slope is along **minus D**. The total nominal normalized slope is `-dot(gFull,D)/VarY`. A single separate call to `E.teacher_terms(*E._split(P,V))` returns `(t,gpt,gvt)`, giving the raw-loss teacher blocks `(-2*al*a*gpt,-2*al*a*gvt,-2*al*t)` and teacher nominal slope `-dot(gTeacher,D)/VarY`. The student nominal slope is total minus teacher; it is not an independent student derivative evaluation. The freshly returned `t` is compared with the base evaluator's `t`.

All these gradients are analytical population formulas approximated by parameter-dependent quadrature. Their reported slopes are **nominal**, not asserted exact derivatives of the finite-order evaluated objective. For each component C, the central difference is `[C(theta-epsilon*D)-C(theta+epsilon*D)]/(2*epsilon)`. The four perturbed full evaluations already contain both T and S; no additional component evaluations occur. We report signed `FD - nominal` discrepancies, absolute and relative magnitudes, FD and mismatch additivity residuals, and the two-scale change. Agreement uses `1e-3*max(abs(FD),abs(nominal)) + 128*machine_epsilon*max(1,abs(Cbase))/epsilon` separately for each component. This is a diagnostic tolerance and not a population-accuracy certificate.

Teacher and student component slopes may have either sign. Only the full nominal descent slope is expected negative. Nominal and FD cancellation factors `(abs(sT)+abs(sS))/abs(sTotal)` are reported separately; a zero total is explicit with a null factor. Opposing small component errors can produce a large relative total error, so relative discrepancies alone do not attribute a kernel defect. Completion records successful measurement, irrespective of whether slopes agree. A component mismatch does not prove the cause or support a kernel rewrite by itself.

The workload is fixed: 20 full `evaluate` calls (four gradient bases plus 16 nongradient perturbations) and four separate `teacher_terms` calls, 24 kernel calls total. No old-step, Armijo, cache, AGOP, refit or training calls occur. Source/input pins and exact-manifest independent approval are checked before any NumPy/model import. Reused byte-identical guard functions require Linux SERVER, 32 allocated CPU, at least 8 GiB available memory and 5 GiB disk, nice>=10, then pin this process to one CPU with all BLAS threads=1 and no GPU. The internal deadline is170 s; every slot and epsilon subslot starts explicit and missing/capped/failed work remains explicit. Incremental results and started/returned kernel-call receipts survive partial completion. Input hashes are rechecked at exit; a fresh direct-child output directory is mandatory.

Root must review and approve the exact manifest, publish it, verify live capacity, and apply the same external hard timeout before execution. From this directory, with the verified server Python:

```
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 175s <verified-python> probe.py --output probe_a01
```

This is one CPU with an external 180-second hard bound. No scheduler is added. A hard kill can leave a slot marked running: that is missing work, never an inferred numerical result. All four slots must complete for a complete diagnostic. The only local check is one synthetic scalar arithmetic/JSON round-trip test plus source compilation/hash checks; no model is imported or evaluated locally.
