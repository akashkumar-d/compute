# Opt-in loss-descent safeguard: source draft

The smallest change is a bounded backtracking check around the existing block direction. `fork/swiglu/code/engine/swsmall_armijo.py` is a separate opt-in engine; `run_one_armijo.py` is its separate runner. Every original scientific Python file is retained byte-for-byte. The fork source manifest is marked draft, so the runner refuses execution until independent review and the fixed-state check below. No new scientific run or run configuration was created.

For θ = (P,V,a), let g be the existing returned gradient of raw profiled-intercept squared loss, approximated by the existing deterministic quadrature. The unchanged proposed direction is −D, where D = M g and M = diag(m I_P, m I_V, m head_lr I_a). All bias coordinates are included. For the exact population gradient this is steepest descent in metric M⁻¹ for raw loss. For F = L_raw / Var(Y), the equivalent metric is (Var(Y) M)⁻¹. The intercept is profiled at every trial; it is not a separately stepped parameter.

The **nominal** normalized slope is

`σ = −m (||gP||² + ||gV||² + head_lr ||ga||²) / Var(Y)`.

The head rate appears once in this inner product, not squared. It is not yet established that this is the exact derivative of the finite-order evaluated objective: the analytical population gradients and loss are approximated through parameter-dependent quadrature. The required directional finite-difference check is below. Direct acceptance still enforces descent of the evaluated finite-order loss independently of whether the nominal slope is accurate.

Start with the existing proposal `dt = min(dt_max, h/max_j(||D_j||/max(||θ_j||,1e−300)))`. At most 25 trials use dt, dt/2, dt/4, … . Accept only if the entire evaluation is finite, the trial loss is nonnegative, and

`F_trial < F_base` and `F_trial <= F_base + 1e−4 dt σ`.

No absolute or relative tolerance permits an increase or equality. The original h, dt_max, direction, initialization, teacher, numerical orders, horizon/step/loss limits and checkpoint selection remain unchanged. This changes the discrete optimization path and accepted clock. It does not fix an inconsistent quadrature gradient, certify population descent, or establish earlier feature learning. In particular it cannot turn the old six negative initial-window records into positive records.

Each trial evaluates loss **and gradients**. The accepted evaluation, including G/D/Cp used by AGOP/refit diagnostics, is passed to the next iterate unchanged. With no rejection this costs one initial evaluation plus one evaluation per accepted update, as before. Rejections add evaluations; there is no second evaluation of an accepted base state. Trial-state construction uses new arrays, and rejected evaluations never replace the accepted cache.

Only accepted states enter `Lhist`, checkpoint snapshots, initial-window crossings and the accepted clock. `step_history.dt` is the accepted outgoing dt (zero at a terminal state); `proposed_dt`, trial count and nominal slope are separate fields. Rejected/accepted trial decisions are recorded in `trial_history` and an append-only `_trials.jsonl` journal. The runner hashes that journal when present. Journal acceptance records a decision; after a hard kill it must not be treated as proof that the candidate was committed to the durable accepted-state history. The current accepted checkpoint remains the recovery reference.

Nonfinite/negative trial losses or nonfinite trial arrays/gradients are rejected. An evaluator exception stops with an explicit error instead of concealing it through repeated retries. Trial exhaustion, invalid direction, zero/underflowed nominal slope, unresolvable Armijo decrement, state/time stagnation and external interruption terminate with named reasons and preserve the current accepted state. A stop is checked before and after every trial evaluation; a callback already running is not preemptible, so the unchanged parent hard deadline/cleanup remains necessary. These numerical halts are not convergence or scientific success. Diagnostics after a halt remain subject to the existing budgets.

Eight new scalar synthetic tests cover rejection/backtracking, invalid loss/gradient handling, exact accepted-cache reuse, exhaustion, stop-before/after-evaluation, stagnation, evaluator exceptions and the metric/variance convention. No model module was imported or evaluated locally. `ENGINE.diff` and `RUNNER.diff` show the complete changes; unchanged numerical or plotting suites were not rerun.

## One decisive server probe before any trajectory

`probe.py` is prepared but unlaunched. It uses the exact last-valid 5% endpoint and bad-step base of the q = 0.3, seed 641 pilot, with their saved next states. The endpoint is update 472, snapshot 43, t = 147.40763866670704, normalized loss 0.9528468742666656. The bad-step base is update 2260, snapshot 673, t = 566.5272355485674, loss 0.0967187558763789, old dt = 0.46444123148065264; the next state is update 2261/snapshot 674 with loss 0.6925173772423037. The endpoint is taken from the canonical saved SUMMARY, not chosen by alignment.

The four exact saved array planes (snapshot indices 43, 44, 673, 674) were extracted with standard-library byte operations into `probe_inputs/states.npz`; no state was generated or evaluated. Provenance pins the original snapshot SHA256 `3d8fb42989b711160f1e5896032de5eeb4f7e2d90e328509d27370b649de10ce` and every extracted plane. Old data remain read-only.

At each anchor, native orders run first; doubled orders are attempted only while the same budget permits. Compute the existing weighted direction D and nominal slope, then centered differences `[F(θ−εD)−F(θ+εD)]/(2ε)` at exactly `ε = old_dt * 1e−3` and `old_dt * 1e−4`. Report both signs, relative/absolute slope differences, their stability and the explicit roundoff scale `128 machine_epsilon max(1,|F_base|)/ε`. The diagnostic comparison uses relative 1e−3 plus that scale; cancellation-dominated, unstable or nonnegative estimates remain unresolved. Completion of the probe is not a validation pass.

At both quadrature orders, use the native saved dt as the initial proposal for the old step and for Armijo. This deliberately fixes the proposed displacement scale when comparing quadrature orders. The recomputed relative-cap dt is reported for comparison only: the doubled-order slot does not reproduce an actual doubled-order engine step or trajectory. Evaluate that exact old dt proposal and its Armijo-backtracked alternative. For native orders, compare the old candidate to the saved next arrays and loss (bitwise equality is recorded; numerical comparison uses rtol 1e−10/atol 1e−12). The bad step must be rejected. The first old-proposal evaluation is reused in the backtracking probe. For an accepted Armijo candidate, compare the entire cached evaluation with one fresh evaluation of those exact arrays. Record all trial decisions and evaluation counts; verify unchanged base-array and input-file hashes. No candidate is committed into a training trajectory.

The existing scheduler's unchanged resource/priority functions are reused via `probe_resources.py`; there is no new scheduler. The probe requires SERVER, at least 32 allocated CPUs before restricting itself to one CPU, one thread per numeric library, nice >= 10, 8 GiB available memory and 5 GiB disk. A 170 s internal soft limit and 64-evaluation ceiling keep every capped/missing state-order slot explicit. Root must provide the external hard bound: TERM at 175 s, KILL five seconds later. Source/input pins and the exact-manifest review gate are checked before any NumPy/scientific import. The output directory must be fresh.

After root review/publication, the bounded command is:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 175s python3 probe.py --output probe_a01
```

Use the root-verified server interpreter. `PROBE_REVIEW.json` remains pending. `SOURCE_MANIFEST.json` remains draft, separately preventing a training runner launch. If slope consistency, native saved-step reproduction, old bad-step rejection or cache reproduction fails, review that issue before a trajectory. A passing probe supports only this numerical safeguard; a later matched old-versus-safeguarded trajectory requires separate review and preregistration. No local model evaluation, remote call or experiment was performed here.
