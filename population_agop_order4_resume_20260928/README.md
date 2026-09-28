# Exact-state continuation of interrupted order4 validation

Prepared only. This resumes the existing seed642/order4 h2 validation from its exact final saved P,V,a at step437, physical time38.72140153556326 and normalized loss0.9462111101722706. It is not a restart, another seed, a new control, or completion retroactively attributed to the interrupted process. Original recovered files remain untouched; pinned copies reside in input/.

## Exact state and missing control

There are438 all-update records and45 snapshots. Every snapshot timestamp matches its raw row; the last snapshot is the externally stopped terminal evaluated state. The recovered full and periodic-partial snapshot payloads match. INPUT_CHECK.json records input and initial/final-array hashes, dimensions, finiteness, and control reconstruction. No model evaluation was needed to establish this structure.

The engine draws randomness only at initialization. It has no momentum/optimizer accumulator. P,V,a,physical time,step number and immutable recipe determine the next deterministic quadrature update. The original total max_steps3000,t_max100,L_stop0.8,h0.01,head_lr0.001,orders64/192/96/48 remain unchanged. They are total trajectory limits, not fresh budgets added at the boundary. Initial-state P,V,a from the first snapshot restore the original balance-drift reference.

Checkpoint-grid state is reconstructed from all saved t/L pairs with the original natural checkpoint predicate. The interruption's forced final checkpoint is preserved as an extra but does not reset the future grid. Last natural checkpoint431 gives next_cp40.53101837127362,last_dL0.05299185929986161,last_L0.9470081407001384. Both initial-prefix exits are already fixed (1% at258,5% at382); they are restored from full history and cross-checked against saved boundaries.

At dispatch the adapter reevaluates the step437 state, requires exactly the recorded L,relative update rate anddt, then applies its original437->438 update once. It does not append the anchor evaluation twice or regenerate initialization. Mismatch aborts before advancing; no tolerance fallback is provided. Frozen kernels and the post-bootstrap run body preserve the source arithmetic. Matching saved aggregates is a strict execution check, not a stored-gradient-vector comparison: the old gradient itself was not saved. The parent should use the original numerical runtime; different numeric backends can fail the strict check.

## Files and segment provenance

All11 scientific files under swiglu/ are unchanged. adapter/resume_engine.py is a separate minimal run-loop adapter; RESUME_ENGINE.diff shows its changes. adapter/run_resume.py adds loading and explicit segment output to the pinned runner; RESUME_RUNNER.diff shows those changes. Launcher changes are in LAUNCHER.diff. Adapter and input hashes are recorded in launch.json and RESUME_INPUT.json before model work, so another interruption retains actual execution provenance.

Each new execution writes a merged raw trajectory and snapshots under data/<originaltag>/ using the existing analysis-compatible names, plus continuation_segment.json and continuation_segment_snaps.npz. The segment-only view explicitly shares the step437 anchor; the merged all-update history includes it once. Continuation metadata retains the original external_stop termination, source hashes, global limits, reconstruction state and new segment bounds. The new result.json clearly identifies segmented completion; no original result.json is invented or updated. Periodic partials and all old input files remain available after any failure.

## Diagnostics and scientific interpretation

The29 old successful frozen diagnostics are preserved with explicit source-segment labels, tied to their pinned JSON and unchanged source snapshots. Exact original order4 checkpoints358 and380 are evaluated first, followed by the other14 missing original checkpoints, then new saved states in the established diagnostic priority order. No interpolation, surrogate order2 state, reconstructed checkpoint, or changed checkpoint grid substitutes for them. The actual diagnostic plan and origins are stored in the result.

The same all-update initial1%/5% max/min prefixes and same-checkpoint minimum alignment gain>=0.5, cutoff-envelope refit gain>=0.1VarY, later raw-loss drop>=0.1VarY and numerical guards apply. Both prefixes ended before the continuation anchor; continuing can establish later release and complete missing diagnostics, not change original prefix membership. Finite-order numerical evidence remains distinct from a population certificate. All missing/failed/capped diagnostics and process outcomes remain visible.

## Server bounds, review, dispatch and preservation

Linux SERVER-only; one single-thread worker, verified nice>=10, GPU visibility disabled. Admission requires32 allocated CPUs,8GiB available memory and5GiB free disk. The parent must check current other task training<=29 immediately before dispatch so total<=30 and2 CPUs remain free. No automatic dispatch or cross-bundle coordination is implemented.

Budgets are1200s soft continuation training +600s diagnostics +10s cleanup,1810s hard arm ceiling,2100s global ceiling. The launcher signals only its own child process group, preserves unreaped children/pending outcomes, and refuses existing output directories. The dedicated kernel lock and durable restart journal block ambiguous restarts. Never clear the journal or overwrite older outputs automatically. A subsequent interrupted continuation requires a separately reviewed state/segment choice, not a blind new invocation from a more favorable checkpoint.

Independent source/state/runtime review of the final manifest is required in REVIEW.json before launch. Static commands, from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_resume_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s adapter -p test_resume.py -v
```

After parent review, staging and external-concurrency checks, exact server-side command:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_resume_a01
```

The parent owns publication and dispatch. Only saved-data inspection and synthetic/mock tests are permitted locally in this preparation; no scientific model evaluation or training is performed.
