# Independent continuation dispatch review

Approved for root-controlled server dispatch of manifest **`42665f58da72f1f8ef1271fa0b499b152ee0fdddc5551a3a7e6f97a68f02fb73`**. This approves the exact adapter and resource plan, not any scientific outcome. Machine-readable authorization and limitations are in `REVIEW.json`.

All 32 pinned file hashes match; the 11 scientific files remain byte-identical to the original validation bundle. The saved input has 438 evaluated states and 45 finite snapshots. Input/array hashes and reconstructed scheduler fields independently match `INPUT_CHECK.json`.

The adapter restores exact step-437 parameters and time, original initial balance references, all prior rows/history, and both completed initial-prefix exits. Its first evaluation must exactly match the saved loss, relative update rate, and adaptive step before applying the original 437→438 update once. It draws no new initialization and adds no duplicate anchor. The numerical loop after bootstrap is unchanged. The original total limits remain step 3000, physical time 100, and loss 0.8.

Independent replay places the last natural checkpoint at step 431, with next checkpoint time **40.53101837127362**, `last_dL=0.05299185929986161`, and `last_L=0.9470081407001384`. Forced terminal step 437 is preserved as an extra without resetting that schedule.

Eight independent synthetic/state tests pass. They use AST-extracted functions with a deterministic artificial evaluator and import no scientific engine. Interrupted-plus-resumed and uninterrupted runs have exactly equal final parameters, every loss/time/update record, prefix exits, and the natural checkpoint grid; the only additional checkpoint is the forced interruption state. Tests also cover forbidden RNG, exact anchor mismatch refusal, changed configuration/clock refusal, and diagnostic preservation. Twenty-four launcher mock tests and the static dry-run independently pass.

The 29 successful original diagnostic records are retained with explicit origins. Missing exact order-4 states 358/380 are first, followed by the remaining 14 missing old states and then new checkpoints. A test verifies this order, complete coverage, no duplicate work, immutable original records, and rejection of mismatched checkpoint indices. Existing scientific criteria and unresolved/censored outcomes remain intact.

Adapter and input hashes are written before scientific execution. New outputs separately identify the merged trajectory and continuation segment sharing anchor 437. They retain the interrupted original termination and do not retroactively create an original successful receipt. A newly hashed merged result is explicitly a continuation artifact.

The launcher enforces Linux SERVER mode, one single-thread child, disabled GPU visibility, and a verified reduced-priority gate before dispatch. Its admission checks require 32 allocated CPUs, 8 GiB available memory, and 5 GiB free disk. The resource plan is 1200 seconds soft continuation training plus 600 seconds diagnostics and 10 seconds cleanup, within an 1810-second arm cap and 2100-second global cap. Process-group cleanup, unreaped-child reporting, existing-output refusal, and restart locking were reviewed and mock-tested. These are code-level checks; actual server enforcement remains to be observed.

Before dispatch, root must verify at most 29 other live task training workers, keeping combined training at most 30 and two CPUs free. This external concurrency check is not implemented across bundles. Use the original numerical runtime: a backend-induced anchor mismatch must fail rather than acquire a tolerance fallback. Any manifest/source/input/resource change invalidates this review.

No local scientific evaluation, training, or network action was performed. Future completion, later loss release, complete diagnostics, numerical accuracy, and trajectory convergence still require review of the resulting artifacts.
