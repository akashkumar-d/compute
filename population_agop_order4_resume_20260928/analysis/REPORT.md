# Completed quadrature-trajectory validation

The completed order4 continuation preserves the fixed **5% plateau → all-direction learning and refit improvement → later loss decrease** outcome seen at order2. Both orders first meet the unchanged same-checkpoint criterion at step358. Neither has a qualifying saved checkpoint in the initial1% window. This is numerical agreement on **one reused development seed642**, not two independent successes or fresh-seed confirmation.

Both runs use the normalized additive h2 teacher (saved mean0, variance1), rank8, dimension64, width64, SwiGLU with trainable gate/value/head, profiled intercept and biases, initial scale0.1, head ratio1, head learning multiplier0.001, adaptive Euler h0.01, and original global limits step3000/time100/normalized loss0.8. Raw and variance-relative losses coincide because Var(Y)=1. Order2 uses pair/diagonal/z/x quadrature32/96/48/24; order4 uses64/192/96/48. Apart from the output path and quadrature orders, configurations agree, and initial parameter arrays are bitwise equal.

## Unchanged all-update windows and criteria

The canonical reader hashes to `fe0aedb926eafff5adcc966c3dc5042e2263da83e7e33ff6b5210f36a9e6ae0c`. It uses the exact maximum/minimum ratio over every evaluated update. Last-valid endpoints are included and the first crossing is excluded; windows are neither moved nor feature-selected. A material saved checkpoint requires minimum-direction alignment gain≥0.5 and initial-min/current-max four-cutoff refit-envelope gain≥0.1Var at that same state, with original1e-9 spectral rank/gap guards. The later raw loss drop≥0.1Var is measured strictly after that same candidate. The additional numerical qualification remains separate from the original flag.

| Order | Window | Last valid step / time | First exit | Raw endpoint loss | Exact max/min | Candidates original / qualified |
|---|---|---|---|---:|---:|---:|
| 2 | 1% | 257 / 24.1554760390 | 258 | 0.990172374444 | 1.009924810091 | 0 / 0 |
| 2 | 5% | 381 / 34.0439307178 | 382 | 0.952398125792 | 1.049980696241 | 3 / 3 |
| 4 | 1% | 257 / 24.1554760382 | 258 | 0.990172374445 | 1.009924810091 | 0 / 0 |
| 4 | 5% | 381 / 34.0438135719 | 382 | 0.952396018616 | 1.049983019324 | 3 / 3 |

Both5% candidate lists are exactly **358,380,381**. All candidates are observations from the same trajectory; they do not increase the independent-run count. The initial raw loss is0.9999996472183378. Maximum minimum-direction gain within the fully diagnosed1% saved grids is only0.0104783753/order2 and0.0104784203/order4. This establishes non-observation on those grids, not absence at unsaved states.

| First5% candidate metric | Order2 | Order4 continuation |
|---|---:|---:|
| Step | 358 | 358 |
| Physical time | 31.9514934472 | 31.9513790278 |
| Raw training loss | 0.955899183996 | 0.95589747718 |
| Minimum AGOP alignment | 0.570649621988 | 0.553168228583 |
| Minimum-direction gain | 0.565493162628 | 0.548011769222 |
| Mean AGOP alignment | 0.921584474148 | 0.91940668069 |
| Raw refit risk (cutoff1e-12) | 0.181385694786 | 0.181392655918 |
| Cutoff-envelope gain | 0.787644892354 | 0.787637931221 |
| Later raw loss decrease through final state | 0.157080022896 | 0.157312157638 |

The first subsequent≥0.1 loss decrease is at step586 for both orders: time51.0424818531/loss0.855197469946 for order2 and time51.0445901272/loss0.855183712382 for order4. Both stop at step619 after reaching their original loss target: order2 time53.8267580175/loss0.798819161101; order4 time53.8283908754/loss0.798585319542. Neither complete trajectory is wall-censored, and the initial windows are closed.

All53 order2 and54 order4 diagnostics pass the recorded AGOP and refit screens. At the order4 first candidate, relative rank eigenvalue4.410023397e-7 and relative rank gap1.235979208e-7 exceed the unchanged1e-9 guards; the four-cutoff spread is0, maximum normal-equation residual3.020846480e-16, and equilibrated eigenvalue ratio0.08531147647. There are no negative-risk gains or missing diagnostic placeholders in the completed continuation. These are numerical screens, not certified eigenspace errors.

The SwiGLU refit number is a conservative envelope over the declared numerical unrestricted cutoff solves. It is **not** a certified lower bound on improvement of unrestricted optimal-head risk and is not a fixed-norm-budget comparator. Candidate diagnostics were not additionally order-doubled within these result files; the two complete trajectories provide the explicitly separate order comparison here.

## Exact continuation preservation

The original order4 execution stopped externally at437/time38.72140153556326/loss0.9462111101722706. It remains an interrupted execution with no retrospectively invented success receipt. The new exit0 completion is explicitly a merged, segmented continuation.

All438 original dense loss/update records, all45 prior checkpoint rows, and every old t/P/V/a snapshot are preserved exactly; arrays are bitwise equal. All29 successful prior diagnostics are retained field-for-field with an origin label. The16 old budget-missing diagnostics are now computed, including exact order4 states358/380. Nine later checkpoints were added. The final merged trajectory has620 evaluated states and54 snapshots; its segment-only view has183 dense states and10 snapshots because it explicitly shares anchor437. The merged history contains that anchor once.

Replaying the saved loss/time scheduler independently gives last natural checkpoint431 and next_cp40.53101837127362, last_dL0.05299185929986161, last_L0.9470081407001384. Forced interruption snapshot437 is retained as an extra and does not reset this scheduler. The final54-state grid equals the full natural grid plus exact prefix boundary states plus437; the other53 steps match order2. Every required original and later checkpoint is diagnosed. Both prefix exits remain byte-for-byte unchanged from the recovered segment.

The frozen adapter requires exact equality of anchor loss, relative adaptive update rate, and dt before advancing. The pinned code, successful exit, and post-anchor progression establish that its strict gate passed. Anchor values are loss0.9462111101722706, rel0.12413394104468191, dt0.08055814482197507. This audit did not reevaluate that model, and no stored original gradient vector is available for an independent vector comparison. The original total step/time/loss caps and initial balance references remain unchanged.

## Numerical agreement and remaining limits

Comparison uses equal update numbers, not exactly equal physical times; no interpolation or surrogate states are used. Across620 common dense states, maximum raw-loss difference is0.0002338415582 at619, clock difference0.002291521362 at600, and relative proposed-dt difference0.100402064% at619 (order4 denominator). Across53 common saved states, the maximum relative concatenated P/V/a parameter difference is0.695596449% at619, using the order4 parameter norm; through the5% endpoint the maximum is0.0137616646%. The independent receipt uses the order2 norm denominator, hence its terminal0.695387638% differs slightly without an arithmetic disagreement. The maximum minimum-alignment difference is0.01748139341 at358; maximum mean-alignment difference0.002177793459 is also at358. Maximum refit-risk difference is0.0003140105976 at619. All same-state comparisons are retained in COMMON_SNAPSHOT_COMPARISON.json.

The5% threshold verdict survives integration at the finer quadrature order, including the previously missing candidate diagnostics and later release. Alignment remains measurably sensitive, and later parameter separation is larger than in the old truncated comparison. These observations support finite-order trajectory agreement; they do not prove population integration accuracy, step-size convergence, formal AGOP accuracy, or a mathematical refit certificate.

## Provenance, artifacts and reproduction

The32 continuation manifest hashes and24 original validation manifest hashes match local files and server provenance; all11 files under swiglu/ are byte-identical across bundles. Launch/config/source and final raw-output hashes agree. The parent transfer receipt independently verified all17 continuation execution files; this audit tracked88 input artifacts and checked that none changed. Order4 manifest SHA256: `42665f58da72f1f8ef1271fa0b499b152ee0fdddc5551a3a7e6f97a68f02fb73`; original two-order manifest: `39503531b9f5cd0df5fa51d421e48ce797c33bfdbfd3fe5f926511bf44e19a30`.

- `ORDER4_SUMMARY.json/md`: unchanged canonical reader on the completed merged continuation.
- `ORIGINAL_EXECUTION_SUMMARY.json/md`: canonical reader on the original execution, preserving completedorder2 and interruptedorder4 as originally recovered. Its latter missing-diagnostic and integrity qualifications are historical, not the completed continuation's verdict.
- `VALIDATION.json`: preservation, source, scheduler, strict-anchor, candidate and comparison checks with input hashes.
- `COMMON_SNAPSHOT_COMPARISON.json`: every common-state paired metric.
- `INDEPENDENT_ARITHMETIC.json/md`: independently computed prefix/candidate/guard/grid checks; denominator choices are explicit.

Run from this directory with the local NumPy Python runtime: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/legendkiller/anaconda3/bin/python3 reproduce.py`. The script reads saved JSON/NPZ and performs array arithmetic only. It imports the pinned saved-data reader, no scientific engine. It recreates these summaries, numerical audit and report; no training, model evaluations, network operations, original-data edits, or manuscript changes occur.
