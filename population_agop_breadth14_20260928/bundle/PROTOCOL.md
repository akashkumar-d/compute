# Canonical fourteen-link screen, frozen before outcomes

This is the first common 14-link screen for ReLU and SwiGLU students. The exact raw functions and scalar Gaussian moments are in TEACHERS.json. Four Hermite degrees, three related kinked links, three smooth rectifiers, three bounded odd links and one even radial-basis link provide coverage; this is not fourteen independent learning mechanisms.

## Design

- Rank8, dimension64, orthogonal coordinate teacher directions, equal additive coefficients.
- Both students target the same uncentered additive function normalized to E[Y²]=1. No target is silently centered. Both learners explicitly profile a trainable output constant at each state; initial raw loss is approximately Var(Y), which is often below1.
- Two fresh exploratory seeds641/642 for every student/link:56 attempts,28 cells. Seed641 covers all cells before seed642; within each seed, alternate ReLU and SwiGLU for each listed link. Outcomes cannot change this queue.
- ReLU: width256, raw Gaussian coordinate scale0.01/sqrt(d), all heads/weights/biases trained, force-step0.5 with the existing Armijo rule, target20000 updates or force-time1500 or raw MSE0.001. Diagnostic stride25 plus the existing geometric grid and exact initial1%/5% boundaries.
- SwiGLU: width64, Gaussian scale0.1, head ratio1, all gate/value/head parameters trained, head learning-rate multiplier0.01, gate/value multipliers1, relative Euler cap0.01, dt cap50, target time600/5000 updates or MSE/Var(Y)0.1. Existing exact initial1%/5% boundary snapshots are retained. Logarithmic checkpoint ratio1.06, minimum spacing0.5, loss-change ratio1.5. Quadrature orders pair16/diagonal48/conditional24/teacher12.

These are student-specific development recipes selected before this screen. Width, scale, update method and step meaning differ; this is not a controlled ranking of one architecture against the other. Seeds are fresh, but this exploratory breadth screen does not establish high-probability success.

## Compute bounds

Following the user's64-core allocation and request to leave four free, use at most60 single-thread workers; no GPU or local training. Launch requires at least64 effective allocated CPUs,24GiB available memory and10GiB free disk. Each arm is limited to900 seconds, including710 seconds training,180 seconds diagnostics,10 seconds cleanup. All60 attempts fit in one wave; the dispatcher global cap is1200 seconds. Setup is separately capped at120 seconds. This replaces the earlier unlaunched4-worker/210-second proposal. It must not silently run on the old4-core allocation.

Four extra arms reuse the previously prepared pure-sine scale design: ReLU rank8/dimension64/width128, seeds631/632, coordinate scales0.0001 versus0.01, force-step0.05, no profiled intercept, targets20000 updates/force-time1000/rawloss0.02. They receive the new900-second allocation and the declared loss-only diagnostic priority. Their original frozen preparation remains unchanged. These extra runs do not count as new canonical links or as a matched comparison with the56 common-grid runs.

Every failed, capped, numerically unresolved or unstarted arm stays in the coverage table. Some scientific horizons will remain unreached, especially higher-degree SwiGLU cases. Concurrent throughput is unknown; a wall timeout is not evidence of impossible learning. The180-second diagnostic budget can still be shorter than a dense width256 grid, so incomplete diagnostics remain visible.

## Evidence rules

The primary windows are the maximal initial prefixes for which the maximum/minimum of the positive raw training loss over **every update** is at most1.01 or1.05, reported separately. At one actual saved checkpoint, require (i) minimum squared-principal-cosine gain at least0.5, (ii) passing initial/current original AGOP resolution screens, and (iii) numerical lower same-representation refit MSE gain at least0.1 Var(Y). A subsequent same-run raw MSE drop of at least0.1 Var(Y), measured from this candidate, is reported separately. A high mean or top-one alignment cannot replace the minimum-direction criterion.

Numerical evidence needs finite losses, declared-order convergence/sensitivity checks and no unresolved refit solver gap at the candidate. New smooth-teacher quadrature is explicitly numerical. Original relative AGOP eigenvalue/gap guards are unchanged; an uncertain subspace is not upgraded by a cosmetic plot repair. Promising cases require a follow-up saved-state audit and additional seeds before stronger claims.

The diagnostic budget is deliberately smaller than the cost of some complete saved-state grids. After initialization, exact loss-prefix boundaries and terminal states, evaluate the already-saved states nearest25%,50%,75% of each prefix's last admissible update (within that prefix; earlier state breaks ties). This priority depends only on update indices and the loss window, never alignment/refit. Then attempt all remaining saved states in the inherited order. No saved state is removed from the required grid: uncomputed states remain censored, and a sparse negative result is not a resolved full-grid negative. The omitted-option priority is unchanged for legacy configurations.

A secondary later window may be selected by the already declared loss-only longest-band rule, separately labeled exploratory. It does not replace an unsuccessful initial prefix. No window is selected from alignment or refit curves.

Plot raw MSE, AGOP minimum/mean alignment and same-budget refit error. Use the established colors and black labels. Median curves require common observed support of all planned seeds; retain individual tails and numerical gaps without interpolation across missing required diagnostics. Keep each per-run joint verdict alongside aggregate plots.

## Preservation

Older sources, data, plots and the papers are untouched. New source adapters and fixed teacher moments undergo review before bundle freezing; the manifest hashes every executable/configuration input. Shared compute source is versioned in a distinct repository subdirectory/branch. Full job records and data are downloaded to a new directory. Restart never overwrites a completed attempt.
