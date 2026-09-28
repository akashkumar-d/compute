# Fourteen teachers × ReLU and SwiGLU

The new common screen contains14 distinct raw scalar links for each student and two fresh seeds per cell:56 runs. Four supplementary pure-sine scale comparisons bring the planned batch to60. See [GOAL](GOAL.md), [protocol](PROTOCOL.md), [prior-coverage inventory](coverage/coverage.md), and [source review](ROOT_SOURCE_REVIEW.json).

**Current preparation:** both teacher adapters are implemented and numerically checked. No outcome from this new batch is implied by those checks. The user authorized a64-core Lightning CPU with four cores left free. The launcher uses at most60 single-thread workers and requires64 effective CPUs,24GiB available memory and10GiB free disk. It refuses the old4-core allocation. Each run receives at most900s including diagnostics; the dispatcher global cap is1200s, plus at most120s setup. Nothing is launched locally.

Links: normalized probabilists' h2/h3/h4/h5, ReLU, leaky ReLU(slope0.1), absolute value, softplus, SiLU, exact GELU, sine, tanh, erf and Gaussian RBF exp(-z²/2). Centered/raw duplicates and previous mixtures do not inflate the count. Every function is in Gaussian H1; positive dynamics are not guaranteed by membership.

Both students retain the same raw teacher mean and normalize E[Y²]=1 using fixed shared moments. The56-run grid explicitly profiles an output intercept for both learners. Student-specific widths and optimization rules differ, so these runs are not a controlled architecture ranking. Saved SwiGLU losses must be multiplied by Var(Y) once to obtain raw MSE.

## Reproduction

The reviewed portable source is published in a dedicated shared-compute branch/subdirectory; the final repository commit and exact remote run name are recorded in CHECKPOINT.md after publication. On a verified Linux64CPU host with the pinned dependencies installed:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python start_cpu.py
```

`start_cpu.py` first validates frozen hashes, allocation and dependency versions, then runs bounded launcher/diagnostic checks. It starts the detached job only when invoked through lrun; it does not resize hardware, acquire a GPU or install packages automatically. The dispatcher writes `bundle/execution/STATUS.json`, all process receipts, per-run losses, states and diagnostics. It refuses to overwrite an existing execution directory.

Inspect the plan without model evaluations or creating an output directory:

```sh
python bundle/launcher/launch.py --dry-run
```

Use the exact lrun commands in CHECKPOINT.md to launch, inspect and download this attempt. A stopped/repeated experiment needs a new named attempt; preserve every prior output. Never select only successful seeds from different attempts.

## Interpretation

The primary report keeps all28 common-grid cells, both seeds, failed/missing/capped results and unresolved numerical diagnostics. A material candidate requires a same-checkpoint0.5 minimum-alignment gain and0.1Var(Y) lower refit gain within an all-update1% or5% initial loss prefix; later loss release is measured separately. A high mean or top-one alignment does not establish all-direction recovery. Weak or incomplete cells remain targets for follow-up, not evidence of a universal impossibility.

No manuscript was edited. Older positive/negative/censored cohorts remain intact. The old pure-sine preparation is preserved separately; its four longer-budget variants are labeled supplementary in this manifest.
