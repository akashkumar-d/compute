# Campaign E: transformer pilot (plan fixed before any result)

Written by claude/plateau-review, 2026-09-30, before any development result was read. Akash asked for transformers. He chose:
- a pilot for the Oct 6 submission, at p = 31;
- both 1- and 2-layer models;
- all three optimizers ("see which works");
- the Claude Code sessions for the compute.

This is a development-then-confirmation pilot, not a preregistered campaign like C. The selection rule and the endpoint below
are nevertheless fixed before any result, and the worker applies them mechanically (`tfm_analyze.select_setting`).

## Task and model

- **Task.** Modular addition at p = 31, all 961 pairs, a uniform random 80% training split (768 pairs), cross-entropy on the
  training pairs, full batch. This is the paper's setup.
- **Model.** A decoder-only transformer on the tokens [a, b, =]:
  - vocabulary 32, learned token and position embeddings;
  - d_model 128, 4 heads, a ReLU MLP of width 512;
  - causal attention, no LayerNorm, no biases;
  - logits read at "=".
- **Depth.** 1 or 2 blocks.
- **Precision and code.** float32. The polar steps are computed in float64. The runner is `tfm_runner.py` (numpy). Its tests
  check gradients against finite differences, the last-position shortcut, and bit-identical resume from checkpoints.
- **Initialization.** Embeddings and matrices are N(0, 1)·s/√fan_in, with s the init multiplier; the unembedding is
  N(0, 1)/√d_model.

## Optimizers (arms)

| arm | attention and MLP matrices | embeddings | unembedding |
|---|---|---|---|
| paper | polar step, η 0.03, decoupled decay λ_W | polar step, η 0.03, λ_W | normalized step, η 0.03, λ_V 0.2 |
| hybrid | polar step, η 0.03, λ_W | AdamW | AdamW |
| adamw | AdamW | AdamW | AdamW |

- **Polar step.** M ← M + η·(polar(−G)/√rank − λ·M). This is the paper's feature step: Muon's direction, exact and without
  momentum.
- **AdamW.** lr 1e-3, betas (0.9, 0.98), eps 1e-8, decoupled weight decay wd, as `torch.optim.AdamW`.

## Stage 1: development (tuning seed: init 101, split 201)

- **Settings.** Four per cell (cell = depth × optimizer). Every setting uses init multiplier s ∈ {1, 4}.
  - **paper and hybrid:** λ_W ∈ {0.005, 0.015}.
  - **adamw:** wd ∈ {0.3, 1.0}.
- **Scale.** 6 cells × 4 settings = 24 runs.
- **Length.** 40k steps, stopping once held-out accuracy ≥ 0.99 has held for 1.5× as long as it took to get there (at least 10k
  more steps).

## Per-run endpoint (campaign C's, unchanged except the knee for arms without the normalized head)

- **Knee.**
  - **paper:** the first heavy step with |W_U| ≥ 0.99/λ_V, as in campaign C.
  - **hybrid and adamw:** the unembedding has no norm cap, so the knee is the first logged step at which every training margin
    is positive (the first fit).
- **Band.** The maximal interval around the post-knee peak of the two-update cycle-mean training CE on which it stays within 3%
  of that peak. It is chosen from the training loss alone.
- **Pass.** All of:
  - the band holds ≥ 75% of the held-out 0.1 → 0.9 rise;
  - final held-out accuracy ≥ 0.99;
  - A_H of the last block's MLP activations at "=" rises by ≥ 0.02 across the band;
  - every logged training margin in the band is positive.
- **Also reported:**
  - the raw-CE range from the first 10% to the first 90% held-out accuracy;
  - the CE level (× log p);
  - t10 and t90;
  - whether the train-only ridge refit (on the final residual stream at "=") reaches 90% before the network.

## Stage 2: fresh seeds

The rule is applied by each fresh-seed chain to the compacted development runs of its cell.

1. **Pick the setting.**
   - Among the development settings that pass, take the one with the smallest raw-CE range over 10% → 90%.
   - If none passes, take the one with the largest band share among the runs with final held-out ≥ 0.99. Ties go to the first
     id in sorted order.
2. **Fresh seeds.** Run 10 fresh initialization/split pairs (init 1001+i, split 2001+i, i = 0..9) at the picked setting, but only
   if that setting passed at the development stage. Otherwise the cell's fresh chains record a skip, and the cell is reported
   from its development runs as a failure of the settings tried.
3. **Report.** Report passes out of 10 with a Wilson interval. Nothing is re-tuned after the fresh runs.
