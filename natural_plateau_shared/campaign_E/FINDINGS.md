# Campaign E: what the transformer runs show (post hoc reading, 2026-09-30)

Written by claude/plateau-review after the results were in. The report that follows the plan is `ANALYSIS.md`, from `analyze.py`.
The plan, the endpoint and the selection rule are in `PLAN.md`; they were fixed before any result. Everything below is descriptive and was not planned. The numbers come from
`describe.py`; its output is `results_archive/describe.json`. The figures are in the draft, `experiments/campaigns_CDE_20260930/`:
`E_overview`, `E_traj` and `E_dev_all`.

## Result under the plan

- **Completion.** 84/84 chains finished and none failed. The 24 development runs took 31.5 core-hours on the eight Claude Code
  sessions (08:03–12:40 UTC).
- **No development run passes** campaign C's per-run endpoint.
  - Every run fails the share, ΔA_H (F02) and fit checks.
  - 14 of the 24 also end below 0.99 held-out accuracy.
- **No fresh seeds.** Every share is 0%, so in each cell the rule's tie-break picked the first setting in sorted order. None of these
  passed, so the rule ran no fresh seeds: all 60 fresh chains recorded a skip, each with the rule's table.

## What the runs show

1. **No loss plateau.** The training set is fit after 60–1,740 updates, and then the training CE falls far below any plateau.
   - While held-out accuracy rises from 10% to 90%, the median two-update mean CE is 2e-8 to 7e-4 × log p across the 24 runs. Its
     90% quantile is at most 0.03 × log p.
   - For comparison, the five MLP runs at p = 31 in `depths_1to5_p31` (depths 1–5) hold 89–100% of their rise on plateaus at
     0.46–0.58 × log p.
2. **The loss is unstable, in two ways.**
   - **Paper's learner.** After the fit, the loss swings by a median of 1.2–5.4 decades from one update to the next. Some training
     pair is misclassified at 0.9–37% of the logged steps.
   - **Hybrid and AdamW.** The loss is smooth between periodic spikes.
     - Each run has 28–65 spikes above 1e-3 × log p. In 14 of these 16 runs they come every 1.1k–1.5k updates.
     - The largest reach 17–36 × log p (hybrid) and 0.08–31 × log p (AdamW).
     - During a spike some training pairs are misclassified and held-out accuracy dips.
3. **Generalization.**
   - Held-out accuracy first reaches 90% after 600–36,300 updates; one run never reaches it.
   - Final held-out accuracy is 0.33–1.00. It is ≥ 0.99 in 10 runs, all hybrid or AdamW.
   - AdamW at wd 1 generalizes about as fast as it fits: with 1 layer, 90% comes after 600 and 1,400 updates.
   - **Delayed rise.** With 2 layers, the paper's learner and init ×4, held-out accuracy stays below 10% for 10k–14k updates after
     the fit. It then rises slowly, to 0.95 at 40k (λ_W 0.005) or 0.33 (λ_W 0.015). This is grokking after memorization, on a
     near-zero, oscillating loss, not on a plateau.
4. **Why the endpoint degenerates here.**
   - **Paper's learner.** The knee is where the head norm reaches 0.99/λ_V = 4.95.
     - The transformer's unembedding starts at norm 5.56 (1 layer) or 5.67 (2 layers), about √31. So the knee is at update 0 and
       the band sits on the initial loss.
     - The head norm then falls to 0.31–0.58 and never returns to the cap.
     - `PLAN.md` fixed both the knee and the initialization before any run, and this mismatch was missed.
     - It does not change the conclusion, because no run has a flat loss during its rise.
   - **Hybrid and AdamW.** The knee is the first fit. The band is then 1–2 updates wide, on the largest post-fit spike.
5. **A possible reason (not tested).**
   - In the MLP, the normalized head sits at its norm cap and the hidden features are bounded. So the logit scale is capped and
     the loss stays on a plateau.
   - Here the head shrinks, and the residual stream, which has no normalization, carries the logit scale (logit RMS 14–261 at the
     end). So the margins grow and the loss collapses.
   - A transformer version would need a bounded feature scale, for example a normalization before the unembedding and a small
     initial head. That would be a new pilot with its own plan, not a re-analysis of E.

## For the paper

Campaign E does not show the regime in transformers. If the paper mentions it, it belongs among the limitations: in a small pilot
(p = 31; 1 and 2 layers; three optimizers; 24 development runs), the training loss did not plateau while held-out accuracy rose.
