# Campaign F: ten seeds at the settings of the campaign-D and depth-2 figures (plan fixed before any run)

Written by claude/plateau-review on 2026-09-30, before any run of this campaign.
- **Requested by** Akash.
- **Design** by claude/1hl-figures (`_HUB/inbox/2026-09-30_claude-1hl-figures-to-plateau-review_D-10-seed-request.md`).
- **Addition.** Akash added depth 2 at p = 23, 31 and 47.

## Purpose

The paper's trajectory figures should show each run and the median over about 10 seeds. Campaign D ran 1 or 2 seeds per setting,
and the depth-2 development runs have 1 to 5 seeds. This campaign runs new seeds at exactly the settings those figures use, so that
every setting has 10. Nothing else changes.

## The settings (locked here)

They are the settings the reference figures already use (the draft's `experiments/campaigns_CDE_20260930/`). They were chosen
from runs that already exist:
- **Campaign D.** At each p, the setting with the highest mean share over seeds 1 and 2. Ties go to more passes, then to higher
  mean final held-out accuracy. For width 2048 at p = 97, the better of its two settings (one seed each).
- **Depth 2 (development runs).**
  - p = 31 and 47: r = 0.5, tuned on a separate tuning seed.
  - p = 23: r = 0.45 with λ_W 0.015 and 80k updates. r was chosen between 0.4 and 0.45 by the mean share over seeds 0 and 1.

| family | p | width | setting | seeds that exist | new seeds | new runs | horizon | early stop | threads |
|---|---|---|---|---|---|---|---|---|---|
| H: one hidden layer | 23 | 512 | λ_W 0.01 | 1, 2 | 3–10 | 8 | 40k | stop_generalized | 1 |
| H | 31 | 512 | λ_W 0.015 | 1, 2 | 3–10 | 8 | 40k | stop_generalized | 1 |
| H | 47 | 512 | λ_W 0.02 | 1, 2 | 3–10 | 8 | 40k | stop_generalized | 1 |
| H | 61 | 512 | λ_W 0.02 | 1, 2 | 3–10 | 8 | 50k | stop_generalized | 1 |
| H | 97 | 512 | λ_W 0.02 | 1, 2 | 3–10 | 8 | 50k | stop_generalized | 1 |
| HW: one hidden layer, wide | 97 | 2048 | λ_W 0.015 | 1 | 2–10 | 9 | 50k | stop_generalized | 4 |
| D3: depth 3 | 23 | 256 | r 0.45 | 1, 2 | 3–10 | 8 | 120k | stop_generalized | 1 |
| D3 | 47 | 256 | r 0.45 | 1, 2 | 3–10 | 8 | 100k | stop_generalized | 1 |
| D3 | 61 | 256 | r 0.45 | 1, 2 | 3–10 | 8 | 100k | stop_generalized | 1 |
| D3 | 97 | 256 | r 0.45 | 1 | 2–10 | 9 | 100k | stop_generalized | 4 |
| D4: depth 4 | 23 | 256 | r 0.32 | 1, 2 | 3–10 | 8 | 120k | stop_generalized | 1 |
| D4 | 47 | 256 | r 0.45 | 1, 2 | 3–10 | 8 | 100k | stop_generalized | 1 |
| D4 | 61 | 256 | r 0.45 | 1, 2 | 3–10 | 8 | 100k | stop_generalized | 1 |
| D4 | 97 | 256 | r 0.45 | 1 | 2–10 | 9 | 100k | stop_generalized | 4 |
| D2: depth 2 (development) | 23 | 256 | λ_W 0.015, r 0.45 | 0, 1 | 2–9 | 8 | 80k | none | 1 |
| D2 | 31 | 256 | λ_W 0.015, r 0.5 | 0–4 | 5–9 | 5 | 40k | none | 1 |
| D2 | 47 | 256 | λ_W 0.015, r 0.5 | 0 | 1–9 | 9 | 40k | none | 1 |

That is 137 new runs: the 115 of the request and 22 at depth 2.

## How each new run is made

- **Configuration.** The existing seed-1 configuration (campaign D, from `campaign_D/manifest.json`) or seed-0 configuration
  (depth 2, from the development run files). Only `init_seed` and `split_seed` change.
- **Seed rule.** Unchanged.
  - Campaign D: init 5000 + i, split 6000 + i.
  - Depth 2: init 101 + i, split 201 + i.
- **Code.** Campaign D's runner, `natplat_campaign.py`, byte-identical.
  - The depth-2 configurations use none of its optional keys. On them it reproduces the development runner
    (`code/natplat_server.py`) bit for bit, as its docstring and tests state. `CHECKS.md` records this check being done again.
- **Early stop.** As for the existing seeds: `stop_generalized` for the campaign-D settings, and none for depth 2, which run to the
  full horizon.
- **Threads.** As in campaign D: 4 for the p = 97 runs at depths 3 and 4 and at width 2048, 1 otherwise.
- **Run ids.** They follow the existing schemes and repeat none of them. Examples: `H_p23_lw0.0100_s3`, `HW_p97_w2048_lw0.0150_s2`,
  `D3_p97_r0.45_s2`, `seedD2_r0.5_s5`, `T16_D2_p23_lw0.015_r0.45_s2_T80k`, `T3_D2_p47_r0.5_s1`.

## Per-run endpoint

Campaign C's, unchanged (`endpoints.py`, byte-identical). A run passes when all four hold:
- the loss-only peak band holds at least 75% of the held-out 10% → 90% rise;
- the final held-out accuracy is at least 0.99;
- the last hidden layer's A_H rises by at least 0.02 across the band;
- every logged training margin in the band is positive.

## Report

For each setting:
1. **Fresh seeds.** These are the seeds run in this campaign, after this plan was fixed. Report the passes k/n with a Wilson 95%
   interval. n is 8 or 9, and 8, 5 and 9 for depth 2 at p = 23, 31 and 47.
2. **All 10 seeds.**
   - Report k/10 and the medians of the share, CE level, t10, t90 and final held-out accuracy.
   - The older seeds are reported with the new ones but are not fresh. They either picked the setting (campaign D; depth 2 at
     p = 23) or ran before this plan.
3. **Every run's fields** go in `analysis.json`: band, share, pass, failed checks, t10, t90, CE level, final accuracy, ΔA_H,
   first fit, and the rest of campaign C's fields.

Commitments:
- No setting is re-tuned after these runs, and no other setting is run.
- No run is dropped or repeated. The one exception: a run that stops with an error is rerun with the same seed, as the worker does.

## Compute

About 335 core-hours, from the measured durations of the existing seeds. Runs that do not stop early take longer: up to about
2.5 times as long for width 2048 at p = 97.
- It runs on 8 Claude Code sessions × 4 cores and takes about 10–11 hours.
- Each session starts with its 4-thread p = 97 runs.
