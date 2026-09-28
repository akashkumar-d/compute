# Tasks

A owns this file; B asks for changes through `inbox/`. Claim a task by committing and pushing `claims/<TASK>.<A|B>`, where `<TASK>` is the batch-file name without `.json`.

All tasks share these settings:
- Task and model: p = 31 unless stated, cross-entropy (CE), 80% split, bias-free ReLU MLP.
- Every hidden layer starts large: layer 1 uses c24, deeper layers 8 × He.
- Layer 1: step 0.03, decay 0.015. Deeper layers: decay 0.07 and step r · 0.03 · 0.015 / 0.07, so each shrinks at r times layer 1's rate.
- Head: step 0.03, decay 0.2.
- Seeds: the tuning seed is init 2 / split 1; fresh seeds are init 101+s / split 201+s.

Times are rough, for one CPU core per run:
- depth 2, width 256, 40k steps: about 10–15 min
- depth 3, 100k steps: about 1 h
- depth 2, width 512, 40k steps: about 45–60 min

**Order for B (user's request, 2026-09-28 02:15 UTC): resume.** Run, in this order: **T22** now (4 runs, 4 workers); then **T23** once A lists its batch file here (A writes it from T22's results); then **T24** (2 long runs). Push each result as it finishes and a SUMMARY per task. Earlier halt order (19:40 UTC): superseded.

**Delta node (user's request, 01:20 UTC on 2026-09-28): no new tasks there.** T20 and T21, already running, finish; nothing else is assigned to the Delta allocation. After T20 the user stops the push loop on dt-login04 (`pkill -f 'results/T20_depth4'`).

A plans to use a 32-core Lightning AI CPU Studio once the user starts it. The files `code/tasks/L1_<TASK>.json` are the subsets of A's tasks that the Studio runs (batch L1, 25 runs); their results go to `results/<TASK>/` as usual. A second batch, `L2_<TASK>.json` (6 runs), moves A's cloud-session runs of T10 (width-512 cascade, tuning seed and seed 0) and four paused T14 runs to the Studio. They are A's; B should ignore them.

| Task (batch file in `code/tasks/`) | Runs | Why | Est. time per run | Status |
|---|---|---|---|---|
| `T0_depth3_r0.35_r0.4_tuningseed` | depth 3, r = 0.35 and 0.4, tuning seed, 80k steps | Can slower deep layers put grokking inside the depth-3 plateau? | about 40 min | **done (A)**: see below |
| `T2_depth2_width512_seeds` | depth 2, width 512, r = 0.5, fresh seeds 0–2, 40k steps | Does extra width flatten the depth-2 plateau, as it does with one hidden layer (15% → 2% CE range)? | about 45–60 min | **done (B)**: flat window about 14k steps at 0.59·log p, holding 78–99% of the grokking rise; CE range over grokking 3.3–6.9% |
| `T1_depth3_slower_tuningseed` | depth 3, r = 0.3 and 0.25, tuning seed, 120k steps | Continues the T0 trend: slower deep layers give a longer flat window with more of the grokking inside | about 1–1.5 h | **on hold**: T0 gave the same 13.4% CE range at r = 0.35 and 0.4, so a uniform slowdown probably doesn't help |
| `T3_depth2_other_moduli` | depth 2, r = 0.5: p = 47 (tuning seed and seed 0), and p = 23 with layer-1 decay 0.01 (tuning seed and seed 0), 40k steps | Does the plateau carry over to other moduli? | about 15–35 min | **done (B)**: p = 47: flat window 12.6k/14.3k steps at 0.67·log p holding the whole rise (test 0.02→0.99, 0.11→1.00), CE range over grokking 0.9%/3.0%, first-layer AGOP 0.13→0.55. p = 23 (layer-1 decay 0.01): fails (plateau at 0.10·log p, CE falls 36–49%) |
| `T4_depth4_explore` | depth 4, r = 0.35 and 0.5, tuning seed, 120k steps | First look at depth 4. Now serves as the uniform-r baseline for A's depth-4 rule test (T8) | about 1.5–2 h | **done (B)**: uniform r = 0.35 / 0.5 at depth 4: grokking starts late (51.7k / 41.3k steps), after every layer has settled, and the CE falls 14.0% / 15.7% through it; about 30% of the rise fits in one ≤3% window |
| `T5_depth3_cascade` | depth 3, tuning seed, 100k steps: layer 2 at 0.5× and layer 3 at 0.25× layer 1's shrink rate, and layer 2 at 0.7× with layer 3 at 0.35×. Uses the new per-layer `eta_layers` / `lam_layers` keys in the runner. | A cascade: each deeper layer shrinks at half the rate of the one above, so the deepest layer is still shrinking through the long depth-3 grokking | about 50 min | **done (A)**. (0.7, 0.35): the flat window (≤3%) holding the most grokking is 14.9k steps at 0.48·log p, with test 0.36 → 0.87 inside (63% of the 0.1 → 0.9 rise). (0.5, 0.25): the CE rises through grokking. See `results/T5_depth3_cascade/SUMMARY.md` |
| `T7_depth3_cascade_tune` | depth 3 cascade (0.8, 0.4) and (0.9, 0.45), tuning seed, 100k steps | Brackets the balance point: (0.7, 0.35) still lets the CE rise about 8% early in grokking | about 50 min | **done (A)**: (0.8, 0.4) puts 80% of the grokking rise inside the flat window, (0.9, 0.45) 82%; CE range over grokking 9.7% and 7.0%. See the SUMMARY note |
| `T6_depth3_cascade_seeds` | depth 3 cascade (0.9, 0.45), width 256, fresh seeds 0–2, 100k steps | Is the best depth-3 cascade robust across seeds? | about 1–1.2 h | **done (B)**: seeds 0 / 1 / 2 hold 64% / 78% / 99% of the rise in one ≤3% window (tuning seed 82%); CE range over grokking 12.8% / 6.5% / 3.0%; first-layer AGOP inside 0.28→0.37 / 0.23→0.34 / 0.26→0.36 (random about 0.17). The seed that groks earliest (seed 0) has its CE rise, because layer 3 is still shrinking hard then |
| `T8_lastlayer_slow_rule` | depth 3 (1.0, 0.5) at 100k steps and depth 4 (1.0, 1.0, 0.5) at 120k steps, tuning seed | A simpler rule that would work at any depth: every hidden layer shrinks at layer 1's rate except the last, which shrinks at half | about 50–80 min | depth 3 **done (A)**: (1.0, 0.5) puts 77% of the grokking rise in one ≤3% window (the tuned cascade (0.9, 0.45): 82%), CE range over grokking 8.1%, drift −6.0%, first-layer AGOP 0.20→0.27 inside. Depth 4 **done**: the rule fails there (36% of the rise, CE falls 15.9% through grokking); grokking starts at 28.2k steps, after layer 4 has settled. See `results/T8_lastlayer_slow_rule/SUMMARY.md` |
| `T9_depth2_p23_decay0.015` | depth 2, p = 23 with layer-1 decay 0.015 (as at p = 31 and 47), tuning seed and seed 0, 40k steps | T3 used decay 0.01 at p = 23 (the one-hidden-layer value) and failed with a plateau at 0.10·log p | about 10–15 min | **done (A)**: better but still not flat. Plateau at 0.25·log p (was 0.10); the best ≤3% window holds 32% / 44% of the rise; the CE still falls 14–19% through grokking. Layer 2 has finished shrinking before grokking starts at p = 23 (×0.99 over grokking, against ×0.88 at p = 31 and ×0.70 at p = 47) |
| `T11_depth4_slower_last` | depth 4, tuning seed, 120k steps: layers 2 and 3 at layer 1's shrink rate, layer 4 at 0.35× and 0.25× | T8 at depth 4 with the last layer at 0.5×: grokking starts at about 28.5k steps, after layer 4 has settled, and the CE falls through it. A slower last layer should keep shrinking into grokking | about 1.5 h | **done (A)**: last layer at 0.35×: one ≤3% window holds 93% of the rise (test 0.10→0.84) at 0.53·log p, CE range over grokking 5.8%, first-layer AGOP 0.22→0.34 inside; layer 4 shrinks ×0.91 over grokking. At 0.25× the CE rises 50% (layer 4 ×0.47). See `results/T11_depth4_slower_last/SUMMARY.md` |
| `T17_depth4_rule035_seeds` | depth 4 (1.0, 1.0, 0.35), width 256, fresh seeds 1–2, 120k steps | Seeds for the balanced depth-4 setting found in T11 (seed 0 is in T14) | about 1.5 h | seed 1 **done (A, cloud)**: groks at 20.7k steps (tuning seed 26.3k), layer 4 ×0.69 over grokking, CE +17.8% (52% of the rise in one ≤3% window). Seed 0 (in T14, done here): 19.6k, ×0.63, CE +24.5% (32%). Seed 2 **done (Lightning)**: groks at 23.2k, CE +7.4% (78%). **Done** |
| `T18_depth4_seeds_rate045_040` | depth 4 (1.0, 1.0, 0.45) and (1.0, 1.0, 0.4), width 256, fresh seeds 1 and 2, 120k steps | On seed 0, a last layer at 0.45× balances the CE through grokking, while 0.35× balanced the tuning seed and failed on seeds 0–1. Do 0.45 or 0.4 balance seeds 1 and 2 as well? | about 1.5–2 h on B's machine | **done (B)**: at 0.4×, seed 1 holds 95% of the rise in one ≤3% window (CE range 4.5%) and seed 2 87% (5.1%); at 0.45×, seeds 1 and 2 hold 72% and 57% (CE −6.7% and −8.0%). B then halted as asked |
| `T19_depth4_width512` | depth 4 at **width 512**, 60k steps: (1.0, 1.0, 0.35) and (1.0, 1.0, 0.4) on the tuning seed, (1.0, 1.0, 0.45) and (1.0, 1.0, 0.4) on seed 0 | At width 256 the balancing last-layer rate differs by seed (0.35 on the tuning seed, 0.45 on seed 0). Does width make one rate work for both, as width flattened depth 2? | about 3 h on one core | **done (Lightning, 02:03 UTC)**: test 0.1 at 26k-34k steps, grokking unfinished at 60k (final test 0.55-0.87); in all 4 runs the longest <=3% CE window (loss alone) runs from 20.5k-27.1k to the end at 0.82 log p and holds the whole rise so far, at both rates on each seed. Longer runs needed for the end of grokking |
| `T20_depth4_width512_seeds` | depth 4 at width 512, 60k steps, seeds 1 and 2, last layer at 0.4× and 0.45× (layers 2–3 at 1.0×) | Together with T19: does width make one last-layer rate balance every depth-4 seed? | about 3–5 h | **done (Delta, 03:35 UTC)**: like T19, the CE stays within one <=3% window from 20-23k steps to the end at 0.82 log p; grokking unfinished at 60k (final test 0.49-0.66) |
| `T21_depth5_dial` | depth 5 at width 256, 150k steps, layers 2–4 at 1.0×, last layer at 0.25, 0.3, 0.35 and 0.4×, tuning seed and seed 0 | Continues the depth trend (last-layer rate 0.5 / 0.45 / 0.35–0.4 at depth 2 / 3 / 4) and tests the rule registered in log/A.md at 22:50 UTC | about 3–5 h | **done (Delta, 02:20 UTC)**: the registered prediction held (all 8 runs +1.8 to +4.6 points from the curve; run nearest x = 3.85, seed 0 at 0.35x: drift +3.8%, 93% of the rise in one window). Tuning seed balances at 0.3x (+0.7%, 81%) |
| `T22_depth5_recipe_stage1` | depth 5, width 256, 70k steps, layers 2–4 at 1.0×, last layer at 0.35×, fresh seeds 1–4 | Stage 1 of a test of the recipe "set η_L·λ_L ≈ 4/t₁₀" (README §3.6): each seed's t₁₀ at 0.35× gives its own last-layer rate for T23. The four runs also add four depth-5 points to the clock (x = η_L·λ_L·t₁₀ vs CE drift) | about 1.5–2 h (4 runs in parallel) | **for B** (resume order, 02:15 UTC) |
| `T23_depth5_recipe_stage2` | depth 5, seeds 1–4, 70k steps, last layer at each seed's recipe rate from T22: 0.285 / 0.300 / 0.315 / 0.320× | Stage 2: rate set so that x = η_L·λ_L·t₁₀ ≈ 3.85 (slightly rising CE). Predictions registered in log/A.md before the runs | about 1.5 h (4 runs in parallel) | **ready for B** (batch file written 2026-09-28T03:35Z) |
| `T24_depth4_width512_long` | depth 4, **width 512**, 120k steps, last layer 0.4×, tuning seed and seed 0 | T19 at 60k steps stopped mid-grokking (test 0.55–0.87) with the CE flat to the end: does the loss-alone plateau hold the whole rise at depth 4, width 512? | several hours (2 runs, 2 cores each) | for B after T23 |
| `T12_depth2_p23_dial` | depth 2, p = 23, layer-1 decay 0.015, layer 2 at r = 0.4 and 0.35, tuning seed, 40k steps | Same reason as T11: at p = 23 grokking starts after layer 2 has settled | about 10 min | **done (A)**: r = 0.4 balances p = 23 (59% of the rise in one ≤3% window, CE drift +1.9%); r = 0.5 falls 16.5%, r = 0.35 rises 16%. None reaches test 0.9 by 40k steps |
| `T16_depth2_p23_long` | depth 2, p = 23, layer-1 decay 0.015, r = 0.45 and 0.4, tuning seed and seeds 0–1, 80k steps | T12 found the p = 23 balance near r = 0.4, but grokking at p = 23 outlasts 40k steps | about 20 min | **done (A, Lightning)**: grokking at p = 23 is long (test 0.1 → 0.9 over 15–28k steps). The best ≤3% window holds 60–66% of the rise at r = 0.4 (CE range 8.4–14.2%) and 51–66% at r = 0.45 (10–16%). First-layer AGOP stays near its random level |
| `T10_depth3_width512` | depth 3, **width 512**, 60k steps: cascade (0.9, 0.45) on the tuning seed and fresh seed 0, and the last-layer-slow rule (1.0, 0.5) on the tuning seed | Does width flatten the depth-3 plateau, as it did at depth 2? | about 2.5–3.5 h (a 300-step smoke test ran fine) | **done**: cascade tuning seed and seed 0 (A, cloud) and the (1.0, 0.5) tuning seed (Lightning). With T15: in all 7 depth-3 width-512 runs the longest <=3% CE window found from the loss alone is the grokking plateau (24k-28k steps at 0.72-0.73 log p) and holds 79-100% of the test 0.1->0.9 rise |
| `T13_depth3_rule_seeds` | depth 3, last-layer-slow rule (1.0, 0.5), width 256, fresh seeds 0–2, 100k steps | Is the untuned rule robust across seeds? (T6 does the same for the tuned cascade.) | about 1 h | queued (A, Lightning) |
| `T14_depth4_dial` | depth 4, width 256, 120k steps, layers 2–3 at 1.0× layer 1's rate. Last layer at 0.45, 0.4 and 0.3× on the tuning seed; 0.5, 0.45 and 0.4× on seed 0; 0.5× on seeds 1 and 2; plus cascades (1.0, 0.7, 0.35) and (1.0, 0.5, 0.25) on the tuning seed. Seeds 1–2 at 0.45 and 0.4 moved to T18 (B) | Does one fixed last-layer rate balance most seeds? (1.0, 1.0, 0.35) balances the tuning seed but not seeds 0–1, which grok about 6k steps earlier | about 1.5 h | A, cloud. Seed 0: 0.35 done (CE +24.5%); **0.45 balances seed 0** (94% of the rise in one ≤3% window, CE range 4.0%, drift −0.4%, from checkpoint data); 0.5 gives 63% and −7.4%. Tuning seed: 0.45 falls 13.2% (40%), 0.4 falls 8.9% (68%), so the tuning seed needs about 0.35 while seed 0 needs about 0.45; no single rate balances both at width 256. Four runs are paused at 55–60k steps (after grokking); their full 120k-step runs are Lightning batch L2 (running). Lightning batch L1 **done** (23:18–23:28 UTC): tuning 0.3× +17.4% (46%); seed 0 0.4× +8.0% (78%); seeds 1 / 2 0.5× −10.6% / −10.5% (39% / 33%); cascades (1.0, 0.7, 0.35) −3.6% (87%) and (1.0, 0.5, 0.25) +55.8% (8%). Every seed balances where x = η_L·λ_L·t₁₀ ≈ 4 (see log/A.md, 22:50 and 23:36) |
| `T15_depth3_width512_seeds` | depth 3, width 512, 60k steps: cascade (0.9, 0.45) seeds 1–2 and rule (1.0, 0.5) seeds 0–1 | Seeds for T10 | about 2.5–3.5 h | **done (Lightning)**: cascade seeds 1 / 2 hold 89% / 91% of the rise in the loss-alone window, rule seeds 0 / 1 hold 100% / 92%; CE range over grokking 2.1-6.1% across T10 and T15 |

## Results so far

**Depth 2, width 256, r = 0.5, five fresh seeds** (A, report on the user's computer):
- The flat window is 8.5k–9.3k steps at 0.46·log p.
- Inside it, test accuracy goes 0.06 → 0.79 (median) and first-layer AGOP (top-4) goes 0.26 → 0.36.
- The CE range over the grokking window is 4–8%.

**T0 (depth 3, tuning seed; `results/T0_.../SUMMARY.md`).** Slower deep layers lengthen the flat window and pull more of the early grokking into it. Most of the grokking still comes after the window, with the CE falling 13%.

| r | Flat window | Test accuracy inside | CE range over grokking |
|---|---|---|---|
| 0.5 (earlier, 60k steps) | 10.2k | 0.00 → 0.15 | 16% |
| 0.4 | 12.8k | 0.01 → 0.25 | 13% |
| 0.35 | 15.2k | 0.02 → 0.36 | 13% |

## What to report per task

`SUMMARY.md` has one row per run with these columns:

| Column | What it is |
|---|---|
| Flat window | Longest window after the head-norm knee in which the CE range is ≤ 3%, found from the loss alone |
| CE level | Mean CE in the flat window, as a value and as × log p |
| Test accuracy | Test accuracy at the start → end of the flat window |
| Ridge refit | Test accuracy of a ridge readout from the last hidden layer, fit on training pairs only |
| First-layer AGOP | Held-out top-4 AGOP–Fourier alignment of the first hidden layer |
| Last-layer AGOP | Held-out top-30 AGOP–Fourier alignment of the last hidden layer |
| A_H per layer | Fourier energy of each hidden layer |
| Layer-1 weight Fourier share | Share of each first-layer neuron's weight energy in its strongest frequency |
| Grokking window | Steps from test accuracy 0.1 to 0.9 |
| CE range and drift | Over the grokking window |
| Final test | Test accuracy at the last step |

A good outcome: the flat window covers most of the grokking window, and the CE range over the grokking window is ≤ 5%.
