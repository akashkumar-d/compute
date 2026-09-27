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

**Order for B: finish T4 (the uniform-r depth-4 baseline) and T6 (depth-3 cascade seeds), both running. Then start T10 (depth 3, width 512) with 3 workers.** Do not start T1.

Why T10: at depth 2, going from width 256 to 512 moved almost all of grokking inside the flat window (T2). At depth 3 and width 256, the best settings still leave the CE rising and falling by 7–8% over grokking (T7, T8).

| Task (batch file in `code/tasks/`) | Runs | Why | Est. time per run | Status |
|---|---|---|---|---|
| `T0_depth3_r0.35_r0.4_tuningseed` | depth 3, r = 0.35 and 0.4, tuning seed, 80k steps | Can slower deep layers put grokking inside the depth-3 plateau? | about 40 min | **done (A)**: see below |
| `T2_depth2_width512_seeds` | depth 2, width 512, r = 0.5, fresh seeds 0–2, 40k steps | Does extra width flatten the depth-2 plateau, as it does with one hidden layer (15% → 2% CE range)? | about 45–60 min | **done (B)**: flat window about 14k steps at 0.59·log p, holding 78–99% of the grokking rise; CE range over grokking 3.3–6.9% |
| `T1_depth3_slower_tuningseed` | depth 3, r = 0.3 and 0.25, tuning seed, 120k steps | Continues the T0 trend: slower deep layers give a longer flat window with more of the grokking inside | about 1–1.5 h | **on hold**: T0 gave the same 13.4% CE range at r = 0.35 and 0.4, so a uniform slowdown probably doesn't help |
| `T3_depth2_other_moduli` | depth 2, r = 0.5: p = 47 (tuning seed and seed 0), and p = 23 with layer-1 decay 0.01 (tuning seed and seed 0), 40k steps | Does the plateau carry over to other moduli? | about 15–35 min | **done (B)**: p = 47: flat window 12.6k/14.3k steps at 0.67·log p holding the whole rise (test 0.02→0.99, 0.11→1.00), CE range over grokking 0.9%/3.0%, first-layer AGOP 0.13→0.55. p = 23 (layer-1 decay 0.01): fails (plateau at 0.10·log p, CE falls 36–49%) |
| `T4_depth4_explore` | depth 4, r = 0.35 and 0.5, tuning seed, 120k steps | First look at depth 4. Now serves as the uniform-r baseline for A's depth-4 rule test (T8) | about 1.5–2 h | running (B, started 12:05 UTC) |
| `T5_depth3_cascade` | depth 3, tuning seed, 100k steps: layer 2 at 0.5× and layer 3 at 0.25× layer 1's shrink rate, and layer 2 at 0.7× with layer 3 at 0.35×. Uses the new per-layer `eta_layers` / `lam_layers` keys in the runner. | A cascade: each deeper layer shrinks at half the rate of the one above, so the deepest layer is still shrinking through the long depth-3 grokking | about 50 min | **done (A)**. (0.7, 0.35): the flat window (≤3%) holding the most grokking is 14.9k steps at 0.48·log p, with test 0.36 → 0.87 inside (63% of the 0.1 → 0.9 rise). (0.5, 0.25): the CE rises through grokking. See `results/T5_depth3_cascade/SUMMARY.md` |
| `T7_depth3_cascade_tune` | depth 3 cascade (0.8, 0.4) and (0.9, 0.45), tuning seed, 100k steps | Brackets the balance point: (0.7, 0.35) still lets the CE rise about 8% early in grokking | about 50 min | **done (A)**: (0.8, 0.4) puts 80% of the grokking rise inside the flat window, (0.9, 0.45) 82%; CE range over grokking 9.7% and 7.0%. See the SUMMARY note |
| `T6_depth3_cascade_seeds` | depth 3 cascade (0.9, 0.45), width 256, fresh seeds 0–2, 100k steps | Is the best depth-3 cascade robust across seeds? | about 1–1.2 h | running (B, started 12:26 UTC) |
| `T8_lastlayer_slow_rule` | depth 3 (1.0, 0.5) at 100k steps and depth 4 (1.0, 1.0, 0.5) at 120k steps, tuning seed | A simpler rule that would work at any depth: every hidden layer shrinks at layer 1's rate except the last, which shrinks at half | about 50–80 min | depth 3 **done (A)**: (1.0, 0.5) puts 77% of the grokking rise in one ≤3% window (the tuned cascade (0.9, 0.45): 82%), CE range over grokking 8.1%, drift −6.0%, first-layer AGOP 0.20→0.27 inside. Depth 4 running (A, cloud) |
| `T9_depth2_p23_decay0.015` | depth 2, p = 23 with layer-1 decay 0.015 (as at p = 31 and 47), tuning seed and seed 0, 40k steps | T3 used decay 0.01 at p = 23 (the one-hidden-layer value) and failed with a plateau at 0.10·log p | about 10–15 min | running (A, cloud, started 13:05 UTC) |
| `T10_depth3_width512` | depth 3, **width 512**, 60k steps: cascade (0.9, 0.45) on the tuning seed and fresh seed 0, and the last-layer-slow rule (1.0, 0.5) on the tuning seed | Does width flatten the depth-3 plateau, as it did at depth 2? | about 2.5–3.5 h (a 300-step smoke test ran fine) | **open: next for B after T4 and T6** |

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
