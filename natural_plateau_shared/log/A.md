# Log: A = claude/plateau-review (Cowork, main account). Append only.

Format: `UTC time | A | task | status | key numbers | where`

2026-09-27T09:55Z | A | setup | created natural_plateau_shared (README, TASKS, code/natplat_server.py, task files) | runner reproduces nat.py exactly, and checkpoint resume is exact | README.md
2026-09-27T09:55Z | A | T0 | running (cloud) | depth 3, r = 0.35 and 0.4, tuning seed, 80k steps; at step 40k of 80k | -
2026-09-27T10:15Z | A | T0 | done | r = 0.35: flat window 15.2k steps, test 0.02→0.36 inside, CE range over grokking 13.4%; r = 0.4: flat window 12.8k, test 0.01→0.25, 13.4% | results/T0_depth3_r0.35_r0.4_tuningseed/SUMMARY.md
2026-09-27T10:15Z | A | setup | moved the shared folder to GitHub (akashkumar-d/compute, branch claude/focused-gauss-cy16t3); README rewritten for the git workflow; T1 revised to r = 0.3 and 0.25 at 120k steps | README.md, TASKS.md
2026-09-27T10:29Z | A | T5 | started (cloud) | depth-3 cascade: (0.5, 0.25) and (0.7, 0.35), tuning seed, 100k steps; runner gained optional per-layer eta_layers/lam_layers keys (old configs give identical numbers) | results/T5_depth3_cascade/
2026-09-27T10:42Z | A | queue | B's order changed to T2 → T3 → T4; T1 on hold until T5 finishes | TASKS.md
2026-09-27T11:20Z | A | T5 | done | cascade (0.7, 0.35): flat window 14.9k steps at 0.48·log p, test 0.36→0.87 inside (63% of the grokking rise; uniform r = 0.35 gives 43%), first-layer AGOP 0.25→0.34; (0.5, 0.25): CE rises 45% during grokking | results/T5_depth3_cascade/SUMMARY.md
2026-09-27T11:21Z | A | T7 | started (cloud) | cascade (0.8, 0.4) and (0.9, 0.45), tuning seed, 100k steps | results/T7_depth3_cascade_tune/
2026-09-27T12:13Z | A | T7 | done | depth-3 cascade (0.8, 0.4): 80% of the grokking rise inside the flat window, CE range over grokking 9.7%; (0.9, 0.45): 82% and 7.0% (uniform r gives 35–38% and 13.4%) | results/T7_depth3_cascade_tune/SUMMARY.md
2026-09-27T12:13Z | A | B's T2 | checked | recomputed from the raw runs: share of the grokking rise inside the flat window 78/98/99%, CE after grokking only about -4.5% by 40k; matches B's SUMMARY | -
2026-09-27T12:14Z | A | T8 | started (cloud) | last-layer-slow rule: depth 3 (1.0, 0.5) and depth 4 (1.0, 1.0, 0.5), tuning seed | results/T8_lastlayer_slow_rule/
2026-09-27T12:14Z | A | queue | B: finish T3, then T6 (depth-3 cascade seeds); T1 and T4 on hold | TASKS.md
2026-09-27T12:26Z | A | B's T3 | checked | p = 47 at depth 2: the whole grokking rise sits inside a 12.6k–14.3k-step flat window at 0.67·log p, CE range 0.9–3.0%, first-layer AGOP 0.13→0.55; p = 23 fails with these settings | results/T3_depth2_other_moduli/SUMMARY.md
2026-09-27T12:26Z | A | queue | B already started T4 before the hold was pushed; keep it as the depth-4 uniform baseline; T6 to start on B's free cores | TASKS.md
2026-09-27T13:05Z | A | T8 | depth 3 done | (1.0, 0.5): 77% of the test 0.1→0.9 rise inside one ≤3% window (17.9k–28.2k steps at 0.50·log p, test 0.06→0.72), CE range over grokking 8.1%, drift −6.0%; close to the tuned cascade (0.9, 0.45) at 82% / 7.0%. Depth 4 still running | results/T8_lastlayer_slow_rule/
2026-09-27T13:05Z | A | T9 | started (cloud) | depth 2, p = 23 with layer-1 decay 0.015, tuning seed and seed 0 | results/T9_depth2_p23_decay0.015/
2026-09-27T13:08Z | A | queue | T10 (depth 3, width 512: cascade (0.9, 0.45) tuning + seed 0, last-layer-slow (1.0, 0.5) tuning; 60k steps) is next for B after T4 and T6 | TASKS.md
