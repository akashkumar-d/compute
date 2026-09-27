# Log: A = claude/plateau-review (Cowork, main account). Append only.

Format: `UTC time | A | task | status | key numbers | where`

2026-09-27T09:55Z | A | setup | created natural_plateau_shared (README, TASKS, code/natplat_server.py, task files) | runner reproduces nat.py exactly, and checkpoint resume is exact | README.md
2026-09-27T09:55Z | A | T0 | running (cloud) | depth 3, r = 0.35 and 0.4, tuning seed, 80k steps; at step 40k of 80k | -
2026-09-27T10:15Z | A | T0 | done | r = 0.35: flat window 15.2k steps, test 0.02→0.36 inside, CE range over grokking 13.4%; r = 0.4: flat window 12.8k, test 0.01→0.25, 13.4% | results/T0_depth3_r0.35_r0.4_tuningseed/SUMMARY.md
2026-09-27T10:15Z | A | setup | moved the shared folder to GitHub (akashkumar-d/compute, branch claude/focused-gauss-cy16t3); README rewritten for the git workflow; T1 revised to r = 0.3 and 0.25 at 120k steps | README.md, TASKS.md
2026-09-27T10:29Z | A | T5 | started (cloud) | depth-3 cascade: (0.5, 0.25) and (0.7, 0.35), tuning seed, 100k steps; runner gained optional per-layer eta_layers/lam_layers keys (old configs give identical numbers) | results/T5_depth3_cascade/
2026-09-27T10:42Z | A | queue | B's order changed to T2 → T3 → T4; T1 on hold until T5 finishes | TASKS.md
