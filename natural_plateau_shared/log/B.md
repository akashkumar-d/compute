# Log: B = claude/plateau-runner (Claude Code, second account). Append only.

Format: `UTC time | B | task | status | key numbers | where`

2026-09-27T10:30Z | B | T2 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 3 workers | results/T2_depth2_width512_seeds/
2026-09-27T11:32Z | B | T2 | done | flat window 14.1k / 14.3k / 13.9k steps (s0 / s1 / s2) at 0.59·log p; test accuracy inside it 0.27→1.00 / 0.11→0.93 / 0.10→0.89; CE range over grokking 6.9% / 3.3% / 3.3% | results/T2_depth2_width512_seeds/SUMMARY.md
2026-09-27T11:32Z | B | T3 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 3 workers for 4 runs | results/T3_depth2_other_moduli/
2026-09-27T12:04Z | B | T3 | done | p = 47: flat window 12.6k / 14.3k steps (tune / s0) at 0.67 / 0.66·log p, test accuracy inside it 0.02→0.99 / 0.11→1.00, CE range over grokking 0.9% / 3.0%; p = 23 (layer-1 decay 0.01): flat window 4.6k / 5.0k at 0.10·log p, test 0.01→0.21 / 0.01→0.14, CE range over grokking 49% / 36% (tune never holds 0.9; final 0.90) | results/T3_depth2_other_moduli/SUMMARY.md
2026-09-27T12:05Z | B | T4 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 2 workers | results/T4_depth4_explore/
2026-09-27T12:26Z | B | T6 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 2 workers for 3 runs, alongside T4 (2 workers), which the user asked to keep running as the depth-4 baseline | results/T6_depth3_cascade_seeds/
2026-09-27T13:55Z | B | T4 | done | r = 0.35: flat window 17.2k steps (30.7k–47.9k) at 0.61·log p with test 0.00→0.05 inside it; grokking comes later (51.7k→85.3k) with CE range 14.0%; r = 0.5: the longest flat window (13.8k, 106k–120k, 0.44·log p) comes after grokking (41.3k→73.0k), CE range over grokking 15.7% | results/T4_depth4_explore/SUMMARY.md
2026-09-27T14:30Z | B | T6 | done | in all 3 seeds the longest flat window is the post-grokking tail (22.6k–25.0k steps from about 75k, 0.39–0.40·log p, test 1.00); grokking 17.1k→28.8k / 18.6k→34.1k / 19.9k→30.9k (s0 / s1 / s2) with CE range 12.8% / 6.5% / 3.0% and drift +12.0% / +2.2% / +0.2% (A's tuning seed: 7.0%, −3.5%) | results/T6_depth3_cascade_seeds/SUMMARY.md
2026-09-27T14:30Z | B | queue | idle | T2, T3, T4 and T6 done and pushed; stopping as TASKS.md asks | -
2026-09-27T18:14Z | B | T18 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 4 workers | results/T18_depth4_seeds_rate045_040/
2026-09-27T20:39Z | B | T18 | done | in all 4 runs the longest flat window is the post-grokking tail (24.5k–28.0k steps from about 92k–95k, 0.42·log p, test 1.00); grokking on seeds 1 / 2: last layer 0.4× 22.3k→36.5k / 23.9k→39.1k with CE range 4.5% / 5.1% (drift +1.5% / −3.0%); 0.45× 23.1k→38.5k / 24.8k→39.0k with CE range 7.1% / 8.0% (drift −6.7% / −8.0%) | results/T18_depth4_seeds_rate045_040/SUMMARY.md
2026-09-27T20:39Z | B | queue | halted | T18 done and pushed; the user paused B's work on this folder, so B takes no new tasks until the user resumes it | -
2026-09-28T02:19Z | B | queue | resumed | user's request: T22, then T23 once A lists it, then T24 | TASKS.md
2026-09-28T02:19Z | B | T22 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 4 workers, 1 BLAS thread each | results/T22_depth5_recipe_stage1/
