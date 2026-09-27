# Log: B = claude/plateau-runner (Claude Code, second account). Append only.

Format: `UTC time | B | task | status | key numbers | where`

2026-09-27T10:30Z | B | T2 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 3 workers | results/T2_depth2_width512_seeds/
2026-09-27T11:32Z | B | T2 | done | flat window 14.1k / 14.3k / 13.9k steps (s0 / s1 / s2) at 0.59·log p; test accuracy inside it 0.27→1.00 / 0.11→0.93 / 0.10→0.89; CE range over grokking 6.9% / 3.3% / 3.3% | results/T2_depth2_width512_seeds/SUMMARY.md
2026-09-27T11:32Z | B | T3 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 3 workers for 4 runs | results/T3_depth2_other_moduli/
