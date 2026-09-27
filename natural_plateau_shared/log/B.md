# Log: B = claude/plateau-runner (Claude Code, second account). Append only.

Format: `UTC time | B | task | status | key numbers | where`

2026-09-27T10:30Z | B | T2 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 3 workers | results/T2_depth2_width512_seeds/
2026-09-27T11:32Z | B | T2 | done | flat window 14.1k / 14.3k / 13.9k steps (s0 / s1 / s2) at 0.59·log p; test accuracy inside it 0.27→1.00 / 0.11→0.93 / 0.10→0.89; CE range over grokking 6.9% / 3.3% / 3.3% | results/T2_depth2_width512_seeds/SUMMARY.md
2026-09-27T11:32Z | B | T3 | started | Claude Code cloud container (Intel Xeon 2.1 GHz, 4 cores, numpy 2.4.6), 3 workers for 4 runs | results/T3_depth2_other_moduli/
2026-09-27T12:04Z | B | T3 | done | p = 47: flat window 12.6k / 14.3k steps (tune / s0) at 0.67 / 0.66·log p, test accuracy inside it 0.02→0.99 / 0.11→1.00, CE range over grokking 0.9% / 3.0%; p = 23 (layer-1 decay 0.01): flat window 4.6k / 5.0k at 0.10·log p, test 0.01→0.21 / 0.01→0.14, CE range over grokking 49% / 36% (tune never holds 0.9; final 0.90) | results/T3_depth2_other_moduli/SUMMARY.md
