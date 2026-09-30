# Campaign E: transformer pilot at p = 31

84/84 chains finished (60 skipped by the rule, 0 failed). Endpoint, knee and selection rule: PLAN.md (fixed before any result).

## Development stage (tuning seed)

| cell | setting | pass | share | CE × log p (band) | raw CE range 10→90% | refit first | t10 → t90 | final test | why not |
|---|---|---|---|---|---|---|---|---|---|
| L1_paper | s1_lw0.005 | no | 0% | 1.00 | 23126474110904.1% | yes | 420 → 39880 | 0.938 | share, final, F02, fit |
| L1_paper | s1_lw0.015 | no | 0% | 1.00 | 301226545.4% | yes | 480 → 39860 | 0.953 | share, final, F02, fit |
| L1_paper | s4_lw0.005 | no | 0% | 54.07 | 298845728779337.3% | yes | 540 → None | 0.886 | share, final, F02, fit |
| L1_paper | s4_lw0.015 | no | 0% | 54.03 | 1008129422.5% | yes | 440 → 39820 | 0.969 | share, final, F02, fit |
| L1_hybrid | s1_lw0.005 | no | 0% | 30.02 | 714509334086.3% | no | 220 → 39080 | 1.000 | share, F02, fit |
| L1_hybrid | s1_lw0.015 | no | 0% | 23.21 | 456754128546.8% | yes | 240 → 39260 | 0.984 | share, final, F02, fit |
| L1_hybrid | s4_lw0.005 | no | 0% | 16.69 | 537999614946.2% | yes | 680 → 37520 | 1.000 | share, F02, fit |
| L1_hybrid | s4_lw0.015 | no | 0% | 17.90 | 91182012937.8% | yes | 540 → 39320 | 0.990 | share, final, F02, fit |
| L1_adamw | s1_wd0.3 | no | 0% | 1.81 | 1284542583.7% | no | 60 → 39220 | 1.000 | share, F02, fit |
| L1_adamw | s1_wd1 | no | 0% | 0.17 | 137301.6% | no | 80 → 600 | 1.000 | share, F02, fit |
| L1_adamw | s4_wd0.3 | no | 0% | 1.74 | 2425779792.1% | no | 460 → 38680 | 1.000 | share, F02, fit |
| L1_adamw | s4_wd1 | no | 0% | 0.08 | 30006196.9% | no | 360 → 1400 | 1.000 | share, F02, fit |
| L2_paper | s1_lw0.005 | no | 0% | 1.01 | 4106606157557751.0% | no | 300 → 39980 | 0.964 | share, final, F02, fit |
| L2_paper | s1_lw0.015 | no | 0% | 1.01 | 4757241593.0% | yes | 320 → 39220 | 0.943 | share, final, F02, fit |
| L2_paper | s4_lw0.005 | no | 0% | 8648.70 | 11026849271690844.0% | no | 11800 → 39980 | 0.948 | share, final, F02, fit |
| L2_paper | s4_lw0.015 | no | 0% | 8634.81 | 49346712359.5% | no | 14800 → None | 0.326 | share, final, F02, fit |
| L2_hybrid | s1_lw0.005 | no | 0% | 34.35 | 2601697460856.2% | no | 180 → 39740 | 0.927 | share, final, F02, fit |
| L2_hybrid | s1_lw0.015 | no | 0% | 24.16 | 994521886425.5% | no | 200 → 39620 | 1.000 | share, F02, fit |
| L2_hybrid | s4_lw0.005 | no | 0% | 36.13 | 1046860197235.4% | no | 2540 → 38620 | 0.969 | share, final, F02, fit |
| L2_hybrid | s4_lw0.015 | no | 0% | 21.12 | 107145654752.4% | no | 2400 → 39700 | 1.000 | share, F02, fit |
| L2_adamw | s1_wd0.3 | no | 0% | 3.65 | 18740110060.2% | no | 40 → 38820 | 1.000 | share, F02, fit |
| L2_adamw | s1_wd1 | no | 0% | 1.14 | 18580114.8% | no | 40 → 39980 | 1.000 | share, F02, fit |
| L2_adamw | s4_wd0.3 | no | 0% | 30.56 | 16554438811.8% | no | 3220 → 39880 | 0.969 | share, final, F02, fit |
| L2_adamw | s4_wd1 | no | 0% | 1.22 | 21363.9% | no | 1260 → 33400 | 0.984 | share, final, F02, fit |

## Chosen setting and fresh seeds

| cell | chosen development setting | passed at development | fresh seeds: pass / run | Wilson 95% |
|---|---|---|---|---|
| L1_paper | dev_L1_paper_s1_lw0.005 | no | skipped by the rule | – |
| L1_hybrid | dev_L1_hybrid_s1_lw0.005 | no | skipped by the rule | – |
| L1_adamw | dev_L1_adamw_s1_wd0.3 | no | skipped by the rule | – |
| L2_paper | dev_L2_paper_s1_lw0.005 | no | skipped by the rule | – |
| L2_hybrid | dev_L2_hybrid_s1_lw0.015 | no | skipped by the rule | – |
| L2_adamw | dev_L2_adamw_s1_wd0.3 | no | skipped by the rule | – |

## Fresh-seed runs

| run | pass | share | CE × log p | raw CE range 10→90% | refit first | t10 → t90 | final test | why not |
|---|---|---|---|---|---|---|---|---|
