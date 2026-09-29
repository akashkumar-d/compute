# Ten fresh seeds per high-rank cell (v11 publication cohort)

Purpose: publication figures for the aggregated clean high-rank cells and the SwiGLU rank study, with ten seeds per cell. Seeds **9351–9360** were fixed before any outcome. Development seeds 641/642 stay separate and are never pooled. No seed is replaced after its result is seen, and every outcome is kept: capped, failed, unresolved or negative. The case aggregation that selected these cells lives in `goal_followup_v11_highrank_figures/` on the Mac.

## Cells

d = 64 throughout. Every configuration is a reviewed template with only its id/tag/cell and seed changed, except for the SwiGLU recipe changes stated in the table.

| Cell | Template | Declared recipe change | New arms |
|---|---|---|---:|
| ReLU h2, r = 2 / 4 / 8 | v10 rank-sweep configs (m = 256, s = 0.01, h = 0.5, profiled intercept, force horizon 1500, loss stop 0.01) | – | 10 each |
| ReLU h2, r = 16 | same | – | 6 (9355–9360); 9351–9354 reuse the completed identical-recipe confirmation outcomes |
| ReLU \|z\|, r = 8 | v7 CPU32 breadth config (loss stop 0.001 raw) | – | 10 |
| ReLU exp(−z²/2), r = 8 | v7 CPU32 breadth config | – | 10 |
| SwiGLU h2, r = 2 / 4 / 8 / 16 | v10 rank-sweep configs (m = 64, s = 0.1, head lr 0.01, doubled quadrature n_pair = 32, L_stop 0.01) | – | 10 each |
| SwiGLU \|z\|, r = 8 | v10 rank-8 SwiGLU config | link h2 → abs | 10 |
| SwiGLU exp(−z²/2), r = 8 | v10 rank-8 SwiGLU config | link h2 → gaussian_rbf | 10 |
| SwiGLU h2 slower head, r = 8 / 16 | v10 rank-8/16 SwiGLU configs | head_lr 0.01 → 0.001 | 10 each |

That is 136 new arms: 56 ReLU and 80 SwiGLU. The scientific sources (`code/`, `diagnostics/`, `swiglu/`, `TEACHERS.json`) are byte-identical to the reviewed v10 rank-sweep bundle (manifest `e9a0a074…`), and the launcher gate checks this. The success criteria are unchanged: the 1% and 5% all-update initial windows; ΔA_min ≥ 0.5 and refit gain ≥ 0.1 Var(Y) at the same saved state; then a later drop ≥ 0.1 Var(Y).

## Bundles and runtime

The runtime uses the frozen v10 bounds: 28 workers with 4 CPUs reserved, 1990 s per arm and 3600 s per job. Three bundles run in sequence through the launcher's named dependency:

- `bundle_a01`: the 56 ReLU arms first, then 28 long SwiGLU arms (h2 r16, slower-head r16, and h2 r8 seeds 9351–9358). Expected ~2,500 s.
- `bundle_a02`: waits for run `…-a01`. h2 r8 seeds 9359/9360, slower-head r8, |z| r8, and RBF seeds 9351–9356. Expected ≤ 2,000 s.
- `bundle_a03`: waits for run `…-a02`. RBF seeds 9357–9360, h2 r4 and h2 r2. Expected ≤ 2,000 s.

The dependency only makes a02/a03 wait; the wait does not count toward their compute budget. A queue timeout (4,500 s and 9,000 s) is a safe failure: nothing runs, and a manual launch then follows.

## Exact commands

See `REMOTE_RUN.md`. The static checks, run in each bundle, need no scientific import:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

## Interpretation rules

- Analyse with the canonical v10 summarizer on each bundle's `MANIFEST.json` and execution directory. Report 1% and 5% counts per cell out of 10.
- For ReLU h2 r16, the ten seeds are the six new arms plus the four confirmation arms under the identical recipe. State this.
- Medians use all ten seeds on common finite support, and bands are seed ranges or percentiles, not confidence intervals.
- The SwiGLU slower-head and |z|/RBF arms are declared recipe variants. Report them separately from the base rank study.
- Process exit 0 is not scientific success. Numerical candidates are not population certificates.
