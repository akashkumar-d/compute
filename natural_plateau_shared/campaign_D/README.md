# Campaign D: the natural CE plateau across p = 23, 31, 47, 61, 97 (development sweep)

**Owner:** `claude/plateau-review` (session_011vNRtJNdeKePn3Gex5aRg7). Written 2026-09-29 at Akash's request ("vary different
p values and also include the p = 97 case for the theory").

**Status:** ready to run; not started.

## Why

- **Across p.** Campaign C confirmed the natural plateau at depths 3–5 at one modulus, p = 31. Its p = 23 stage failed with the
  p = 31 settings, and depth 2 at p = 47 balanced at a different point. So the settings that give a plateau depend on p.
- **The theorems' prime.** The R91 theorem covers every prime p ≥ 97, at a width of at least p(p² + 3)/2 (456,482 at p = 97).
  This campaign runs the §5/§5b learner at p = 97 with practical widths, iid Gaussian initialization and a random 80% split.
  So it tests the broader empirical claim; it is **not a literal replication**. `codex/modarith-1hl-01a097a9` is building a
  literal R91 replication at p = 97 separately.
- **What it is.** A development sweep. Each family dials the one setting that controls the plateau, at every p, on 1–2 seeds.
  A preregistered confirmation on fresh seeds (like campaign C) can follow once the settings for each p are known.

## Design

The learner is campaign C's.
- Bias-free ReLU MLP on (a + b) mod p, full batch, float64, 80% random split.
- Initialization: c = 24; the deep design uses 8×He deep layers.
- Hidden layers: support-Polar steps with decoupled decay. Head: normalized steps with λ_V = 0.2.
- Every layer trains at every step with fixed hyperparameters.

| family | model | dial | p | seeds | horizon (steps) |
|---|---|---|---|---|---|
| H | 1 hidden layer, width 512 | λ_W ∈ {0.005, 0.0075, 0.01, 0.015, 0.02}; at p = 97 also 0.0025 | 23, 31, 47, 61, 97 | 2 | 40k; 50k at p = 61 and 97 |
| HW | 1 hidden layer at p = 97, width 1024 / 2048 | λ_W ∈ {0.005, 0.01, 0.015} / {0.0075, 0.015} | 97 | 1 | 50k |
| D3 | depth 3, width 256 | last-hidden-layer rate r ∈ {0.2, 0.3, 0.45, 0.65, 0.9} × layer 1's | 23, 47, 61, 97 | 2 (1 at p = 97) | 120k at p = 23; 100k |
| D4 | depth 4, width 256 | r ∈ {0.15, 0.22, 0.32, 0.45, 0.65} | 23, 47, 61, 97 | 2 (1 at p = 97) | the same |

- **Per-p settings.**
  - Deep layers other than the last run at r = 1.
  - The deep runs keep λ_W1 = 0.015 at every p, as campaign C did (including its p = 23 stage).
  - p = 31 has no deep runs here: campaign C covers it with 120 runs, and the development runs cover its dial.
- **Seeds.** Init 5000 + i, split 6000 + i (i = 1, 2): the same seeds at every p and dial value.
- **Early stop.** A run stops once held-out accuracy has stayed ≥ 0.99 from some step s through step max(1.5·s, s + 10,000)
  (`stop_generalized` in `natplat_campaign.py`). The learner is unchanged.
- **Observers.** As campaign C: light observers every 20 steps, heavy ones every 500 (every 1000 at p = 97), protocol-v1
  observers included.
- **Size.** 127 runs, about 270 core-hours at the manifest's estimate (`make_manifest.py` has the per-step timings).
  - That is about 8.3 h per session on 8 sessions × 4 cores.
  - The 12 longest p = 97 runs use all 4 cores of a session, so each takes about 2.5–3.5 h of wall time. They start first.

**Calibration pilots.** Two runs at p = 97 were started in the Cowork container on 2026-09-29, with the p = 31 settings (seed
3001/4001, not a campaign seed):
- **1 hidden layer, width 512, λ_W = 0.015.**
  - It memorizes by about 8k steps while the CE sits near 0.9·log p.
  - Held-out accuracy goes 0.15 → 0.72 → 0.94 at 7k, 8k and 9k steps, and ≥ 0.99 from about 13k.
  - Over that climb the CE moves 0.894 → 0.869·log p, and A_H goes 0.04 → 0.09 (0.12 by 13k).
  - So p = 97 works with the p = 31 setting, at a CE plateau much closer to log p than at p = 31 (0.45·log p). R91's plateau is
    within 0.02 of log p.
- **Depth 3, width 256, rates (1, 0.5).** After memorization the CE rose to 0.90·log p by 11k steps; held-out accuracy was
  still 0 then. Its t₁₀ sets the p = 97 clock.

The 1HL p = 97 grid and horizons above use the first pilot.

## What is measured

Each run gets campaign C's per-run fields (`endpoints.py`, unchanged):
- the peak band, found from the loss alone, and the share of the test 0.1 → 0.9 rise inside it;
- the CE level (× log p) and the raw CE range in the band;
- the drift over grokking;
- t₁₀, t₉₀ and the clock x = η_L·λ_L·t₁₀;
- final accuracies;
- ΔA_H of the first and last hidden layers across the band;
- class-centred AGOP against 8 random frames;
- P10/P03.

"Pass" is campaign C's primary endpoint applied to one run.

`analyze.py report` writes `ANALYSIS.md`:
- **1HL:** a per-p table over the λ_W dial, the best λ_W at each p, and the width effect at p = 97;
- **Deep:** per-p tables over the rate dial, and the clock fitted at each p, drift = a + b·exp(−x), with its balance point x₀(p);
- **Summary:** one table across p.

## Running it

1. On the Mac: `bash push_A_files.sh` in `natural_plateau_shared/`. It now also copies `campaign_D/`.
2. Start 8 Claude Code sessions with the prompt in `START_PROMPTS.md` (IDs S1–S8, repository `akashkumar-d/compute`, branch
   `claude/focused-gauss-cy16t3`).
   - Sessions in campaign C lasted about 8 hours. If they stop before `ALL DONE`, start them again with the same prompts.
   - Finished runs are never repeated.
3. **Progress:** `python3 natural_plateau_shared/campaign_D/worker.py status --all` from any clone.
4. **Analysis:** `python3 analyze.py collect`, then `python3 analyze.py report`. **Keep the repository readable (public) until the
   results are collected**, because `claude/plateau-review` reads it without credentials.

## Files

| file | what it is |
|---|---|
| `make_manifest.py` → `manifest.json` | the 127 runs: configurations, seeds, threads, time estimates, the session that runs each |
| `natplat_campaign.py` | the runner: campaign C's plus `stop_generalized`. It is bit-identical to campaign C's without that key (`tests/test_runner.py`) |
| `endpoints.py` | per-run fields, copied unchanged from campaign C |
| `worker.py` | campaign C's worker (claims, work stealing, standby, pushes), adapted to one run per manifest entry |
| `analyze.py` | `collect`, `report`, `status` |
| `WORKER.md`, `START_PROMPTS.md` | the worker sessions' instructions and the eight prompts |
| `HASHES.json` | SHA-256 of the campaign files; the selftest checks them |
| `results/<run>/`, `status/` | written by the workers on their own branches |
