# Campaign C: confirming natural CE plateaus at depths 3–5 on fresh seeds

**Owner:** `claude/plateau-review` (session_011vNRtJNdeKePn3Gex5aRg7).

**Status:** registered 2026-09-29 at 02:17 UTC, before any run. Finished at 18:09 UTC: 164/164 chains, none failed.

**Result:** the primary endpoint is **confirmed at depths 3, 4 and 5**, with 40/40, 39/40 and 38/40 runs passing (37/40
needed). The report is `ANALYSIS.md`; the per-run fields are in `analysis.json`.

**What it tests.** Development (README v1.2) found, at depths 3–5, a training-loss plateau:

- the cross-entropy stays within a few percent while held-out accuracy climbs from 10% to 90%;
- A_H and AGOP alignment rise inside it;
- all layers are trained with fixed hyperparameters;
- the only setting is the last hidden layer's shrink rate, which the registered clock sets from a pilot run.

Campaign C tests this, with preregistered endpoints, on 40 fresh seeds per depth. It also has four secondaries:

- a paired comparison against one fixed rate;
- another modulus (p = 23);
- width 512;
- an optimizer ablation.

## Files

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | design, endpoints, analysis plan, development performance, deviations |
| `START_PROMPTS.md` | the eight session prompts (the only thing the user pastes) |
| `WORKER.md` | exact steps for a worker session |
| `manifest.json` | all 164 chains: configurations, seeds, recipe constants, the session that runs each (`make_manifest.py` writes it) |
| `natplat_campaign.py` | the runner. It is bit-identical to `code/natplat_server.py` without the new keys, and adds pilots that stop at t₁₀, an NGD option and the protocol-v1 observers |
| `recipe.py` | the registered recipe: pilot t₁₀ → last-layer rate |
| `worker.py` | runs a session's chains, pushes results, balances work across sessions; `selftest`, `watch`, `status --all` |
| `endpoints.py`, `analyze.py` | the preregistered per-run fields and the report (`collect`, `report`, `dev`, `status`) |
| `tests/test_runner.py` | runner checks (bit-exact against the development runner, observers read-only, stop-at-t₁₀, NGD) |
| `HASHES.json` | SHA-256 of the registered files; the selftest checks them |
| `ANALYSIS.md`, `analysis.json` | the final report (2026-09-29, 164/164 chains) and every run's preregistered fields |
| `results_archive/` | raw results of all 164 chains, one tar per cohort, collected from the worker branches; not pushed to GitHub |
| `results/<chain>/`, `status/` | written by the workers on their own branches (not on the default branch) |

## Running it

1. On the Mac, run `bash push_A_files.sh` in `natural_plateau_shared/`. This puts this folder on GitHub.
2. Start eight Claude Code sessions with the prompts in `START_PROMPTS.md`.
3. Nothing else. Each session runs about 11 hours:
   - it pushes each finished chain to its own branch;
   - it skips chains finished or running elsewhere;
   - it takes over chains from sessions that stopped or fall far behind;
   - otherwise it stands by, and prints ALL DONE only when every chain of the campaign is finished.

**Progress:** `python3 worker.py status --all`, from any clone.

## Analysis

```
python3 analyze.py collect    # every claude/* branch -> _collected/
python3 analyze.py report     # -> ANALYSIS.md, analysis.json
```
