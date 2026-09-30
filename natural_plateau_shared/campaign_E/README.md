# Campaign E: transformer pilot (natural CE plateaus in small transformers, p = 31)

Written by claude/plateau-review on 2026-09-30, at Akash's request: "can we have transformers as well?" and "let's do it on Claude
Code". The plan, the endpoint and the selection rule are in `PLAN.md`; they were fixed before any result.

## What runs

- **Cells.** 1- and 2-layer transformers × three optimizers = 6 cells:
  - paper: the paper's learner (polar steps and a normalized head);
  - hybrid: polar steps on the hidden matrices, AdamW on the embeddings and the head;
  - adamw.
- **Development stage.** 4 settings per cell on one tuning seed: 24 runs.
- **Fresh seeds.** 10 per cell: 60 chains.
  - Each waits for its cell's development runs, applies the rule, and trains the chosen setting on its own seeds.
  - If the chosen setting did not pass at the development stage, the chain records a skip instead.
- **Compute.** 8 Claude Code sessions × 4 cores (S1–S8), one core per run. It takes about 4–6 hours in all; the 2-layer
  development runs (about 1.5 h each) are the critical path.

## Files

| file | what |
|---|---|
| `PLAN.md` | design, endpoint and selection rule (fixed first) |
| `tfm_runner.py` | the transformer runner: numpy, the three optimizers, observers, checkpoints |
| `tfm_analyze.py` | per-run fields (campaign C's endpoint with the arm-specific knee) and `select_setting` (the rule) |
| `endpoints.py` | campaign C/D definitions (band, share, helpers), unchanged |
| `worker.py` | the session worker: campaign D's, plus the dependency of fresh chains on their cell's development runs |
| `make_manifest.py`, `manifest.json`, `HASHES.json` | the 84 chains and the registered file hashes |
| `analyze.py` | `collect` (from every session branch) and `report` (ANALYSIS.md, analysis.json) |
| `tests/test_tfm.py` | runner tests, run by the worker's selftest |
| `WORKER.md`, `START_PROMPTS.md` | instructions for the sessions and the prompts Akash pastes |

## Starting

1. On the Mac, run `bash push_A_files.sh` in `natural_plateau_shared/`.
2. Start eight Claude Code sessions with the prompts in `START_PROMPTS.md`.

## Checked before launch

- **Runner tests pass.**
  - Gradients against finite differences: worst relative error 5e-7.
  - The last-position shortcut: 4e-16.
  - Resume from a checkpoint is bit-identical for all three optimizers.
- **The worker's selftest passes** in the cloud container (without the git step). It covers:
  - three miniature development runs;
  - the rule on them;
  - a forced fresh-seed run of the chosen setting.
- **An end-to-end test on a local git remote passed.** It used two workers on two branches and a miniature manifest, and covered:
  - development chains;
  - fresh chains that depend on the other session's development runs;
  - a fresh chain that ran (forced in the test manifest) and one skipped by the rule;
  - pushes to both branches and "finished";
  - then `analyze.py collect` and `report` from a third clone.

  It found one bug, fixed before launch: a skipped chain has no run file, and the commit step expected one.
