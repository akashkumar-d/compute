# Campaign F: ten seeds at the settings of the campaign-D and depth-2 figures

Written by claude/plateau-review on 2026-09-30, at Akash's request.
- **Design.** By claude/1hl-figures: `_HUB/inbox/2026-09-30_claude-1hl-figures-to-plateau-review_D-10-seed-request.md`.
- **Addition.** Akash added depth 2 at p = 23, 31 and 47.
- **Plan.** `PLAN.md` fixes the settings, the seeds, the endpoint and the report before any run.

## Why

The paper's trajectory figures should show each run and the median over about 10 seeds.
- Campaign D ran 1 or 2 seeds per setting.
- The depth-2 development runs have 1 to 5 seeds.

This campaign runs new seeds at exactly the settings those figures use, so that each setting has 10. The learner, the early stop
and the seed rules are unchanged. Only the seed numbers are new.

## What runs

- **137 new runs** in 17 settings:
  - the best campaign-D setting at each p: one hidden layer at p = 23, 31, 47, 61, 97; width 2048 at p = 97; depths 3 and 4 at
    p = 23, 47, 61, 97. That is 115 runs, seeds 3–10, or 2–10 where campaign D ran only seed 1;
  - depth 2 with the development configurations at p = 23, 31, 47. That is 22 runs, up to seed 9.
- **Configurations.** Copied from the existing seed 1 (campaign D) or seed 0 (development runs), with only `init_seed` and
  `split_seed` changed.
- **Runner.** Campaign D's `natplat_campaign.py`, byte-identical. On the depth-2 configurations it is the development runner
  (`code/natplat_server.py`) bit for bit.
- **Compute.** About 335 core-hours at the measured durations of the existing seeds. On 8 Claude Code sessions × 4 cores that is
  about 10–11 hours.
  - Each session starts with its 4-thread p = 97 runs.
  - Runs that do not stop early take longer.

## Files

| file | what |
|---|---|
| `PLAN.md` | the settings, seeds, endpoint and report, fixed before any run |
| `make_manifest.py` → `manifest.json` | the 137 runs: configurations (copied from `campaign_D/manifest.json` and the development run files), seeds, threads, measured durations, the session that owns each; `settings` lists each setting's existing and new run ids |
| `natplat_campaign.py`, `endpoints.py` | campaign D's runner and campaign C's per-run definitions, byte-identical |
| `worker.py` | campaign D's worker with the campaign name changed (claims, work stealing, checkpoints, pushes) |
| `analyze.py` | `collect` (from every session branch) and `report`: every setting at 10 seeds, with the existing seeds read from campaign D's archive and the development archives |
| `tests/test_runner.py` | campaign D's runner tests, plus the depth-2 bit-for-bit check against the development runner |
| `WORKER.md`, `START_PROMPTS.md` | instructions for the sessions and the prompts Akash pastes |
| `HASHES.json` | SHA-256 of the files the workers run; the selftest checks them |
| `CHECKS.md` | what was verified before launch |

## Starting

1. On the Mac, run `bash push_A_files.sh` in `natural_plateau_shared/`.
2. Start eight Claude Code sessions with the prompts in `START_PROMPTS.md`.
3. Keep `akashkumar-d/compute` public until the results are collected; make it private right after.

## Afterwards

claude/plateau-review then does four things:
1. Collects the runs: `python3 analyze.py collect`.
2. Archives them in new tars that repeat no existing id.
3. Writes the report:

       python3 analyze.py report --prior-d ../campaign_D/results_archive \
           --prior-dev ~/Downloads/"modular arithmetic"/claude_plateau_review_20260926/natural_plateau_v1/runs

4. Sends claude/1hl-figures the archives, `analysis.json` (every run's fields for all 10 seeds) and the per-setting fresh-seed
   passes with Wilson 95% intervals.
