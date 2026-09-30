# Campaign F: checks before launch (claude/plateau-review, 2026-09-30)

1. **Same code as the existing seeds.**
   - `natplat_campaign.py` and `endpoints.py` are byte-identical to campaign D's (sha256 21c4bd56… and 08c18b7f…).
   - `worker.py` is campaign D's with the campaign name changed (docstrings, version string, commit and status labels).
2. **Same configurations.**
   - For every campaign-D setting, each new run's configuration equals the seed-1 configuration in `campaign_D/manifest.json`
     except `init_seed` and `split_seed` (`make_manifest.py` builds it that way; checked on all 115).
   - The three depth-2 templates equal the seed-0 development configurations, read from the run files in
     `natural_plateau_v1/runs/*.tar.gz`, except the seeds.
3. **Runner tests pass**: all nine of `tests/test_runner.py`. That is campaign D's eight plus a new one: a depth-2 configuration
   with `eta_deep` and none of the optional keys is the development runner (`code/natplat_server.py`) bit for bit.
4. **Full reruns of existing seeds, through this campaign's worker.**
   - **Campaign D, `H_p23_lw0.0100_s1`.** All 27,021 losses and every observation equal campaign D's archived run at the worker's
     7 significant digits. It stopped at the same step (27,020, `generalized`). Only the wall time differs (103 s against 105 s).
   - **Depth 2, `seedD2_r0.5_s0`.** All 40,001 losses and all 2,001 observations equal the development run file at full precision.
     That file's SHA-256, c995417f…, is its entry in `RUN_MANIFEST.tsv`. So on the depth-2 configurations this campaign's code is
     the development code.
5. **End-to-end test on a local git remote.**
   - Setup: two workers on two branches, a miniature manifest with 8 runs in 3 settings (one hidden layer, depth 3 with 2 threads,
     and depth 2 in the development style), and made-up earlier seeds.
   - The workers ran and pushed every run and printed "finished". From a third clone, `analyze.py collect` found all 8.
   - `report` merged them with the earlier seeds of both kinds, counting fresh and all seeds per setting.
6. **Report on the real earlier seeds.** `analyze.py report`, run with no new runs, reads the 25 existing campaign-D seeds from its
   archive and the 8 existing depth-2 seeds from the development archives. Its per-run fields equal campaign D's `analysis.json`
   for all 25: share, pass, failed checks, t10, t90, CE level, final accuracy, ΔA_H, first fit and band.
7. **Worker selftest.** It passes in the cloud container without the git step (python, one worker per clone, the 7 registered
   hashes, the queue, the 9 runner tests, one miniature run). The sessions run it with the git step, which includes a real test
   push.
8. **Balance.** At the measured durations every session has about 10.4 hours of work, each starting with its 4-thread p = 97
   runs. The total is 333 core-hours.
