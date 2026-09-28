# Secondary loss-only windows

This exploratory analysis applies the existing v6 longest-band rule to the unified saved-data schema for all canonical links and both students. It supplements, and does not replace, the primary initial1%/5% criterion. The rule was already specified in the v6 window protocol; this extension was implemented after preliminary outcomes of23 new runs were known. It is not an independent preregistration or fresh-seed confirmation.

Let V be target variance and R the Hermite energy of degree2 and higher, using the separately recorded rank8 scalar teacher-energy calculation. For each ratio1.01/1.05, select the longest observed optimization-clock interval satisfying the ratio at every update within either [.5R,1.05V] or [.5R,1.05R]. Ties choose earliest start then earliest end. Require50 updates and at least10% of the observed eligible clock. These bands compare loss to constant/affine prediction benchmarks; they do not prove that the network has fitted a particular affine predictor.

Selection uses loss, clock, configuration and teacher moments only. The selected intervals are written to LOSS_ONLY_SELECTION.json before their feature/refit endpoints are compared. A window may begin at initialization; starts_after_initialization explicitly distinguishes genuinely later starts. Two bands or tolerances can identify the same run/window, so window counts must not be reported as numbers of independent successful experiments.

Use the first and last REQUIRED saved checkpoints inside the selected interval, even if their diagnostics are unavailable; do not move an endpoint to a more favorable measured point. Report missing states and require at least50% clock coverage. A secondary candidate needs both endpoint AGOP/refit screens, verified snapshot integrity, a minimum-alignment gain>=.5, and initial-endpoint numerical-lower minus final-endpoint feasible refit risk>=.1V. SwiGLU's cutoff envelope remains a sensitivity measurement rather than a formal optimal-risk improvement bound. Later loss release is separately measured from the window's final diagnostic state. Horizons, quadrature limitations and finite diagnostic grids remain limitations.

```sh
python analysis/later_loss_windows.py --summary analysis/cpu32_SUMMARY.json --energies audit/TEACHER_ENERGIES.json --output analysis/cpu32_later_windows
python -m unittest discover -s analysis -p test_later_loss_windows.py
```

The output directory must be new; no old records are overwritten. Five tests include1000 randomized comparisons with exhaustive interval enumeration, feature-independent selection, fixed endpoint handling, missing integrity, and later-loss arithmetic. No model is evaluated. The first23-run partial analysis found no additional later-start successes: its28 qualifying window entries were duplicate views of seven runs already exhibiting initial-window candidates. All other37 planned runs remained represented as unavailable in that partial download. Wait for the full60-run report before drawing cohort conclusions.
