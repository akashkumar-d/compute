# Two-arm bundle ready for independent review

`bundle/MANIFEST.json` SHA256: `39ebc788d76e31c0ff031a8a48d6319dd25db1f3dc559f6717aabe7185efeed2`.

Both configurations differ from the completed raw-ReLU/SwiGLU rank-8 controls only in output tag and `head_lr: 0.01 -> 0.001`. Seeds 641/642, head_ratio 1, initial scale, dimensions, widths, quadrature orders, horizons, stop rules and full 1%/5% criteria are exact. Keep both outcomes; these reused seeds are not fresh confirmation.

All 38 frozen files checked. The 11 SwiGLU scientific/support files are byte-identical. Original control configuration hashes, execution snapshot, source hashes, remote inventory and exit-zero receipts match. Saved first P/V/a hashes are recorded; equality to the new runs must be checked after execution. Both controls were wall-capped, so compare curves on common observed support.

The reviewed scheduler's execution, command construction, capacity detection, priority, single-thread environment and cleanup functions are unchanged. The delta limits workers to two, adds the exact paired-design gate and uses a distinct restart lock. Runtime is 2 workers plus 30 reserved slots on at least 32 CPU, 12 GiB available memory and 5 GiB disk; 1800 s training + 180 s diagnostics + 10 s cleanup per arm, 2150 s globally. Static validation and the 24 existing adapted mock checks passed (0.892 s); no scientific code was imported.

`REVIEW.json` remains pending. No model evaluation, training, network, publication or manuscript change occurred. The parent handles independent review, fresh combined capacity check, publication and dispatch. Detailed evidence is in `QA.json`, `DRY_RUN.json`, `MOCK_TESTS.log`, `bundle/CONTROL_PROVENANCE.json`, and `bundle/LAUNCHER.diff`.
