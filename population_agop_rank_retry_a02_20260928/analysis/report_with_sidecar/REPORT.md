# Combined rank report with explicit diagnostic completion

This new derived view preserves the frozen 20-original/12-retry attempt rule and all original training/process receipts. The separate reviewed rank-two quadratic SwiGLU seed642 sidecar supplies only missing frozen-state diagnostics through an explicit in-memory overlay. No result.json or synthetic training/process receipt is written. The original attempt summary and pre-sidecar report_a02 remain unchanged.

All269 original saved checkpoints have successful diagnostics: five original records preserved with exact types/float-bit values and264 newly computed records. Original snapshot shapes, timestamps, input hashes and configuration match exactly. No training updates or new independent seeds are introduced. The other31 combined arm records remain verbatim.

The unchanged canonical metric reader and criteria evaluator recompute only the target arm from the original trajectory and verified diagnostic list. Its original completion_record, training_record, process_state, process_receipt, configuration and full training history remain exact. The old active_snapshot process label is a historical original receipt; it does not describe the completed diagnostic sidecar or current liveness.

| Initial tolerance | Qualified complete sequences, all32 arms |
|---|---:|
| 1.01 | 15/32 |
| 1.05 | 15/32 |

Remaining incomplete full diagnostic grids are the two rank-four seed642 retries (quadratic and raw-ReLU teachers). Their gaps, finite unresolved values and wall censoring remain visible. No interpolation crosses missing diagnostic slots; rank/cell/clock/normalization rules are unchanged. Medians use the two prescribed seeds on common finite support, with ranges rather than confidence intervals.

The six PNG/SVG sheets and two PDFs are in plots/. Numerical/metadata QA is saved in PLOT_QA.json, with full input/output hashes in PLOT_PROVENANCE.json. TARGET_PREFIX_COMPARISON.json records any newly identified earlier canonical candidate without hiding the original partial-grid assessment. Independent and visual reviews remain separate.
