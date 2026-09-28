# User-authorized concurrency increase

Continue the previously frozen rank-8 breadth screen on the assigned DATA_PREP CPU, increasing the scheduler limit from four to sixteen. Scientific source, configurations, seeds and individual budgets remain identical. The original absolute 55-minute cutoff is retained.

The controller suspends only the verified original scheduler, lets all original worker groups finish (or reach their existing deadlines), and records their exit status. It then terminates only the idle old scheduler. Its original STATUS remains a historical snapshot, explicitly superseded by HANDOFF.json. The new continuation runs only never-started IDs. Existing outputs are never overwritten.

All source/configuration hashes are checked; the successor is statically validated before termination. Handoff receipts record unresolved consumed IDs separately. The two-seed screen is exploratory; completion is not scientific success. No manuscript changes.

Validation: 17 standard-library scheduler mock tests, including 16 simultaneous slots, disjoint ID coverage, original hard deadline; independent migration-control review. No local training.
