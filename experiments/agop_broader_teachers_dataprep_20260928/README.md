# Broader-teacher AGOP batch on DATA_PREP CPU

Target: scratch-studio-devbox, university-of-california-san-diego/default-project. The user assigned DATA_PREP; a fresh check found 32 CPU cores, Python 3.12.11 and no GPU. The other workstream was explicitly paused, with checkpoints preserved. The earlier candidate chosen-blush-blzp is not the target.

This fresh attempt uses the same 28-arm design and scientific code as broader_teachers_v2. The original queue timed out without training. The old dependency is removed because CPU capacity has been released; no other job or its records are changed. Four single-thread workers and the 55-minute compute cap remain unchanged.

Use the dedicated per-command LRUN_CONFIG recorded in DISPATCH.json/CHECKPOINT.md. Global connection settings remain unchanged. The guarded server entrypoint is dispatch_server.sh with AGOP_EXECUTION_SITE=SERVER. It creates a private environment with pinned NumPy, SciPy and Clarabel and requires a numerical preflight before training. No local training.
