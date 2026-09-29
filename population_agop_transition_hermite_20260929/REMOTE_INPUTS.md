# Runtime inputs remain on Lightning

This code-only bundle intentionally does not publish its three NPZ inputs.
Before running the reviewed probe on the authorized Studio, copy the existing
task-owned inputs from the completed fast diagnostic in the neighboring run
directory. PROBE_MANIFEST.json verifies every copied file before model imports.
Do not regenerate inputs or use approximate matches.

Source run: agop-transition-fast-20260929-a01
Source leaf: population_agop_transition_fast_20260929

Copies:
- probe_inputs/states.npz -> probe_inputs/states.npz
- probe_a01/last_valid_5pct_order16.npz -> reference_saved/last_valid_5pct_order16.npz
- probe_a01/old_bad_step_order16.npz -> reference_saved/old_bad_step_order16.npz

These are read-only source copies within the same user-authorized Studio. They
are not uploaded to GitHub. Original completed outputs are preserved.
