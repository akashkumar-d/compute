# Runtime input stays on Lightning

Publish this bundle's reviewed sources and small JSON metadata only. Exclude
`probe_inputs/states.npz` and every output. The manifest intentionally retains
the runtime input's hash.

`prepare_runtime.py` copies the exact saved state array from the completed
`agop-transition-hermite-20260929-a01` run on the same authorized Studio. It checks
the input hash before copying and refuses to overwrite a destination. It does not
regenerate states. `probe.py` checks all 19 pins before model imports.

The launcher uses one single-thread CPU, niceness 10, and external TERM at 355
seconds/KILL at 360 seconds. Run only after the root verifies the authorized
account, allocation and task concurrency. There is no training or automatic
follow-up, and numerical comparison failures remain in the report even at exit 0.
