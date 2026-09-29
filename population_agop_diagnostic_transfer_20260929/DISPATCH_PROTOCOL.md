# Dispatch specification
This probe calls the unchanged diagnostics.metrics through a cached-evaluation
adapter, so it makes exactly four reference and four candidate model evaluations.
Original diagnostics.py and engine/swsmall.py are copied unchanged and pinned.
All four original stored states472,473,2260,2261 are retained. Reference uses
transition order16/cutoff12; candidate uses Hermite N128/scalar24, inherited
transition diagonal/teacher order16/cutoff12.

Prospective numerical comparisons: loss1e-7Var; actual head-weighted direction
relative1e-4 plus1e-10*m*Var/max(1,theta_norm); common-slope relative1e-4 plus
1e-10*Var/max(1,theta_norm); no materially negative scalar energies. AGOP:
Frobenius relative1e-6 and spectral difference<=1e-3 times reference rank-r gap,
both canonical resolution flags, and min/mean alignment differences<=1e-4.
Report actual projector spectral distance; do not turn unresolved geometry into
a pass. Refit: both canonical four-cutoff screens, equal retained ranks, and
all per-cutoff self-risk and cross-evaluated same-head risk discrepancies<=1e-7
in variance-normalized units. Preserve each failed flag independently.

Cached diagnostics and cross-risk algebra make no model evaluations. Save full
Cp/D, AGOP, equilibrated spectra/masks/weights, raw energy and tail diagnostics,
original states and loss/gradient. No training, line-search or fresh-seed claim.

One CPU; soft350seconds/externalTERM355/KILL360, max8calls. Publish code only:
probe_inputs/states.npz must be copied by pinned hash from the completed Hermite
or fast run already on this same authorized Studio. No binary states to GitHub.
An exit0 means completion, not that comparison flags pass. New training does not
follow automatically. All original results and manuscripts remain preserved.
