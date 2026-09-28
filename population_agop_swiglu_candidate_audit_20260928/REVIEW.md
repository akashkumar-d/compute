# Candidate fixed-state audit: pre-execution review

PASS for the proposed `--multipliers 1,2` audit. This review used source inspection, syntax parsing, hashes, saved JSON, and snapshot-time decoding only; no model evaluation, training, or network access occurred.

`SOURCE.diff` exactly matches the current script against the previously reviewed v9 audit. The only changes add the selectable multiplier argument, require at least two distinct increasing integers in 1..4, save those factors, and enumerate their pairwise combinations. With the default factors, comparison ordering changes but the three pairs are identical. The actual numerical calculations are unchanged: block-rate-weighted directions, relative-neuron cap, each order's adaptive step, higher-order comparison denominators, AGOP eigenspaces, and projector/operator distances. No parameter update is applied. Output-exists refusal and before/after input/source hashing remain intact.

The frozen selection contains exactly the completed order-2 seed-642 states:

| Step | Snapshot index | Time | Role |
|---:|---:|---:|---|
| 358 | 38 | 31.9514934472 | First qualifying saved 5% candidate |
| 381 | 40 | 34.0439307178 | Exact last-valid 5% boundary |

Both steps occur uniquely in raw rows, and decoded snapshot times match. The result receipt declares completion, and its raw JSON/snapshot hashes match. The saved configuration has `(pair,diag,z,x)=(32,96,48,24)`: factors **1/2 therefore mean absolute original-base orders 2/4**, with the higher tuple `(64,192,96,48)`. Reports must retain this distinction.

The README and script correctly limit conclusions to these fixed states. Agreement cannot certify the full trajectory, unsampled states, or population accuracy; any changed candidate threshold must be retained. The one-worker, 180-second timeout, cleanup, priority, and spare-capacity checks are external launch requirements described in the README, not internally enforced by this audit script.

Reviewed script SHA-256: `c46a8b6b60414ea70bdd18ea3f8c96f0ed692ac2314c97111f116b0bc4a21fbf`.
Selection SHA-256: `e0e43311572947ddecdeb6c59fdc1bbf1a9c73e4dacf0698fb224e242561a76f`.
Prior reviewed script SHA-256: `db6b0c886a2fc0c985ef767dc9a4350338db34fa482ed011c376f058c2f11152`.
