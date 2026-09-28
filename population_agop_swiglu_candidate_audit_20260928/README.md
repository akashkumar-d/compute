# Fixed-state precision check on the completed order2 trajectory

Compare the actual block-rate-weighted update, capped step and AGOP at its first5%candidate(step358) and exact5%boundary(step381). Multipliers1/2 mean absolute quadrature orders2/4 relative to the original v8 base; no update is applied. This cannot replace trajectory validation or prove population accuracy. The only change to the reviewed v9 audit is explicit selectable increasing multipliers and their pairwise combinations; SOURCE.diff records it. Preserve all audit outcomes, including changed threshold status.

Use the completed original saved order2 files in the existing numerical-validation job; do not copy a different seed or endpoint. Remote execution is single-thread,nice10,180s timeout with5s cleanup, after verifying spare authorized CPU capacity. Source hash guards and output-exists refusal remain intact. Keep inputs and outputs separate. Selection is frozen before evaluations and includes both stated states.
