# Independent batching source review

**No source-level blocker found for the separately bounded fixed-state probe.** Exact manifest `380c88f1359e5ebef8a5061c66c8a7a1f9cae0e3aedf2c32874dafe9019bf88b` is approved in PROBE_REVIEW.json; approval is not authorization for training.

All15 source/input pins match; all11 shared non-probe files are byte-identical to the full-width reference. An AST comparison confirms `pair_terms` is identical outside the nonzero-residual inner-rule packing branch. It preserves geometry, rank handling, outer partitions, all ten ordered integrands and their reductions. Teacher and diagonal formulas remain inherited.

The batch helper reproduces scalar sorted/deduplicated edge sets by clipping exterior gate levels to already-present endpoints and discarding exactly zero-width duplicate intervals. Signed zero is normalized to the scalar base partition's representation. It retains tiny distinct intervals, uses the same interval/node/normal-weight expressions, and enumerates intervals by bias then increasing edge. Its owner indices therefore replace scalar repeat/concatenate without changing node order or outer-weight multiplication. No tolerance, order, normalizing factor or scientific formula is changed.

Reviewed the eight-test source and matching BATCH_QA pins. The existing receipt reports bitwise scalar-node/weight and outer-packing equivalence across ordinary, boundary, degenerate and extreme finite cases; unchanged tests were not rerun. Scalar evidence is not an independent full-model comparison. Compare the fast run's overlapping saved arrays and journal losses with the verified reference after dispatch; report any discrepancy before further use.

The probe changes only class selection and the soft deadline from890 to590 seconds. It retains two original64-neuron states, original alpha/head rate, orders12/16, two FDscales and limit10/12 comparison:22 calls and4 order slots. OneCPU with external TERM595/KILL600 is mandatory. Missing/capped work remains censored; no runtime speedup or numerical success is claimed by source review.

No model imports/evaluations, network calls or dispatch occurred in this review. The helper/wrapper/probe source was compiled without execution.
