# Independent head-scale display-clock review

**PASS, after one input-guard correction.** This preparation review covers `../plot_headscale.py`, the frozen protocol and six-arm manifest. No measured pilot outcomes were read, and no model evaluation, training or network operation was performed.

The adapter retains all six q ∈ {1, 0.3, 0.1} × seed ∈ {641, 642} outcomes in three separate cells. Canonical two-seed aggregates are computed once per q in physical time. Normalized rendering scales copies of aggregate `x` and `support` only; values, medians, ranges, finite-support masks and numerical flags remain byte-identical. There is no pooling across q.

Independent synthetic checks at all three scales verified that `display_cell` changes only copied history/checkpoint `time` fields and prefix `end_time`. Complete candidate objects, including their physical and later times, remain unchanged. Raw losses, refit values, criteria, candidate membership, counts and original input objects are unchanged. Unequal shared support, a missing diagnostic and a finite unresolved value retain their original effects after display scaling. Loss-only zooms use the same canonical prefix boundaries, transformed by q.

The first source version accepted an outer/config seed, teacher-link or rank mismatch. The parent corrected this with explicit seed/link/rank/tag/cell agreement and additional frozen-recipe checks. I independently verified rejection of all five identity mutations using regenerated summaries that match the mutated configs, ensuring rejection comes from the new identity guard rather than an unrelated summary/config mismatch. No unresolved findings remain.

The parent's six focused adapter tests passed according to `CLOCK_TEST_RESULTS.txt`; unchanged tests were not redundantly rerun. Scientific criteria and aggregation/interpolation functions remain in the pinned canonical sources. Captions distinguish physical and normalized clocks, retain two-seed denominators, and state that changing q changes the dynamics and that clock expansion is not evidence of a longer plateau.

This is approval of the display-clock preparation only. Actual initialization matching, downloaded input provenance, observed results, numerical qualification and final visual quality require their separate saved-data checks. Detailed hashes and review records are in `INDEPENDENT_CLOCK_REVIEW.json`.
