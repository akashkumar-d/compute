# Exact scalar partition batching

`transition_batch.py` implements only:

```python
owners, z, weights = transition_nodes_batch(biases, scale, order=8, limit=12.0)
```

`biases` is a finite one-dimensional array; `scale` is one shared finite nonnegative scalar. The returned arrays are ordered exactly like concatenating `transition_nodes(b, scale, order, limit)` for every bias in input order. `owners` labels each output node with its bias-row index. This interface is for the reference pair rule's inner rules, which have no `extra_gates` argument. It does not replace the outer rule.

Construction uses the same base edges, mapped gate levels, Legendre nodes, affine interval arithmetic, and unnormalized Gaussian weights. Mapped levels outside the interval are clipped to an already present endpoint. Sorting each row then discarding only exactly duplicate edge intervals produces the same ordered scalar partition. A positive but tiny interval is retained even if its half-width underflows. Signed zero is normalized to the +0.0 already present in `BASE_BREAKS`, matching scalar set insertion. Zero scale adds no mapped gate levels. Empty input yields three empty arrays. Inputs are never modified.

The Legendre nodes are already cached by the reference `_legendre`; the helper reuses that implementation and does not invent a second cache or quantize floating parameters. The speedup target is the repeated Python call/set/list construction for each inner bias, not a reduced node count or different quadrature.

The intended caller replacement is:

```python
owners, Z2, inner_weights = transition_nodes_batch(
    p0[k] + b1 * outer, b2, order=quad_order, limit=quad_limit)
Z1 = outer[owners]
weights = inner_weights * wo[owners]
```

Use it only in the existing `b2 != 0` inner-rule branch. The root-owned wrapper retains conditional geometry, rank tolerance, outer partition, stable SiLU derivatives, ordered-pair integrands/reductions, teacher formula and loss/gradient consumers. No original reference source was edited by this helper task.

Eight focused scalar tests passed. They require byte-for-byte equality of node/weight arrays, exact owner indices, unchanged inputs, and byte-for-byte equality of the final packed outer nodes/product weights. Cases include fixed-seed random biases and scales, orders1/8/12/16, sharp transitions, zero scales, repeated biases, signed zero, exact and adjacent mapped/base/tail boundaries, extreme finite arguments with overflow-clipped levels, very small intervals, empty/read-only arrays, and invalid inputs. Tests import only NumPy and scalar-rule modules; no models, gradients or training are evaluated. This is equivalence testing, not a runtime benchmark.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/transition_fast -p test_transition_batch.py -v
```

Next validation belongs to the parent: independently review the wrapper and pinned files; run the same scalar tests on the server's NumPy/platform; then use a bounded fixed-state comparison with identical P/V/a, teacher, integration order/limit, rank tolerance and direction definition. Compare every pair/diagonal/teacher output used by loss/gradient, loss, full gradients, Gram/p11 symmetry and dropped residuals, plus the existing directional consistency screen. Record reference/fast runtimes on the server only. Keep any incomplete or failed comparison; do not infer full-width or trajectory accuracy from these scalar tests.

No model evaluations, benchmarks, dispatch, network access, kernel edits or new scientific success criteria occurred in this helper task.
