# Independent source review — student transition prototype

**No formula or API blocker found for a separately approved bounded fixed-state validation. Numerical correctness is not established, and this review does not authorize training.** Source was read only: no imports, model evaluations, network calls or edits to the student prototype.

Reviewed SHA256 pins:

| File | SHA256 |
|---|---|
| student_transition.py | `2cf8bf72461255901a21d38785a21ee5c097566752634c3187943b6b5c52efd7` |
| transition_rule.py | `707aa9f82fb0cbba945dada10b725bcc3644c75b591950d2207fa0e67f5409dc` |
| teacher_h3.py | `ecbb1752b9e97243a82521c62b8ff75fa534a69768d960ad692ac84a7aba6400` |
| ../component_quality/engine/swpop.py | `fa11dd787587be30b8427f6638ee71b164104e159f78b1fe81c19504c84714e7` |

The conditional moment construction is consistent. For an orthonormal gate-span basis, `EVj` and `EVk` are the conditional value means, and `rj @ rk` is the residual cross-covariance. Thus `EVj*EVk+covariance` is the required conditional product. The diagonal routine similarly uses a residual squared norm for conditional variance. All 15 diagonal fields match the inherited consumer, including the diagonal gradient and AGOP terms.

Rank-zero gates retain the full value covariance with constant gates. If only the first gate vanishes, the second gate supplies the first basis vector and the first gate coefficient remains zero. If only the second gate vanishes, its coefficients are zero. Exactly collinear and anticollinear gates reduce to one Gaussian coordinate without dividing by a zero residual; the remaining value components enter covariance. For diagonal pairs, the rank-one construction agrees algebraically with the separate diagonal rule. These are source deductions, not evaluated edge-case results.

All ten ordered-pair integrands preserve the original key orientation. In particular, `p10Vk`/`v10Vk` use the k value, `p10Vj` uses the j value, and `v01Vk` uses the k value with the k gate derivative. The conditional inner gate has bias `p0[k]+b1*z1` and nonnegative residual scale `b2`. Outer extra-gate reflection changes partition placement only; symmetry of the transition-level list makes the negative-slope mapping correct, while the integrand keeps its original signed slope. `rows` produces the required `(len(rows),m)` outputs.

The subclass retains the original evaluated loss, centering, head gradient, diagonal overrides, AGOP and refit consumers. Its teacher override returns correlation gradients in the expected shapes; inherited bias suppression still applies when requested. Imports require the pinned `swpop.py` directory on the server search path.

Two limitations must remain explicit in the next probe:

- Nonzero residuals below `rank_rtol*max(norm(pj),norm(pk))` are deliberately dropped. This is a small geometric approximation **in addition to quadrature**, so the module's “only quadrature is replaced” description is slightly too strong. Record `max_dropped_gate_residual`; do not treat a near-collinear branch as an exact reduction. This is not a blocker for inspecting fixed states.
- Opposite pair orientations use different finite partitions; symmetry and directional derivatives of the evaluated finite rule are not guaranteed. The inherited `2*G@a` head formula presumes the population symmetry. A bounded subset check should therefore include an ordered pair and its reverse, representative diagonal/rank cases, order refinement, and the planned component directional differences. No symmetrization or other kernel change is justified by this source review alone.

The nested rule is a reference implementation and can be substantially more expensive than the old pair rule. Keep the next server check to its declared fixed-state subset and external deadline. Passing that subset would not validate a whole trajectory, tail accuracy, AGOP/refit, or plateau timing.
