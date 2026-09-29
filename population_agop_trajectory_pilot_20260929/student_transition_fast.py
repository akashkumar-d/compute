"""Opt-in batched construction of the identical transition quadrature rule.

Reference teacher/diagonal formulas are inherited unchanged. Only inner pair
partitions are constructed in one batch. No accuracy or runtime claim until
server comparison against the pinned reference succeeds.
"""
import numpy as np
from student_transition import TransitionSwiGLUPop, PAIR_KEYS, _pair_geometry, _mapped_gate
from transition_rule import silu_derivatives
from transition_batch import transition_nodes_batch


class BatchedTransitionSwiGLUPop(TransitionSwiGLUPop):
    def pair_terms(self, pi, p0, nu, v0, rows=None):
        m = pi.shape[1]
        rows = np.arange(m) if rows is None else np.asarray(rows, dtype=int)
        result = {key: np.empty((len(rows), m)) for key in PAIR_KEYS}
        for row, j in enumerate(rows):
            for k in range(m):
                a1, b1, b2, j1, j2, k1, k2, covariance, dropped = _pair_geometry(
                    pi[:, j], pi[:, k], nu[:, j], nu[:, k], self.rank_rtol)
                self.max_dropped_gate_residual = max(self.max_dropped_gate_residual, dropped)
                outer, wo = self._nodes(p0[j], a1, (_mapped_gate(p0[k], b1),))
                if b2 == 0.:
                    Z1, Z2, weights = outer, np.zeros_like(outer), wo
                else:
                    # Same scalar rule and ordering, constructed jointly for all
                    # conditional biases; no change of quadrature nodes.
                    owners, Z2, inner_weights = transition_nodes_batch(
                        p0[k]+b1*outer, b2, self.quad_order, self.quad_limit)
                    Z1 = outer[owners]
                    weights = inner_weights*wo[owners]
                sj0, sj1, sj2 = silu_derivatives(p0[j]+a1*Z1, max_order=2)
                sk0, sk1 = silu_derivatives(p0[k]+b1*Z1+b2*Z2, max_order=1)
                EVj, EVk = v0[j]+j1*Z1+j2*Z2, v0[k]+k1*Z1+k2*Z2
                EVV = EVj*EVk+covariance
                values = (sj0*sk0*EVV, sj2*sk0*EVV, sj1*sk1*EVV, sj1*sk0*EVk,
                          sj1*sk0*EVj, sj1*sk0*EVV, sj1*sk0*EVk,
                          sj0*sk1*EVk, sj0*sk0, sj0*sk0*EVk)
                for key, value in zip(PAIR_KEYS, values):
                    result[key][row, k] = weights @ value
        return result
