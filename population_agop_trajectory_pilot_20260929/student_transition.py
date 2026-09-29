"""Reference transition quadrature for SwiGLU student moments.

Prototype only. Import/run on the server for validation, not local training.
The inherited population identities are unchanged. Quadrature is replaced, and
conditional geometry drops residuals below a declared floating-point tolerance.
Finite quadrature and truncated Gaussian tails are not accuracy certificates.
"""
import numpy as np
from swpop import SwiGLUPop
from transition_rule import transition_nodes, silu_derivatives
from teacher_h3 import teacher_terms_h3

PAIR_KEYS = ("G", "pSSVV2", "p11VV", "p10Vk", "p10Vj", "p10VV",
             "v10Vk", "v01Vk", "v00", "v00Vk")


def _mapped_gate(bias, slope):
    # Symmetric transition levels permit this reflection for break placement.
    return (float(bias), float(slope)) if slope >= 0 else (-float(bias), -float(slope))


def _pair_geometry(pj, pk, vj, vk, rank_rtol):
    """Orthogonal conditional coordinates with explicit residual covariance."""
    nj, nk = np.linalg.norm(pj), np.linalg.norm(pk)
    if nj == 0 and nk == 0:
        return (0., 0., 0., 0., 0., 0., 0., float(vj @ vk), 0.)
    e1 = pj/nj if nj > 0 else pk/nk
    b1 = float(pk @ e1)
    residual = pk-b1*e1
    # Reorthogonalize before computing conditional value projections.
    residual -= float(residual @ e1)*e1
    b2 = float(np.linalg.norm(residual))
    rank_cutoff = rank_rtol*max(nj, nk)
    dropped = b2 if b2 <= rank_cutoff else 0.
    e2 = residual/b2 if b2 > rank_cutoff else np.zeros_like(e1)
    if b2 <= rank_cutoff:
        b2 = 0.
    j1, j2 = float(vj @ e1), float(vj @ e2)
    k1, k2 = float(vk @ e1), float(vk @ e2)
    rj, rk = vj-j1*e1-j2*e2, vk-k1*e1-k2*e2
    return (float(nj), b1, b2, j1, j2, k1, k2, float(rj @ rk), dropped)


class TransitionSwiGLUPop(SwiGLUPop):
    """Slow reference implementation; supports only the pure additive h3 teacher."""

    def __init__(self, *args, quad_order=8, quad_limit=12.,
                 rank_rtol=64*np.finfo(float).eps, **kwargs):
        super().__init__(*args, **kwargs)
        # No inference from a filename or approximately matching moments.
        if not all(getattr(link, "terms", None) == ((1.0, "h3"),) for link in self.T.links):
            raise ValueError("This prototype requires canonical pure h3 teacher links")
        if self.T.EY != 0. or not np.isclose(self.T.V, self.T.gamma**2*(self.T.c @ self.T.c),
                                            rtol=1e-14, atol=0):
            raise ValueError("Incompatible pure h3 normalization")
        transition_nodes(0., 0., order=quad_order, limit=quad_limit)
        if not np.isfinite(rank_rtol) or rank_rtol < 0:
            raise ValueError("Nonnegative finite rank tolerance required")
        self.quad_order, self.quad_limit = quad_order, float(quad_limit)
        self.rank_rtol = float(rank_rtol)
        self.max_dropped_gate_residual = 0.

    def _nodes(self, bias, scale, extra_gates=()):
        return transition_nodes(bias, scale, self.quad_order, self.quad_limit, extra_gates)

    def diag_terms(self, pi, p0, nu, v0):
        result = {}
        for j in range(pi.shape[1]):
            scale = float(np.linalg.norm(pi[:, j]))
            e = pi[:, j]/scale if scale > 0 else np.zeros(pi.shape[0])
            kappa = float(nu[:, j] @ e)
            residual = nu[:, j]-kappa*e
            cvar = float(residual @ residual)
            z, weights = self._nodes(p0[j], scale)
            EV = v0[j]+kappa*z
            EV2 = EV**2+cvar
            s0, s1, s2 = silu_derivatives(p0[j]+scale*z, max_order=2)
            values = dict(mu=s0*EV, ES1=s1, ES0=s0, ES2V=s2*EV, ES1V=s1*EV,
                G=s0*s0*EV2, A_pp=(s1*s1+s0*s2)*EV2, A_pv=2*s0*s1*EV,
                A_pb=s0*s1*EV2, B_vp=2*s0*s1*EV, B_vv=s0*s0, B_vb=s0*s0*EV,
                g11=s1*s1*EV2, g10=s1*s0*EV, g00=s0*s0)
            if not result:
                result = {key: np.empty(pi.shape[1]) for key in values}
            for key, value in values.items():
                result[key][j] = weights @ value
        return result

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
                    # Resolve k's gate at each outer coordinate; pack before
                    # evaluating integrands. This is a reference, not optimized.
                    inner_rules = [self._nodes(p0[k]+b1*z, b2) for z in outer]
                    sizes = np.asarray([len(z) for z, _ in inner_rules])
                    Z1 = np.repeat(outer, sizes)
                    Z2 = np.concatenate([z for z, _ in inner_rules])
                    weights = np.concatenate([w*q for (_, w), q in zip(inner_rules, wo)])
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

    def teacher_terms(self, pi, p0, nu, v0):
        return teacher_terms_h3(pi, p0, nu, v0, self.T.c, self.T.gamma,
                               order=self.quad_order, limit=self.quad_limit)
