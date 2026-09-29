"""swpop.py -- finite-order population Gaussian-quadrature engine for a SwiGLU student.

Student  f(x) = b + (alpha/m) sum_j a_j S(P_j) V_j,   P_j = p_j . xt,  V_j = v_j . xt,  xt = (x, 1),  S = SiLU.
Teacher  y = gamma * sum_i c_i phi_i(x_i)  (u_i = e_i; phi_i from vlab.ACTS or a callable), E y^2 = 1 by construction.
Loss     L = E (y - f)^2, with the intercept b refitted ('refit') or absent ('none').

Everything is an expectation over at most two Gaussian projections after conditioning:
  * pairs (j, k): 2-D Gauss-Hermite over (P_j, P_k); the V's enter polynomially through E[V | P] and Cov(V_j, V_k | P);
  * diagonal (j, j): 1-D Gauss-Hermite over P_j;
  * teacher terms (i, j): x_i by piecewise Gauss-Legendre (splits at kinks), the rest of P_j by Gauss-Hermite.
Vector-valued expectations E[x h(Z)] use Stein's identity E[x h(B^T x + beta)] = B E[grad h].
Gradients evaluate analytic expectation formulas through finite-order quadrature,
with numerical integration and floating-point error (cross-checked in test_swpop).
"""
import math
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

def S0(t):
    return t / (1.0 + np.exp(-t))

def S1(t):
    s = 1.0 / (1.0 + np.exp(-t)); return s * (1.0 + t * (1.0 - s))

def S2(t):
    s = 1.0 / (1.0 + np.exp(-t)); ds = s * (1.0 - s)
    return 2.0 * ds + t * ds * (1.0 - 2.0 * s)

def S012(t):
    """Same expressions as S0/S1/S2; share their identical exp/denominator."""
    denominator = 1.0 + np.exp(-t)
    s = 1.0 / denominator
    ds = s * (1.0 - s)
    return t / denominator, s * (1.0 + t * (1.0 - s)), 2.0 * ds + t * ds * (1.0 - 2.0 * s)

def S01(t):
    """Same expressions as S0/S1 with one exponential evaluation."""
    denominator = 1.0 + np.exp(-t)
    s = 1.0 / denominator
    return t / denominator, s * (1.0 + t * (1.0 - s))

def _gh(n):
    z, w = hermegauss(n); return z, w / w.sum()

def _gl_pieces(breaks, n, L=10.0):
    """Gauss-Legendre nodes/weights for E_{x~N(0,1)}[.] on [-L, L], split at the given breakpoints."""
    pts = sorted(set([-L, L] + [b for b in breaks if -L < b < L]))
    xg, wg = np.polynomial.legendre.leggauss(n)
    X, Wt = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        X.append(0.5 * (b - a) * xg + 0.5 * (a + b)); Wt.append(0.5 * (b - a) * wg)
    X = np.concatenate(X); Wt = np.concatenate(Wt) * np.exp(-0.5 * X * X) / math.sqrt(2 * math.pi)
    return X, Wt

class Teacher:
    def __init__(self, links, c, breaks=None, n_x=12):
        """links: list of r callables (or one callable used for every i); c: weights. gamma normalises E y^2 = 1
        assuming independent coordinates (additive teacher).  x_i quadrature: Gauss-Legendre with n_x nodes on each
        piece of [-10, 10] split at -6,-4,-2,0,2,4,6 and at the given kinks (E h3^2 numerically matches one to about 1e-15 at n_x = 12;
        a single 24-node piece was 2% off). Canonical breadth links provide fixed raw scalar
        normalization moments, independent of n_x; unannotated callables use the inherited
        quadrature moment path. The raw mean is retained in either case."""
        self.r = len(c)
        self.c = np.asarray(c, float)
        self.links = links if isinstance(links, (list, tuple)) else [links] * self.r
        self.xs, self.wx = _gl_pieces(sorted(set(list(breaks or []) + [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])), n_x)
        # Canonical breadth links carry fixed raw scalar moments, so n_x refinement
        # changes integration accuracy without changing the target function itself.
        # Unannotated callables retain the inherited runtime-quadrature path.
        self.quadrature_raw_mean = np.array([np.sum(self.wx * f(self.xs)) for f in self.links])
        self.quadrature_raw_second = np.array([np.sum(self.wx * f(self.xs) ** 2) for f in self.links])
        m1 = np.array([f.gaussian_moments[0] if hasattr(f, 'gaussian_moments') else self.quadrature_raw_mean[i]
                       for i, f in enumerate(self.links)])
        m2 = np.array([f.gaussian_moments[1] if hasattr(f, 'gaussian_moments') else self.quadrature_raw_second[i]
                       for i, f in enumerate(self.links)])
        self.raw_mean, self.raw_second = m1, m2
        self.normalization_sources = [getattr(f, 'moment_source', 'runtime scalar quadrature') for f in self.links]
        ey2 = np.sum(self.c ** 2 * (m2 - m1 ** 2)) + np.sum(self.c * m1) ** 2
        if not np.isfinite(ey2) or ey2 <= 0:
            raise ValueError('Teacher must have finite positive raw second moment')
        self.gamma = 1.0 / math.sqrt(ey2)
        self.EY = self.gamma * float(np.sum(self.c * m1))
        self.V = 1.0 - self.EY ** 2
        self.phix = np.array([f(self.xs) for f in self.links])     # (r, nx)

class SwiGLUPop:
    def __init__(self, d, teacher, alpha=1.0, intercept="refit", bias=True, n_pair=16, n_diag=48, n_z=24):
        self.d, self.T, self.alpha, self.intercept, self.bias = d, teacher, alpha, intercept, bias
        self.zp, self.wp = _gh(n_pair)
        Z1, Z2 = np.meshgrid(self.zp, self.zp, indexing="ij")
        self.Z1, self.Z2 = Z1.ravel(), Z2.ravel()
        self.W2 = np.outer(self.wp, self.wp).ravel()
        self.zd, self.wd = _gh(n_diag)
        self.zz, self.wz = _gh(n_z)

    # ------------------------------------------------------------------ building blocks
    def _split(self, P, V):
        d = self.d
        return P[:d], P[d], V[:d], V[d]

    def diag_terms(self, pi, p0, nu, v0):
        """1-D expectations for every neuron j (arrays of shape (m,))."""
        spp = np.einsum("dj,dj->j", pi, pi); svp = np.einsum("dj,dj->j", nu, pi); svv = np.einsum("dj,dj->j", nu, nu)
        sp = np.sqrt(np.maximum(spp, 1e-300))
        P = p0[:, None] + sp[:, None] * self.zd[None, :]
        kap = svp / sp                                        # E[V|P] = v0 + kap z
        cvar = np.maximum(svv - kap ** 2, 0.0)
        EV = v0[:, None] + kap[:, None] * self.zd[None, :]
        EV2 = EV ** 2 + cvar[:, None]
        s0, s1, s2 = S012(P)
        w = self.wd[None, :]
        E = lambda A: np.sum(w * A, axis=1)
        return dict(mu=E(s0 * EV), ES1=E(s1), ES0=E(s0), ES2V=E(s2 * EV), ES1V=E(s1 * EV),
                    G=E(s0 * s0 * EV2), A_pp=E((s1 * s1 + s0 * s2) * EV2), A_pv=E(2 * s0 * s1 * EV), A_pb=E(s0 * s1 * EV2),
                    B_vp=E(2 * s0 * s1 * EV), B_vv=E(s0 * s0), B_vb=E(s0 * s0 * EV),
                    g11=E(s1 * s1 * EV2), g10=E(s1 * s0 * EV), g00=E(s0 * s0))

    def pair_terms(self, pi, p0, nu, v0, rows=None):
        """2-D expectations for ordered pairs (j, k), j in rows. Returns dict of (len(rows), m) arrays."""
        m = pi.shape[1]
        rows = np.arange(m) if rows is None else rows
        Gpp = pi.T @ pi; Gvp = nu.T @ pi; Gvv = nu.T @ nu              # Gvp[j,k] = nu_j . pi_k
        J = rows[:, None]; K = np.arange(m)[None, :]
        l11 = np.sqrt(np.maximum(Gpp[J, J], 1e-300))
        l21 = Gpp[J, K] / l11
        l22sq = Gpp[K, K] - l21 ** 2
        l22 = np.sqrt(np.maximum(l22sq, 1e-14 * Gpp[K, K] + 1e-300))
        # w-vectors: E[V_j | z] = v0_j + wj1 z1 + wj2 z2 ; E[V_k | z] = v0_k + wk1 z1 + wk2 z2
        wj1 = Gvp[J, J] / l11; wj2 = (Gvp[J, K] - l21 * wj1) / l22
        wk1 = Gvp[K, J] / l11; wk2 = (Gvp[K, K] - l21 * wk1) / l22
        cjk = Gvv[J, K] - (wj1 * wk1 + wj2 * wk2)
        Z1, Z2, W2 = self.Z1, self.Z2, self.W2
        Pj = p0[J][..., None] + l11[..., None] * Z1
        Pk = p0[K][..., None] + l21[..., None] * Z1 + l22[..., None] * Z2
        EVj = v0[J][..., None] + wj1[..., None] * Z1 + wj2[..., None] * Z2
        EVk = v0[K][..., None] + wk1[..., None] * Z1 + wk2[..., None] * Z2
        EVV = EVj * EVk + cjk[..., None]
        sj0, sj1, sj2 = S012(Pj)
        sk0, sk1 = S01(Pk)
        E = lambda A: np.einsum("jkn,n->jk", A, W2)
        return dict(G=E(sj0 * sk0 * EVV),
                    pSSVV2=E(sj2 * sk0 * EVV), p11VV=E(sj1 * sk1 * EVV), p10Vk=E(sj1 * sk0 * EVk), p10Vj=E(sj1 * sk0 * EVj),
                    p10VV=E(sj1 * sk0 * EVV),
                    v10Vk=E(sj1 * sk0 * EVk), v01Vk=E(sj0 * sk1 * EVk), v00=E(sj0 * sk0), v00Vk=E(sj0 * sk0 * EVk))

    def teacher_terms(self, pi, p0, nu, v0):
        """t_j = E[y g_j] and the vector expectations for its gradients."""
        T = self.T; d = self.d; m = pi.shape[1]; r = T.r
        spp = np.einsum("dj,dj->j", pi, pi); svp = np.einsum("dj,dj->j", nu, pi)
        xs, wx = T.xs, T.wx; zz, wz = self.zz, self.wz
        t = np.zeros(m)
        gp = np.zeros((d + 1, m)); gv = np.zeros((d + 1, m))
        for i in range(r):
            ci = T.gamma * T.c[i]; phi = T.phix[i]                     # (nx,)
            pii, nui = pi[i], nu[i]                                   # (m,)
            sres = np.sqrt(np.maximum(spp - pii ** 2, 1e-300))        # sd of the rest of P given x_i
            kap = (svp - nui * pii) / sres                            # E[V|x_i, z] = nu_i x_i + v0 + kap z
            X = xs[None, :, None]; Zr = zz[None, None, :]
            P = pii[:, None, None] * X + p0[:, None, None] + sres[:, None, None] * Zr
            EV = nui[:, None, None] * X + v0[:, None, None] + kap[:, None, None] * Zr
            w = (wx[:, None] * wz[None, :])[None]
            ph = phi[None, :, None]
            s0, s1, s2 = S012(P)
            E = lambda A: np.sum(w * A, axis=(1, 2))
            t += ci * E(ph * s0 * EV)
            # grad wrt p_j : x-part = e_i E[x_i phi S1 V] + pi_{j,-i} E[phi S2 V] + nu_{j,-i} E[phi S1]; bias E[phi S1 V]
            a_x = E(ph * X * s1 * EV); a_P = E(ph * s2 * EV); a_V = E(ph * s1); a_b = E(ph * s1 * EV)
            gp[:d] += ci * (pi * a_P[None, :] + nu * a_V[None, :])
            gp[i] += ci * (a_x - pii * a_P - nui * a_V)
            gp[d] += ci * a_b
            # grad wrt v_j : x-part = e_i E[x_i phi S0] + pi_{j,-i} E[phi S1]; bias E[phi S0]
            b_x = E(ph * X * s0); b_P = a_V; b_b = E(ph * s0)
            gv[:d] += ci * (pi * b_P[None, :])
            gv[i] += ci * (b_x - pii * b_P)
            gv[d] += ci * b_b
        return t, gp, gv

    # ------------------------------------------------------------------ loss, gradient, metrics
    def evaluate(self, P, V, a, need_grad=True, chunk=32):
        d = self.d; m = P.shape[1]; al = self.alpha / m
        pi, p0, nu, v0 = self._split(P, V)
        if not self.bias:
            p0 = np.zeros(m); v0 = np.zeros(m)
        D = self.diag_terms(pi, p0, nu, v0)
        G = np.zeros((m, m)); Cp = {}
        keys = ("pSSVV2", "p11VV", "p10Vk", "p10Vj", "p10VV", "v10Vk", "v01Vk", "v00", "v00Vk")
        for kk in keys: Cp[kk] = np.zeros((m, m))
        for s in range(0, m, chunk):
            rows = np.arange(s, min(m, s + chunk))
            Pt = self.pair_terms(pi, p0, nu, v0, rows)
            G[rows] = Pt["G"]
            for kk in keys: Cp[kk][rows] = Pt[kk]
        idx = np.arange(m)
        G[idx, idx] = D["G"]
        t, gpt, gvt = self.teacher_terms(pi, p0, nu, v0)
        mu = D["mu"]
        if self.intercept == "refit":
            tb = t - self.T.EY * mu; Gb = G - np.outer(mu, mu); Vy = self.T.V
        else:
            tb = t; Gb = G; Vy = 1.0
        L = Vy - 2 * al * a @ tb + al * al * a @ Gb @ a
        out = dict(L=float(L), t=tb, G=Gb, mu=mu, V=Vy)
        if not need_grad:
            return out
        # gradient wrt a
        ga = -2 * al * tb + 2 * al * al * (Gb @ a)
        # gradient of G_jk wrt p_j (j != k):  pi_j*pSSVV2 + pi_k*p11VV + nu_j*p10Vk + nu_k*p10Vj ; bias p10VV
        #                          wrt v_j (j != k):  pi_j*v10Vk + pi_k*v01Vk + nu_k*v00 ; bias v00Vk
        W = a[None, :] * np.ones((m, 1))                                  # W[j,k] = a_k
        off = np.ones((m, m)) - np.eye(m)
        Wo = W * off
        coefP_pj = (Wo * Cp["pSSVV2"]).sum(1); coefP_pk = Wo * Cp["p11VV"]
        coefP_nj = (Wo * Cp["p10Vk"]).sum(1); coefP_nk = Wo * Cp["p10Vj"]; coefP_b = (Wo * Cp["p10VV"]).sum(1)
        dGp = np.zeros((d + 1, m))
        dGp[:d] = pi * coefP_pj[None, :] + pi @ coefP_pk.T + nu * coefP_nj[None, :] + nu @ coefP_nk.T
        dGp[d] = coefP_b
        coefV_pj = (Wo * Cp["v10Vk"]).sum(1); coefV_pk = Wo * Cp["v01Vk"]; coefV_nk = Wo * Cp["v00"]; coefV_b = (Wo * Cp["v00Vk"]).sum(1)
        dGv = np.zeros((d + 1, m))
        dGv[:d] = pi * coefV_pj[None, :] + pi @ coefV_pk.T + nu @ coefV_nk.T
        dGv[d] = coefV_b
        # diagonal: d G_jj / d p_j = 2 E[S S' V^2 xt] ; d G_jj / d v_j = 2 E[S^2 V xt]
        dGp_diag = np.zeros((d + 1, m)); dGv_diag = np.zeros((d + 1, m))
        dGp_diag[:d] = 2 * (pi * D["A_pp"][None, :] + nu * D["A_pv"][None, :]); dGp_diag[d] = 2 * D["A_pb"]
        dGv_diag[:d] = 2 * (pi * D["B_vp"][None, :] + nu * D["B_vv"][None, :]); dGv_diag[d] = 2 * D["B_vb"]
        # d/dp_j of a^T G a = 2 a_j sum_{k!=j} a_k dG_jk/dp_j + a_j^2 dG_jj/dp_j
        dQp = 2 * a[None, :] * dGp + (a * a)[None, :] * dGp_diag
        dQv = 2 * a[None, :] * dGv + (a * a)[None, :] * dGv_diag
        dtp = gpt; dtv = gvt
        if self.intercept == "refit":
            # mu_j gradients: d mu_j/dp_j = (pi E[S2 V] + nu E[S1]; E[S1 V]),  d mu_j/dv_j = (pi E[S1]; E[S0])
            dmup = np.zeros((d + 1, m)); dmuv = np.zeros((d + 1, m))
            dmup[:d] = pi * D["ES2V"][None, :] + nu * D["ES1"][None, :]; dmup[d] = D["ES1V"]
            dmuv[:d] = pi * D["ES1"][None, :]; dmuv[d] = D["ES0"]
            dtp = dtp - self.T.EY * dmup; dtv = dtv - self.T.EY * dmuv
            s = a @ mu
            dQp = dQp - 2 * s * a[None, :] * dmup
            dQv = dQv - 2 * s * a[None, :] * dmuv
        gP = -2 * al * a[None, :] * dtp + al * al * dQp
        gV = -2 * al * a[None, :] * dtv + al * al * dQv
        if not self.bias:
            gP[d] = 0.0; gV[d] = 0.0
        out.update(gP=gP, gV=gV, ga=ga, D=D, Cp=Cp)
        return out

    def agop(self, P, V, a, ev=None):
        """Population AGOP of f: (alpha/m)^2 sum_jk a_j a_k E[grad g_j grad g_k^T]; W = [pi, nu] with 2m x 2m coefficients."""
        d = self.d; m = P.shape[1]; al = self.alpha / m
        pi, p0, nu, v0 = self._split(P, V)
        if not self.bias:
            p0 = np.zeros(m); v0 = np.zeros(m)
        if ev is None:
            ev = self.evaluate(P, V, a, need_grad=True)
        D, Cp = ev["D"], ev["Cp"]
        # E[grad g_j grad g_k^T] = E[S1j Vj S1k Vk] pi_j pi_k^T + E[S1j Vj S0k] pi_j nu_k^T + E[S0j S1k Vk] nu_j pi_k^T + E[S0j S0k] nu_j nu_k^T
        A11 = Cp["p11VV"].copy(); A10 = Cp["p10Vj"].copy(); A01 = Cp["v01Vk"].copy(); A00 = Cp["v00"].copy()
        idx = np.arange(m)
        A11[idx, idx] = D["g11"]; A10[idx, idx] = D["g10"]; A01[idx, idx] = D["g10"]; A00[idx, idx] = D["g00"]
        aa = np.outer(a, a)
        C = np.block([[aa * A11, aa * A10], [aa * A01, aa * A00]])
        Wm = np.hstack([pi, nu])
        M = al * al * (Wm @ C @ Wm.T)
        return 0.5 * (M + M.T)

    def refit(self, ev, lam_rel=1e-10):
        """Population least-squares refit of (head, intercept) on the frozen SwiGLU features."""
        G, t, Vy = ev["G"], ev["t"], ev["V"]
        lam = lam_rel * max(np.trace(G) / G.shape[0], 1e-300)
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), t)
        return float(Vy - t @ w)
