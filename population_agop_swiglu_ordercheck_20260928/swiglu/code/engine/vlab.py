"""vlab.py -- variant lab for the G2 setting (one-hidden-layer student, Gaussian inputs, additive teacher).

The defaults reproduce the conventions of the G2 mean-field theorem (polar_general_G2_2026-09-25/); every piece can be
swapped: the teacher link, the student activation and the optimizer.

Model
  X ~ N(0, I_d); teacher directions u_i = e_i (i <= r).
  teacher   Y = gamma * sum_i c_i * phi(x_i)           phi = teacher link, c = weights (balanced by default),
                                                         gamma chosen so that E[Y^2] = 1; V := Var(Y) is reported.
  student   f(x) = b + (alpha/m) * sum_j s_j * sigma(w_j . x)
                                                         fixed head alpha/m, s_j = +1 (G2) or +-1 (--head pm);
                                                         intercept b refitted on the training data at every step.
  loss      L = mean((Y - f)^2) with the refitted b, i.e. the variance of the residual.
  force     F = E[R X sigma'(W^T X)] diag(s),  R = Y - f (centred);  grad_W L = -(2 alpha/m) F.
  init      w_j(0) ~ N(0, sig0^2 I_d / d) IID.

Optimizers (all act on the ascent direction F; --momentum beta replaces F by the buffer B <- beta B + F, except adam)
  polar  W += eta * Polar(F)                     Polar(F) = A B^T from the compact SVD (literal Muon, G2)
  ns     W += eta * NS5(F)                       quintic Newton-Schulz, 5 steps, coefficients (3.4445, -4.7750, 2.0315)
  ngd    W += eta * sqrt(min(d,m)) * F/|F|_F     same Frobenius step as full-rank polar
  sign   W += eta * sqrt(min(d,m)/(d m)) * sign(F)   same Frobenius step as full-rank polar
  gd     W += lr * (2 alpha/m) * F               plain gradient descent on L
  adam   Adam(beta1=--momentum or .9, beta2=.999) on grad_W L with step lr
  If --lr is not given, gd uses lr = eta*sqrt(min(d,m)) / |(2 alpha/m) F(0)|_F and adam uses
  lr = eta*sqrt(min(d,m)/(d m)), so that the first step has the Frobenius size of a full-rank polar step.

Data
  pool   one training pool of n samples, reused at every step (finite sample; population-like for n >> d m / ...)
  fresh  n new samples at every step (stochastic / online)
  exact  exact population formulas (only: teacher relu, student relu, balanced c, head pos); no sampling at all
  --batch b uses a random minibatch of size b from the pool at each step.
  Test metrics use a separate test pool of n_test samples (or the exact formulas in exact mode).

Metrics at each checkpoint (see README.md for definitions)
  test loss / V; train loss / V (every step); AGOP top-r alignment A_r = tr(P_U P_r)/r, cos^2 of the smallest principal
  angle, eigen-gap lambda_r/lambda_{r+1}; the same alignment for the weight second moment W W^T; teacher weight mass;
  per-neuron teacher energy |P_U w_hat_j|^2 (quantiles) and specialization max_i |u_i . w_hat_j|; displacement;
  optional ridge refit of the head on frozen features (--refit) and a random-update control (--control).

Usage
  python3 vlab.py --teacher relu --student tanh --opt polar --r 2 --d 64 --m 256 --T 200 --out run.json
  python3 vlab.py --list            (lists the available links/activations and optimizers)
Python API: cfg = default_config(); cfg.update(...); out = run(cfg).
New activations: add an entry name -> (f, f') to ACTS (vectorised numpy functions).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

try:
    from scipy.special import erf as _scipy_erf
except Exception:  # pragma: no cover
    _scipy_erf = None

SQ2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)
PI = math.pi


# ------------------------------------------------------------------------------------------------ activations / links
def _erf(x):
    if _scipy_erf is not None:
        return _scipy_erf(x)
    # Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7); used only if scipy is missing
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _Phi(z):
    return 0.5 * (1.0 + _erf(z / SQ2))


def _phi(z):
    return np.exp(-0.5 * z * z) / SQ2PI


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _silu_d(z):
    s = _sigmoid(z)
    return s + z * s * (1.0 - s)


def _sigmoid_d(z):
    s = _sigmoid(z)
    return s * (1.0 - s)


ACTS = {
    # name: (function, derivative); derivatives at kinks use the right-continuous convention of the G2 code (strict >)
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(np.float64)),
    "leaky": (lambda z: np.where(z > 0, z, 0.1 * z), lambda z: np.where(z > 0, 1.0, 0.1)),
    "abs": (np.abs, np.sign),
    "softplus": (lambda z: np.logaddexp(0.0, z), _sigmoid),
    "gelu": (lambda z: z * _Phi(z), lambda z: _Phi(z) + z * _phi(z)),
    "silu": (lambda z: z * _sigmoid(z), _silu_d),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "sigmoid": (_sigmoid, _sigmoid_d),
    "quad": (lambda z: z * z, lambda z: 2.0 * z),
    "he2": (lambda z: z * z - 1.0, lambda z: 2.0 * z),
    "he3": (lambda z: z ** 3 - 3.0 * z, lambda z: 3.0 * z * z - 3.0),
    "relu_shift": (lambda z: np.maximum(z - 0.5, 0.0), lambda z: (z > 0.5).astype(np.float64)),
}
# kinks / jumps of each function AND of its derivative (used by the quadrature of data="quad"); smooth ones have none
ACT_BREAKS = {"relu": (0.0,), "leaky": (0.0,), "abs": (0.0,), "relu_shift": (0.5,)}
OPTS = ("polar", "ns", "ngd", "sign", "gd", "adam")
DATAS = ("pool", "fresh", "exact", "quad")


def gauss_moments(fn, lo=-12.0, hi=12.0, npts=480001):
    """E[fn(Z)], E[fn(Z)^2] for Z ~ N(0,1) by the trapezoid rule on [lo, hi] (tails beyond 12 are below 1e-30)."""
    z = np.linspace(lo, hi, npts)
    h = z[1] - z[0]
    w = _phi(z)
    fz = fn(z)

    def trap(v):
        return h * (v.sum() - 0.5 * (v[0] + v[-1]))
    return float(trap(fz * w)), float(trap(fz * fz * w))


def teacher_constants(link, c):
    """gamma (E Y^2 = 1), V = Var Y and E Y for Y = gamma sum_i c_i link(Z_i) with IID standard Z_i."""
    m1, m2 = gauss_moments(ACTS[link][0])
    v1 = m2 - m1 * m1
    c = np.asarray(c, dtype=np.float64)
    ey2 = (c * c).sum() * v1 + (m1 * c.sum()) ** 2
    gam = 1.0 / math.sqrt(ey2)
    V = gam * gam * (c * c).sum() * v1
    EY = gam * m1 * c.sum()
    return gam, float(V), float(EY)


# ------------------------------------------------------------------------------------------------ exact ReLU/ReLU
def _kap(rho):
    """Cov(ReLU(a.X), ReLU(b.X)) for unit a, b with a.b = rho."""
    rho = np.clip(rho, -1.0, 1.0)
    return (np.sqrt(np.maximum(1.0 - rho ** 2, 0.0)) + rho * (PI - np.arccos(rho)) - 1.0) / (2.0 * PI)


def _unit(W):
    nrm = np.linalg.norm(W, axis=0)
    return W / nrm, nrm


def exact_loss(W, r, alpha, gam, V):
    m = W.shape[1]
    Wh, nrm = _unit(W)
    cov = (alpha / m) * np.dot(nrm, gam * _kap(Wh[:r]).sum(0))
    var = (alpha / m) ** 2 * nrm @ _kap(Wh.T @ Wh) @ nrm
    return float(V - 2.0 * cov + var)


def exact_force(W, r, alpha, gam):
    """Exact population force for the balanced ReLU teacher and ReLU student with refitted intercept (G2, eq. (1.5))."""
    d, m = W.shape
    Wh, nrm = _unit(W)
    th = np.arccos(np.clip(Wh[:r], -1.0, 1.0))
    F = np.zeros_like(W)
    F[:r] += (gam / (2 * PI)) * (PI - th)
    F -= (gam / (2 * PI)) * (1.0 - np.sin(th)).sum(0)[None, :] * Wh
    if alpha > 0:
        thjk = np.arccos(np.clip(Wh.T @ Wh, -1.0, 1.0))
        F -= (alpha / (2 * PI * m)) * (Wh @ ((PI - thjk) * nrm[None, :]).T)
        F += (alpha / (2 * PI * m)) * ((1.0 - np.sin(thjk)) * nrm[None, :]).sum(1)[None, :] * Wh
    return F


def exact_agop(W, alpha):
    """Exact population AGOP (alpha/m)^2 W H W^T, H_jk = (pi - theta_jk)/(2 pi), for the ReLU student."""
    m = W.shape[1]
    Wh, _ = _unit(W)
    H = (PI - np.arccos(np.clip(Wh.T @ Wh, -1.0, 1.0))) / (2 * PI)
    M = (alpha / m) ** 2 * (W @ H @ W.T)
    return 0.5 * (M + M.T)


# ------------------------------------------------------------------------------------------------ population by quadrature
# data="quad": every population expectation is a one- or two-dimensional Gaussian integral, computed by Gauss-Legendre
# quadrature on [-9, 9] split at every kink (and, for nearly parallel pairs, around the smoothed kink), so kinked
# activations are integrated to high accuracy (checked against the exact ReLU formulas to ~1e-10 at 32 nodes).
_GL = {}
Z_SPLITS = (-3.0, -1.0, 0.0, 1.0, 3.0)      # fixed splits of every Gaussian coordinate (resolution of the weight)


def _gl(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


def _pieces_nodes(bps, n, L):
    """Nodes/weights for int f(z) phi(z) dz over [-L, L], split at the breakpoints bps (shape (..., K), any order)."""
    lead = bps.shape[:-1]
    edges = np.concatenate([np.full(lead + (1,), -L), np.sort(np.clip(bps, -L, L), axis=-1),
                            np.full(lead + (1,), L)], axis=-1)
    lo, hi = edges[..., :-1], edges[..., 1:]
    x, w = _gl(n)
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    z = mid[..., None] + half[..., None] * x
    wt = half[..., None] * w * _phi(z)
    return z.reshape(lead + (-1,)), wt.reshape(lead + (-1,))


def pair_integrals(G, dG, bG, H, bH, a, b, c, want, n=16, L=9.0, chunk=512):
    """For pairs p (arrays a, b > 0 and c in [-1, 1]), with s = sqrt(1 - c^2), U = a (c Z1 + s Z2), Z1, Z2 ~ N(0,1) IID:
         S = E[G(U) H(b Z1)],   J1 = E[G(U) H(b Z1) Z1],   K = E[G'(U) H(b Z1)].
    These give, for unit a_hat, b_hat with a_hat.b_hat = c and X ~ N(0, I):
         E[G(a a_hat.X) H(b b_hat.X)] = S,   E[G(a a_hat.X) X H(b b_hat.X)] = (J1 - c a K) b_hat + a K a_hat."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 1e-300)
    b = np.maximum(np.asarray(b, dtype=np.float64), 1e-300)
    c = np.clip(np.asarray(c, dtype=np.float64), -1.0, 1.0)
    P = c.shape[0]
    out = {k: np.empty(P) for k in want}
    for lo in range(0, P, chunk):
        sl = slice(lo, min(P, lo + chunk))
        aa, bb, cc = a[sl], b[sl], c[sl]
        ss = np.sqrt(np.maximum(0.0, 1.0 - cc * cc))
        near = (ss < 0.2) & (np.abs(cc) > 1e-12)
        for grp in (near, ~near):
            if not grp.any():
                continue
            ag, bg, cg, sg = aa[grp], bb[grp], cc[grp], ss[grp]
            csafe = np.where(np.abs(cg) > 1e-12, cg, 1.0)
            ob = [np.full(ag.size, z) for z in Z_SPLITS] + [x / bg for x in bH]
            for x in bG:
                zs = np.where(np.abs(cg) > 1e-12, x / (ag * csafe), -L)
                ob.append(zs)
                if grp is near:
                    dl = sg / np.abs(csafe)
                    ob += [zs + k * dl for k in (-8.0, -3.0, -1.0, 1.0, 3.0, 8.0)]
            ob = np.stack(ob, -1)
            z1, w1 = _pieces_nodes(ob, n, L)
            ib = [np.full(z1.shape, z) for z in Z_SPLITS]
            ib += [np.where(sg[:, None] > 1e-12, (x / ag[:, None] - cg[:, None] * z1) / np.maximum(sg[:, None], 1e-300), -L)
                   for x in bG]
            ib = np.stack(ib, -1)
            z2, w2 = _pieces_nodes(ib, n, L)
            U = ag[:, None, None] * (cg[:, None, None] * z1[:, :, None] + sg[:, None, None] * z2)
            Hv = H(bg[:, None] * z1) * w1
            idx = np.arange(lo, min(P, lo + chunk))[grp]
            if "S" in want or "J1" in want:
                i0 = (G(U) * w2).sum(-1)
                if "S" in want:
                    out["S"][idx] = (Hv * i0).sum(-1)
                if "J1" in want:
                    out["J1"][idx] = (Hv * z1 * i0).sum(-1)
            if "K" in want:
                out["K"][idx] = (Hv * (dG(U) * w2).sum(-1)).sum(-1)
    return out


def moments_1d(fn, bps, scales, weight_z=False, n=16, L=9.0):
    """E[fn(s Z)] (or E[Z fn(s Z)]) for each scale s > 0, split at bps/s."""
    scales = np.maximum(np.asarray(scales, dtype=np.float64), 1e-300)
    ob = np.stack([np.full(scales.size, z) for z in Z_SPLITS] + [x / scales for x in bps], -1)
    z, w = _pieces_nodes(ob, n, L)
    v = fn(scales[:, None] * z) * w
    return (v * z).sum(-1) if weight_z else v.sum(-1)


def population_refit(Q, cvec, V, lam_rel=1e-10):
    """min_a E[(Y - EY - sum_j a_j (phi_j - E phi_j))^2] = V - c^T Q^{-1} c (tiny ridge for stability)."""
    lam = lam_rel * max(np.trace(Q) / Q.shape[0], 1e-300)
    a = np.linalg.solve(Q + lam * np.eye(Q.shape[0]), cvec)
    return float(V - 2 * a @ cvec + a @ Q @ a)


def exact_refit(W, r, gam, V):
    """Population least-squares refit of the head on ReLU features (exact formulas), with intercept."""
    Wh, nrm = _unit(W)
    cvec = gam * nrm * _kap(Wh[:r]).sum(0)
    Q = np.outer(nrm, nrm) * _kap(Wh.T @ Wh)
    return population_refit(Q, cvec, V)


class QuadPopulation:
    """Population force, loss and AGOP for any teacher link / student activation, by quadrature (data="quad")."""

    def __init__(self, link, act, c, gam, V, alpha, s, n=16, n_self=8, self_force=True):
        self.link, self.act, self.c, self.gam, self.V, self.alpha, self.s = link, act, np.asarray(c, float), gam, V, alpha, s
        self.n, self.n_self, self.self_force = n, n_self, self_force
        self.ft, self.dft = ACTS[link]
        self.fs, self.dfs = ACTS[act]
        self.bt, self.bs = ACT_BREAKS.get(link, ()), ACT_BREAKS.get(act, ())
        self.mu_phi = gauss_moments(self.ft)[0]

    def _geom(self, W):
        Wh, nrm = _unit(W)
        return Wh, nrm, np.clip(Wh.T @ Wh, -1.0, 1.0)

    def force(self, W):
        d, m = W.shape
        r = self.c.size
        Wh, nrm, C = self._geom(W)
        s, al = self.s, self.alpha
        e1 = moments_1d(self.dfs, self.bs, nrm, weight_z=True, n=self.n)            # E[Z sigma'(|w_j| Z)]
        # teacher-neuron pairs (i, j): a = u_i (norm 1), b = w_j, c = u_i . w_hat_j
        ct = Wh[:r].ravel()
        tp = pair_integrals(self.ft, self.dft, self.bt, self.dfs, self.bs, np.ones_like(ct), np.tile(nrm, r), ct,
                            ("J1", "K"), n=self.n)
        J1, K = tp["J1"].reshape(r, m), tp["K"].reshape(r, m)
        coefB = self.gam * (self.c[:, None] * (J1 - Wh[:r] * K)).sum(0) - self.gam * self.mu_phi * self.c.sum() * e1
        F = Wh * coefB[None, :]
        F[:r] += self.gam * self.c[:, None] * K
        if al > 0 and self.self_force:
            # neuron-neuron pairs (k, j): a = w_k, b = w_j
            mu = moments_1d(self.fs, self.bs, nrm, n=self.n)                          # E sigma(|w_k| Z)
            kk, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
            sp = pair_integrals(self.fs, self.dfs, self.bs, self.dfs, self.bs, nrm[kk.ravel()], nrm[jj.ravel()],
                                C.ravel(), ("J1", "K"), n=self.n_self)
            J1s, Ks = sp["J1"].reshape(m, m), sp["K"].reshape(m, m)                  # [k, j]
            SB = (s[:, None] * (J1s - C * nrm[:, None] * Ks)).sum(0) - (s * mu).sum() * e1
            SA = s[:, None] * nrm[:, None] * Ks
            F -= (al / m) * (Wh * SB[None, :] + Wh @ SA)
        return F * s[None, :]

    def loss(self, W, refit=False):
        """Population loss with the optimal intercept; with refit=True also the population least-squares refit of the
        head on the frozen features sigma(w_j . x) (plus intercept), i.e. V - c^T Q^+ c."""
        d, m = W.shape
        r = self.c.size
        Wh, nrm, C = self._geom(W)
        s, al = self.s, self.alpha
        mu = moments_1d(self.fs, self.bs, nrm, n=self.n)
        ct = Wh[:r].ravel()
        S_t = pair_integrals(self.ft, self.dft, self.bt, self.fs, self.bs, np.ones_like(ct), np.tile(nrm, r), ct,
                             ("S",), n=self.n)["S"].reshape(r, m)
        cvec = self.gam * (self.c[:, None] * (S_t - self.mu_phi * mu[None, :])).sum(0)      # Cov(Y, sigma(w_j.X))
        cov_yf = (al / m) * (s * cvec).sum()
        iu = np.triu_indices(m)
        S_ss = pair_integrals(self.fs, self.dfs, self.bs, self.fs, self.bs, nrm[iu[0]], nrm[iu[1]], C[iu], ("S",),
                              n=self.n)["S"]
        Q = np.zeros((m, m))
        Q[iu] = S_ss
        Q = Q + Q.T - np.diag(np.diag(Q))
        Q -= np.outer(mu, mu)
        var_f = (al / m) ** 2 * s @ Q @ s
        L = float(self.V - 2.0 * cov_yf + var_f)
        if not refit:
            return L
        return L, population_refit(Q, cvec, self.V)

    def agop(self, W):
        d, m = W.shape
        Wh, nrm, C = self._geom(W)
        iu = np.triu_indices(m)
        # E[sigma'(a.X) sigma'(b.X)]; G = sigma' needs its own derivative only for K, which is not requested
        S = pair_integrals(self.dfs, self.dfs, self.bs, self.dfs, self.bs, nrm[iu[0]], nrm[iu[1]], C[iu], ("S",),
                           n=self.n)["S"]
        Hm = np.zeros((m, m))
        Hm[iu] = S
        Hm = Hm + Hm.T - np.diag(np.diag(Hm))
        Ws = W * self.s[None, :]
        M = (self.alpha / m) ** 2 * (Ws @ Hm @ Ws.T)
        return 0.5 * (M + M.T)


# ------------------------------------------------------------------------------------------------ sampled quantities
def sample_data(rng, n, d, r, link, c, gam):
    X = rng.standard_normal((n, d))
    Y = gam * (ACTS[link][0](X[:, :r]) @ np.asarray(c, dtype=np.float64))
    return X, Y


def _chunks(n, d, m, budget=4_000_000):
    step = max(256, budget // max(m, 1))
    return [(a, min(n, a + step)) for a in range(0, n, step)]


def loss_and_force(X, Y, W, s, alpha, act, need_force=True):
    """Empirical loss (intercept refitted on (X, Y)) and force F = mean[R X sigma'(XW)] diag(s), in one pass."""
    f_act, d_act = ACTS[act]
    n, d = X.shape
    m = W.shape[1]
    f = np.empty(n)
    A1 = np.zeros((d, m)) if need_force else None
    A2 = np.zeros((d, m)) if need_force else None
    for a, b in _chunks(n, d, m):
        Z = X[a:b] @ W
        f[a:b] = (alpha / m) * (f_act(Z) @ s)
        if need_force:
            D = d_act(Z)
            A1 += X[a:b].T @ ((Y[a:b] - f[a:b])[:, None] * D)
            A2 += X[a:b].T @ D
    b0 = float((Y - f).mean())
    R = Y - f - b0
    L = float((R * R).mean())
    if not need_force:
        return L, None
    F = (A1 - b0 * A2) / n * s[None, :]
    return L, F


def sampled_agop(X, W, s, alpha, act):
    """AGOP M = E[grad_x f grad_x f^T] = (alpha/m)^2 E[G G^T], G = W diag(s) sigma'(W^T x), on the sample X."""
    d_act = ACTS[act][1]
    n, d = X.shape
    m = W.shape[1]
    Ws = (W * s[None, :]).T                          # m x d
    M = np.zeros((d, d))
    for a, b in _chunks(n, d, m):
        G = d_act(X[a:b] @ W) @ Ws                   # (b-a) x d, rows = grad_x f / (alpha/m)
        M += G.T @ G
    M *= (alpha / m) ** 2 / n
    return 0.5 * (M + M.T)


def ridge_refit(Xtr, Ytr, Xte, Yte, W, act, lam_rel=1e-8):
    """Refit the head (m weights + intercept) by ridge least squares on the frozen features sigma(XW); test MSE."""
    f_act = ACTS[act][0]
    Str = f_act(Xtr @ W)
    mu = Str.mean(0)
    Sc = Str - mu
    yc = Ytr - Ytr.mean()
    G = Sc.T @ Sc / len(Ytr)
    lam = lam_rel * max(np.trace(G) / G.shape[0], 1e-300)
    a = np.linalg.solve(G + lam * np.eye(G.shape[0]), Sc.T @ yc / len(Ytr))
    b0 = Ytr.mean() - mu @ a
    Ste = f_act(Xte @ W)
    res = Yte - Ste @ a - b0
    return float(((res - res.mean()) ** 2).mean()), float((res ** 2).mean())


# ------------------------------------------------------------------------------------------------ optimizers
def polar(F, tol=1e-12):
    A, sv, Bt = np.linalg.svd(F, full_matrices=False)
    if sv.size == 0 or sv[0] <= 0:
        return np.zeros_like(F), 0
    k = int((sv > sv[0] * tol).sum())
    return A[:, :k] @ Bt[:k], k


def ns5(F, steps=5, eps=1e-7):
    """Quintic Newton-Schulz orthogonalisation as in the Muon reference implementation (approximate Polar)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = F / (np.linalg.norm(F) + eps)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Optimizer:
    def __init__(self, name, eta, lr, alpha, d, m, momentum=0.0, nesterov=False):
        if name not in OPTS:
            raise ValueError("unknown optimizer %r; choose from %s" % (name, OPTS))
        self.name, self.eta, self.lr, self.alpha = name, eta, lr, alpha
        self.d, self.m = d, m
        self.beta, self.nesterov = momentum, nesterov
        self.buf = None
        self.t = 0
        self.mom1 = self.mom2 = None
        self.rank = None

    def calibrate(self, F0):
        """Default learning rates: first step of Frobenius size eta*sqrt(min(d,m)) (a full-rank polar step)."""
        if self.lr is None:
            if self.name == "gd":
                self.lr = self.eta * math.sqrt(min(self.d, self.m)) / max(
                    np.linalg.norm((2 * self.alpha / self.m) * F0), 1e-300)
            elif self.name == "adam":
                self.lr = self.eta * math.sqrt(min(self.d, self.m) / (self.d * self.m))

    def direction(self, F):
        if self.beta > 0 and self.name != "adam":
            self.buf = F.copy() if self.buf is None else self.beta * self.buf + F
            return F + self.beta * self.buf if self.nesterov else self.buf
        return F

    def step(self, W, F):
        self.t += 1
        n = self.name
        if n == "adam":
            g = -(2 * self.alpha / self.m) * F                  # gradient of L
            b1 = self.beta if self.beta > 0 else 0.9
            b2, eps = 0.999, 1e-12
            if self.mom1 is None:
                self.mom1 = np.zeros_like(W)
                self.mom2 = np.zeros_like(W)
            self.mom1 = b1 * self.mom1 + (1 - b1) * g
            self.mom2 = b2 * self.mom2 + (1 - b2) * g * g
            mh = self.mom1 / (1 - b1 ** self.t)
            vh = self.mom2 / (1 - b2 ** self.t)
            D = -self.lr * mh / (np.sqrt(vh) + eps * np.sqrt(vh.mean() + 1e-300))
            return W + D, D
        G = self.direction(F)
        if n == "polar":
            P, self.rank = polar(G)
            D = self.eta * P
        elif n == "ns":
            D = self.eta * ns5(G)
        elif n == "ngd":
            D = self.eta * math.sqrt(min(self.d, self.m)) * G / max(np.linalg.norm(G), 1e-300)
        elif n == "sign":
            D = self.eta * math.sqrt(min(self.d, self.m) / (self.d * self.m)) * np.sign(G)
        elif n == "gd":
            D = self.lr * (2 * self.alpha / self.m) * G
        return W + D, D


# ------------------------------------------------------------------------------------------------ metrics
def top_r_alignment(M, r):
    """A_r = tr(P_U P_r)/r (U = span(e_1..e_r), P_r = top-r eigenprojector of M), cos^2 of the smallest principal
    angle, the eigen-gap lambda_r/lambda_{r+1}, and whether the gap is strict (then P_r is unique)."""
    ev, Q = np.linalg.eigh(0.5 * (M + M.T))
    ev, Q = ev[::-1], Q[:, ::-1]
    cos = np.linalg.svd(Q[:r, :r], compute_uv=False)
    lr_, lr1 = ev[r - 1], (ev[r] if M.shape[0] > r else 0.0)
    gap = float("inf") if lr1 <= 1e-15 * max(abs(ev[0]), 1e-300) else float(lr_ / lr1)
    return dict(A=float((cos ** 2).sum() / r), cos2_min=float(cos.min() ** 2), gap=gap,
                strict=bool(lr_ > lr1 * (1 + 1e-9)))


def neuron_stats(W, r):
    Wh, nrm = _unit(W)
    eU = (Wh[:r] ** 2).sum(0)                                # |P_U w_hat_j|^2; baseline r/d at a random direction
    spec = np.abs(Wh[:r]).max(0)                             # max_i |u_i . w_hat_j|
    arg = np.abs(Wh[:r]).argmax(0)
    counts = [int(((arg == i) & (spec > 0.9)).sum()) for i in range(r)]
    q = np.percentile(eU, [10, 50, 90])
    return dict(eU_mean=float(eU.mean()), eU_p10=float(q[0]), eU_p50=float(q[1]), eU_p90=float(q[2]),
                frac_eU_gt_half=float((eU > 0.5).mean()), frac_spec_gt_09=float((spec > 0.9).mean()),
                spec_counts=counts)


def weight_alignment(W, r):
    C = W @ W.T / W.shape[1]
    out = top_r_alignment(C, r)
    return dict(A_W=out["A"], cos2min_W=out["cos2_min"], mass_U=float((W[:r] ** 2).sum() / (W ** 2).sum()))


# ------------------------------------------------------------------------------------------------ driver
def default_config():
    return dict(teacher="relu", c=None, student="relu", opt="polar", r=2, d=64, m=256, n=32768, n_test=65536,
                data="pool", batch=None, T=200, eta=0.02, lr=None, alpha=1e-3, sig0=1.0, head="pos", momentum=0.0,
                nesterov=False, seed=0, every=None, refit=False, control=False, save_W=None, verbose=True,
                quad_n=16, quad_n_self=8, self_force=True)


def _checkpoints(T, every):
    if every is None:
        every = max(1, T // 60)
    cps = set(range(0, T + 1, every)) | {0, 1, 2, 3, 5, T}
    k = 1
    while k <= T:
        cps.add(k)
        k *= 2
    return sorted(t for t in cps if t <= T)


def run(cfg):
    cfg = dict(default_config(), **cfg)
    r, d, m = cfg["r"], cfg["d"], cfg["m"]
    link, act = cfg["teacher"], cfg["student"]
    if link not in ACTS or act not in ACTS:
        raise ValueError("unknown link/activation; available: %s" % sorted(ACTS))
    if cfg["data"] not in DATAS:
        raise ValueError("data must be one of %s" % (DATAS,))
    c = np.ones(r) if cfg["c"] is None else np.asarray(cfg["c"], dtype=np.float64)
    if c.shape != (r,):
        raise ValueError("teacher weights c must have length r")
    gam, V, EY = teacher_constants(link, c)
    exact = cfg["data"] == "exact"
    quad = cfg["data"] == "quad"
    if exact and not (link == "relu" and act == "relu" and np.allclose(c, 1.0) and cfg["head"] == "pos"):
        raise ValueError("data=exact needs teacher relu, student relu, balanced c and head pos")
    if not cfg["self_force"] and cfg["data"] not in ("exact", "quad"):
        raise ValueError("--no_self_force needs data=exact or data=quad")
    rng = np.random.default_rng([cfg["seed"], 1])
    rng_data = np.random.default_rng([cfg["seed"], 2])
    rng_ctrl = np.random.default_rng([cfg["seed"], 3])
    s = np.ones(m) if cfg["head"] == "pos" else np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    alpha, eta = cfg["alpha"], cfg["eta"]
    W0 = rng.standard_normal((d, m)) * (cfg["sig0"] / math.sqrt(d))
    W = W0.copy()
    if quad:
        qp = QuadPopulation(link, act, c, gam, V, alpha, s, n=cfg["quad_n"], n_self=cfg["quad_n_self"],
                            self_force=cfg["self_force"])
    elif not exact:
        Xtr, Ytr = sample_data(rng_data, cfg["n"], d, r, link, c, gam)
        Xte, Yte = sample_data(rng_data, cfg["n_test"], d, r, link, c, gam)
    opt = Optimizer(cfg["opt"], eta, cfg["lr"], alpha, d, m, cfg["momentum"], cfg["nesterov"])
    T = cfg["T"]
    cps = set(_checkpoints(T, cfg["every"]))
    ser = {k: [] for k in ("t", "L_test", "A", "cos2_min", "gap", "strict", "A_W", "cos2min_W", "mass_U", "eU_mean",
                           "eU_p10", "eU_p50", "eU_p90", "frac_eU_gt_half", "frac_spec_gt_09", "spec_counts", "disp",
                           "L_refit", "A_ctrl")}
    L_train = []
    saved = {}
    t0 = time.time()
    eta_bar = eta / math.sqrt(m)
    kappa_mf = cfg["sig0"] / (eta_bar * math.sqrt(d))
    if cfg["verbose"]:
        print("# vlab: teacher=%s student=%s opt=%s data=%s r=%d d=%d m=%d alpha=%g eta=%g sig0=%g  gamma=%.6f "
              "V=%.6f  kappa_MF=%.4g" % (link, act, cfg["opt"], cfg["data"], r, d, m, alpha, eta, cfg["sig0"], gam, V,
                                        kappa_mf), flush=True)
    for t in range(T + 1):
        # -------- force and train loss at W(t)
        if exact:
            Ltr = exact_loss(W, r, alpha, gam, V)
            F = exact_force(W, r, alpha if cfg["self_force"] else 0.0, gam)
        elif quad:
            F = qp.force(W)
            Lq = qp.loss(W, refit=bool(cfg["refit"])) if t in cps else None
            Ltr = (Lq[0] if isinstance(Lq, tuple) else Lq) if Lq is not None else float("nan")
        else:
            if cfg["data"] == "fresh" and t > 0:
                Xtr, Ytr = sample_data(rng_data, cfg["n"], d, r, link, c, gam)
            if cfg["batch"]:
                idx = rng_data.choice(Xtr.shape[0], size=cfg["batch"], replace=False)
                Ltr, F = loss_and_force(Xtr[idx], Ytr[idx], W, s, alpha, act)
            else:
                Ltr, F = loss_and_force(Xtr, Ytr, W, s, alpha, act)
        L_train.append(Ltr / V)
        if not np.all(np.isfinite(F)) or (not quad and not np.isfinite(Ltr)):
            print("# non-finite state at t=%d; stopping" % t, file=sys.stderr)
            break
        # -------- checkpoint metrics
        if t in cps:
            if exact:
                Lte = Ltr
                M = exact_agop(W, alpha)
            elif quad:
                Lte = Ltr
                M = qp.agop(W)
            else:
                Lte, _ = loss_and_force(Xte, Yte, W, s, alpha, act, need_force=False)
                M = sampled_agop(Xte, W, s, alpha, act)
            al = top_r_alignment(M, r)
            wa = weight_alignment(W, r)
            ns_ = neuron_stats(W, r)
            ser["t"].append(t)
            ser["L_test"].append(Lte / V)
            for k in ("A", "cos2_min", "gap", "strict"):
                ser[k].append(al[k])
            for k in ("A_W", "cos2min_W", "mass_U"):
                ser[k].append(wa[k])
            for k in ("eU_mean", "eU_p10", "eU_p50", "eU_p90", "frac_eU_gt_half", "frac_spec_gt_09", "spec_counts"):
                ser[k].append(ns_[k])
            ser["disp"].append(float(np.linalg.norm(W - W0) / np.linalg.norm(W0)))
            if cfg["refit"]:
                if exact:
                    ser["L_refit"].append(exact_refit(W, r, gam, V) / V)
                elif quad:
                    ser["L_refit"].append(Lq[1] / V)
                else:
                    ser["L_refit"].append(ridge_refit(Xtr, Ytr, Xte, Yte, W, act)[0] / V)
            else:
                ser["L_refit"].append(None)
            if cfg["control"] and t > 0:
                G = rng_ctrl.standard_normal((d, m))
                Wc = W0 + np.linalg.norm(W - W0) * G / np.linalg.norm(G)
                Mc = exact_agop(Wc, alpha) if exact else (qp.agop(Wc) if quad else sampled_agop(Xte, Wc, s, alpha, act))
                ser["A_ctrl"].append(top_r_alignment(Mc, r)["A"])
            else:
                ser["A_ctrl"].append(None)
            if cfg["save_W"] is not None:
                saved["W_%d" % t] = W.copy()
            if cfg["verbose"]:
                print("t=%5d  L_test/V=%.6f  L_train/V=%.6f  A_r=%.4f  cos2min=%.4f  gap=%.4g  A_W=%.4f  eU_med=%.4f  "
                      "spec>.9=%.3f  disp=%.4f%s" % (t, Lte / V, Ltr / V, al["A"], al["cos2_min"], al["gap"], wa["A_W"],
                                                    ns_["eU_p50"], ns_["frac_spec_gt_09"], ser["disp"][-1],
                                                    ("  L_refit/V=%.5f" % ser["L_refit"][-1]) if ser["L_refit"][-1] is not None else ""),
                      flush=True)
        if t == T:
            break
        if t == 0:
            opt.calibrate(F)
        W, _ = opt.step(W, F)
    out = dict(config=cfg, constants=dict(gamma=gam, V=V, EY=EY, eta_bar=eta_bar, kappa_MF=kappa_mf,
                                          kappa_fin=cfg["sig0"] / eta, lr_used=opt.lr),
               series=ser, L_train=L_train,
               summary=summarize(ser, L_train if (cfg["data"] in ("pool", "exact") and not cfg["batch"]) else None),
               seconds=time.time() - t0)
    if cfg["save_W"] is not None:
        np.savez_compressed(cfg["save_W"], W0=W0, **saved)
    return out


def summarize(ser, L_train=None):
    """Plateau / alignment summary. The plateau is the longest prefix [0, t] on which max/min of the test loss over the
    checkpoints stays <= 1.01 (and, when L_train is given -- full-pool or exact data -- also that of the train loss over
    every step)."""
    t = np.array(ser["t"])
    L = np.array(ser["L_test"], dtype=float)
    A = np.array(ser["A"], dtype=float)
    if len(t) == 0:
        return {}

    def prefix_end(vals):
        lo = hi = vals[0]
        k_end = 0
        for k in range(1, len(vals)):
            lo, hi = min(lo, vals[k]), max(hi, vals[k])
            if hi / lo > 1.01:
                break
            k_end = k
        return k_end
    kc = prefix_end(L)
    if L_train is not None and len(L_train):
        t_plat = int(min(t[kc], prefix_end(np.array(L_train, dtype=float))))
    else:
        t_plat = int(t[kc])
    kk = int(np.searchsorted(t, t_plat, side="right") - 1)
    hit = np.where(A >= 0.9)[0]
    t_align = int(t[hit[0]]) if hit.size else None
    return dict(A_0=float(A[0]), A_end=float(A[-1]), A_max=float(A.max()), t_A_ge_09=t_align,
                plateau_1pct_until=t_plat, A_at_plateau_end=float(A[kk]),
                A_gain_in_plateau=float(A[kk] - A[0]),
                loss_drop_over_V=float(L[0] - L[-1]), L_test_end=float(L[-1]),
                strict_gap_at_plateau_end=bool(ser["strict"][kk]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list links/activations and optimizers, then exit")
    ap.add_argument("--teacher", default="relu", help="teacher link (see --list)")
    ap.add_argument("--c", default=None, help="teacher weights, comma separated (default: balanced ones)")
    ap.add_argument("--student", default="relu", help="student activation (see --list)")
    ap.add_argument("--opt", default="polar", choices=OPTS)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=32768, help="training samples (pool size, or fresh samples per step)")
    ap.add_argument("--n_test", type=int, default=65536)
    ap.add_argument("--data", default="pool", choices=DATAS)
    ap.add_argument("--batch", type=int, default=None, help="minibatch size drawn from the pool at each step")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.02, help="step for polar/ns/ngd/sign (and default lr scale)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate for gd/adam (default: matched first step)")
    ap.add_argument("--alpha", type=float, default=1e-3, help="head scale; the head is alpha/m")
    ap.add_argument("--sig0", type=float, default=1.0, help="init scale: w_j(0) ~ N(0, sig0^2 I/d)")
    ap.add_argument("--head", default="pos", choices=("pos", "pm"))
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--nesterov", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=None, help="checkpoint spacing (default T//60, plus powers of 2)")
    ap.add_argument("--refit", action="store_true", help="ridge refit of the head on frozen features at checkpoints")
    ap.add_argument("--control", action="store_true", help="random update of the same size (learning control)")
    ap.add_argument("--save_W", default=None, help="npz path to save W at the checkpoints")
    ap.add_argument("--quad_n", type=int, default=16, help="data=quad: Gauss-Legendre nodes per piece (teacher part, "
                                                           "loss, AGOP)")
    ap.add_argument("--quad_n_self", type=int, default=8, help="data=quad: nodes per piece for the self-force")
    ap.add_argument("--no_self_force", action="store_true",
                    help="exact/quad only: drop the O(alpha) self-force from the update (the alpha -> 0 dynamics of the "
                         "G2 certificate); the loss is still reported at the given alpha")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("links/activations:", ", ".join(sorted(ACTS)))
        print("optimizers:", ", ".join(OPTS))
        print("data:", ", ".join(DATAS))
        return None
    cfg = dict(teacher=a.teacher, c=None if a.c is None else [float(x) for x in a.c.split(",")], student=a.student,
               opt=a.opt, r=a.r, d=a.d, m=a.m, n=a.n, n_test=a.n_test, data=a.data, batch=a.batch, T=a.T, eta=a.eta,
               lr=a.lr, alpha=a.alpha, sig0=a.sig0, head=a.head, momentum=a.momentum, nesterov=a.nesterov,
               seed=a.seed, every=a.every, refit=a.refit, control=a.control, save_W=a.save_W, verbose=not a.quiet,
               quad_n=a.quad_n, quad_n_self=a.quad_n_self, self_force=not a.no_self_force)
    out = run(cfg)
    print("# summary:", json.dumps(out["summary"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh)
    return out


if __name__ == "__main__":
    main()
