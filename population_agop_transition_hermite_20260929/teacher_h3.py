"""Pure He3/sqrt(6) teacher correlation: exact 1-D identities, finite quadrature.

No old kernel is modified. This module is a numerical prototype, not a training
entrypoint or validation claim. Teacher coefficients/gamma are not normalized here.
"""
import math

import numpy as np

from transition_rule import transition_nodes, silu_derivatives


def teacher_terms_h3(pi, p0, nu, v0, coefficients, gamma, order=8, limit=12):
    """Return correlation (t,gp,gv) for y=gamma*sum_i c_i*He3(x_i)/sqrt(6).

    pi,nu have shape (d,m); p0,v0 shape (m,); coefficients shape (r,), r<=d.
    gp/gv include the bias row and have shape (d+1,m). These are correlation
    gradients, not loss gradients. No division by ||p|| occurs, including at zero.
    """
    pi, p0, nu, v0, coefficients = [np.asarray(x, dtype=float)
        for x in (pi, p0, nu, v0, coefficients)]
    if pi.ndim != 2 or nu.shape != pi.shape:
        raise ValueError('pi and nu must have matching (d,m) shapes')
    d, m = pi.shape
    if d < 1 or m < 1 or p0.shape != (m,) or v0.shape != (m,):
        raise ValueError('Positive dimensions and matching bias vectors required')
    if coefficients.ndim != 1 or not 1 <= coefficients.size <= d:
        raise ValueError('Require 1<=teacher rank<=d')
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0 or not all(np.isfinite(x).all()
            for x in (pi, p0, nu, v0, coefficients)):
        raise ValueError('Finite parameters and positive gamma required')
    variance = gamma*gamma*float(coefficients @ coefficients)
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError('Teacher variance must be finite and positive')
    scales = np.sqrt(np.sum(pi*pi, axis=0))
    A = np.empty((7,m))
    for j in range(m):
        z, weights = transition_nodes(p0[j], scales[j], order=order, limit=limit)
        derivatives = silu_derivatives(p0[j]+scales[j]*z)
        A[:,j] = [float(weights @ value) for value in derivatives]
    r = coefficients.size
    c = coefficients[:,None]
    C3 = np.sum(c*pi[:r]**3, axis=0)
    C21 = np.sum(c*pi[:r]**2*nu[:r], axis=0)
    D = np.sum(pi*nu, axis=0)
    u, w = np.zeros_like(pi), np.zeros_like(pi)
    u[:r] = c*pi[:r]**2
    w[:r] = c*pi[:r]*nu[:r]
    kappa = gamma/math.sqrt(6)
    common = v0*A[3]+D*A[4]
    t = kappa*(C3*common+3*C21*A[2])
    gp, gv = np.empty((d+1,m)), np.empty((d+1,m))
    gp[:d] = kappa*(3*u*common + C3*(nu*A[4]+pi*(v0*A[5]+D*A[6]))
        + 6*w*A[2] + 3*C21*pi*A[4])
    gp[d] = kappa*(C3*(v0*A[4]+D*A[5])+3*C21*A[3])
    gv[:d] = kappa*(C3*pi*A[4]+3*u*A[2])
    gv[d] = kappa*C3*A[3]
    return t, gp, gv
