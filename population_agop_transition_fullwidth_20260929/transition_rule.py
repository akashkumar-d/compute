"""Scalar transition-resolving Gaussian quadrature and stable SiLU derivatives."""
from functools import lru_cache
import math
import operator

import numpy as np

BASE_BREAKS = (-12., -8., -5., -3., -1., 0., 1., 3., 5., 8., 12.)
GATE_LEVELS = (-12., -6., -3., -1., 0., 1., 3., 6., 12.)


@lru_cache(maxsize=32)
def _legendre(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def transition_nodes(bias, scale, order=8, limit=12.0, extra_gates=()):
    """Return z and unnormalized standard-normal weights on [-limit,limit].

    Composite Gauss-Legendre uses base breaks plus gate levels mapped by
    (level-bias)/scale, unioned over optional extra (bias,scale) gates. Zero-scale
    gates add no transitions. Gaussian tails outside the interval are omitted,
    not redistributed or silently renormalized.
    """
    if isinstance(order, (bool, np.bool_)):
        raise ValueError('order must be a positive integer')
    order = operator.index(order)
    bias, scale, limit = float(bias), float(scale), float(limit)
    if order < 1 or not all(map(math.isfinite, (bias, scale, limit))) or scale < 0 or limit <= 0:
        raise ValueError('Require finite bias, scale>=0, limit>0 and order>=1')
    breaks = {-limit, limit}
    breaks.update(max(-limit, min(limit, x)) for x in BASE_BREAKS)
    for gate_bias, gate_scale in ((bias, scale), *extra_gates):
        gate_bias, gate_scale = float(gate_bias), float(gate_scale)
        if not math.isfinite(gate_bias) or not math.isfinite(gate_scale) or gate_scale < 0:
            raise ValueError('Extra gates require finite bias and nonnegative scale')
        if gate_scale > 0:
            with np.errstate(over='ignore'):
                mapped = (np.asarray(GATE_LEVELS)-gate_bias)/gate_scale
            breaks.update(float(x) for x in mapped if -limit < x < limit)
    edges = np.asarray(sorted(breaks))
    half = .5*(edges[1:]-edges[:-1])
    middle = .5*(edges[1:]+edges[:-1])
    x, w = _legendre(order)
    z = (middle[:, None]+half[:, None]*x).ravel()
    weights = (half[:, None]*w).ravel()*np.exp(-.5*z*z)/math.sqrt(2*math.pi)
    return z, weights


def _sigmoid_factors(max_order):
    # q^(n) = q(1-q) P_n(q), n>=1;
    # P_(n+1) = (1-2q)P_n + q(1-q)P_n'. Exact integer coefficients.
    factors = []
    polynomial = [1]
    for _ in range(max_order):
        factors.append(tuple(polynomial))
        following = [0]*(len(polynomial)+1)
        for k, value in enumerate(polynomial):
            following[k] += (k+1)*value
            following[k+1] -= (k+2)*value
        polynomial = following
    return tuple(factors)


SIGMOID_FACTORS = _sigmoid_factors(6)


def silu_derivatives(x, max_order=6):
    """List S^(0)(x),...,S^(max_order)(x), with stable sigmoid tails."""
    if isinstance(max_order, (bool, np.bool_)):
        raise ValueError('max_order must be an integer from 0 through 6')
    max_order = operator.index(max_order)
    if not 0 <= max_order <= 6:
        raise ValueError('max_order must be an integer from 0 through 6')
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Finite gate arguments required')
    exponential = np.exp(-np.abs(x))
    r = exponential/(1+exponential)  # r<=1/2; avoids cancellation at q near 1.
    sigmoid = [np.where(x >= 0, 1-r, r)]
    for n in range(1, max_order+1):
        negative_side = r*(1-r)*np.polynomial.polynomial.polyval(r, SIGMOID_FACTORS[n-1])
        sigmoid.append(np.where(x >= 0, (-1)**(n+1)*negative_side, negative_side))
    return [x*sigmoid[0]] + [x*sigmoid[n]+n*sigmoid[n-1] for n in range(1, max_order+1)]
