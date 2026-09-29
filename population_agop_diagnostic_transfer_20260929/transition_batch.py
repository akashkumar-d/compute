"""Batch scalar transition rules without changing nodes, weights or their order.

No model imports. The shared scale is nonnegative; each bias gets exactly the
base-plus-one-gate partition used by transition_rule.transition_nodes.
"""
import math
import operator

import numpy as np
from transition_rule import BASE_BREAKS, GATE_LEVELS, _legendre


def transition_nodes_batch(biases, scale, order=8, limit=12.0):
    """Return (owners,z,weights) in concatenated scalar-rule order.

    owners indexes the one-dimensional biases input. Only exactly duplicated
    edges are discarded; nonzero intervals, even when their half-width rounds
    to zero, are retained. Gaussian tails are omitted without renormalization.
    """
    if isinstance(order, (bool, np.bool_)):
        raise ValueError('order must be a positive integer')
    order = operator.index(order)
    scale, limit = float(scale), float(limit)
    biases = np.asarray(biases, dtype=float)
    if (biases.ndim != 1 or order < 1 or not np.isfinite(biases).all()
            or not math.isfinite(scale) or scale < 0
            or not math.isfinite(limit) or limit <= 0):
        raise ValueError('Require finite 1-D biases, scale>=0, limit>0 and order>=1')
    count = len(biases)
    if not count:
        return np.empty(0, dtype=np.intp), np.empty(0), np.empty(0)
    # The scalar set inserts endpoints and BASE_BREAKS before gate levels.
    base = np.asarray([-limit, limit] +
                      [max(-limit, min(limit, x)) for x in BASE_BREAKS])
    edges = np.broadcast_to(base, (count, len(base)))
    if scale > 0:
        with np.errstate(over='ignore'):
            mapped = (np.asarray(GATE_LEVELS)[None, :] - biases[:, None]) / scale
        # Exterior mapped levels add no scalar break. Clipping them to the
        # existing endpoints is equivalent after removing duplicate intervals.
        mapped = np.clip(mapped, -limit, limit)
        edges = np.concatenate((edges, mapped), axis=1)
    else:
        edges = edges.copy()
    # BASE_BREAKS already contains +0.0, which the scalar set retains when a
    # mapped level equals -0.0. Normalize the duplicate representation likewise.
    edges[edges == 0.] = 0.
    edges.sort(axis=1)
    left, right = edges[:, :-1], edges[:, 1:]
    valid = right > left
    row, interval = np.nonzero(valid)
    left, right = left[row, interval], right[row, interval]
    half = .5 * (right - left)
    middle = .5 * (left + right)
    x, w = _legendre(order)
    z = (middle[:, None] + half[:, None] * x).ravel()
    weights = (half[:, None] * w).ravel() * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi)
    owners = np.repeat(row, order)
    return owners, z, weights
