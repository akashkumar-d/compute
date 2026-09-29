"""The registered recipe for the last hidden layer's shrink rate (campaign C), shared by the worker and the analysis.

A pilot run at a fixed last-layer rate r0 gives the step t10 at which held-out accuracy first reaches 10%. The main run's last-layer
rate r (as a multiple of layer 1's eta*lambda) solves

        r * K1 * t10 * exp(SLOPE * (r - r0)) = x_target,

so that x = eta_L * lambda_L * t10 lands on x_target; the exponential factor allows for t10 moving with the rate (about +9% per +0.1,
measured on the depth-5 dial T21). r is rounded to 0.005 and clipped to [R_MIN, R_MAX].
"""
import math

K1 = 0.03 * 0.015          # layer 1's eta * lambda
SLOPE = 0.9                # d ln t10 / d r
ROUND = 0.005
R_MIN, R_MAX = 0.10, 1.20
ETA_UNIT = 0.03 * 0.015 / 0.07   # eta of a deep layer (decay 0.07) that shrinks at layer 1's rate


def solve_rate(t10, r0, x_target, slope=SLOPE):
    lo, hi = 0.01, 3.0
    f = lambda r: r * K1 * t10 * math.exp(slope * (r - r0)) - x_target
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    r = round(round(0.5 * (lo + hi) / ROUND) * ROUND, 10)     # the outer round removes float noise (70 * 0.005 = 0.35000000000000003)
    return min(max(r, R_MIN), R_MAX)


def first_t10(obs, thr=0.1):
    return next((o['step'] for o in obs if o.get('test_acc', 0) >= thr), None)


def fill_main_cfg(template, rate):
    cfg = dict(template); eta = list(cfg['eta_layers']); eta[-1] = rate * ETA_UNIT; cfg['eta_layers'] = eta
    return cfg


if __name__ == '__main__':
    # the T23 registration (depth 5, x_target 3.85, r0 0.35) reproduced from the T22 pilots' t10
    for t10, want in ((31940, 0.285), (30120, 0.300), (28160, 0.315), (27320, 0.320)):
        r = solve_rate(t10, 0.35, 3.85); print(t10, r, 'T23 used', want); assert abs(r - want) < 1e-9
    print('recipe reproduces T23')
