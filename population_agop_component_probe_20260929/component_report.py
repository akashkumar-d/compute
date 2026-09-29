"""Scalar reporting only; no model imports or numerical evaluations."""
import math
import sys

COMPONENTS = ('total', 'teacher', 'student')


def cancellation(slopes):
    total, teacher, student = (float(slopes[k]) for k in COMPONENTS)
    return dict(factor=(abs(teacher) + abs(student)) / abs(total) if total else None,
                zero_total=bool(total == 0.0))


def directional_report(base, minus, plus, nominal, epsilon):
    """Derivative along -D: [C(theta-epsilon D)-C(theta+epsilon D)]/(2 epsilon)."""
    epsilon = float(epsilon)
    values = [float(v[k]) for v in (base, minus, plus, nominal) for k in COMPONENTS]
    if not math.isfinite(epsilon) or epsilon <= 0 or not all(map(math.isfinite, values)):
        raise ValueError('Finite scalar components and positive epsilon required')
    result = dict(epsilon=epsilon, components={})
    fd, mismatch = {}, {}
    for name in COMPONENTS:
        baseline, lower, upper, slope = (float(v[name]) for v in (base, minus, plus, nominal))
        derivative = float((lower - upper) / (2 * epsilon))
        floor = float(128 * sys.float_info.epsilon * max(1.0, abs(baseline)) / epsilon)
        delta = float(derivative - slope)
        fd[name], mismatch[name] = derivative, delta
        result['components'][name] = dict(
            baseline=baseline, minus_value=lower, plus_value=upper,
            derivative=derivative, nominal_slope=slope, signed_difference=delta,
            absolute_difference=abs(delta),
            relative_difference=abs(delta) / max(abs(derivative), abs(slope), 1e-300),
            roundoff_scale=floor,
            agreement=bool(abs(delta) <= 1e-3 * max(abs(derivative), abs(slope)) + floor),
            roundoff_dominated=bool(abs(derivative) <= floor))
    result.update(
        fd_additivity_residual=float(fd['total'] - fd['teacher'] - fd['student']),
        nominal_additivity_residual=float(nominal['total'] - nominal['teacher'] - nominal['student']),
        mismatch_additivity_residual=float(mismatch['total'] - mismatch['teacher'] - mismatch['student']),
        fd_cancellation=cancellation(fd), nominal_cancellation=cancellation(nominal),
        total_fd_negative=bool(fd['total'] < 0))
    return result


def stability(reports):
    result = {}
    for name in COMPONENTS:
        first, second = (r['components'][name] for r in reports)
        a, b = first['derivative'], second['derivative']
        floor = max(first['roundoff_scale'], second['roundoff_scale'])
        result[name] = dict(signed_change=float(b-a), absolute_change=float(abs(b-a)),
            stable=bool(abs(b-a) <= 1e-3 * max(abs(a), abs(b)) + floor),
            both_agree=bool(first['agreement'] and second['agreement']))
    return result
