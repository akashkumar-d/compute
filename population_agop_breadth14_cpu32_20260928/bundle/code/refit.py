"""Same-budget population refits and numerical primal/lower certificates.

The supplied quadratic is never truncated, ridged, or projected onto the PSD cone.
The solver uses an exact ball-intersection projection, a spectral trust-region
candidate, and projected acceleration with active-face refinement. A warm start
is a coefficient vector for the same ordered normalized feature bank.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy import sparse

_LOCAL_DEPS = Path(__file__).resolve().parents[1]/'dependencies'
if _LOCAL_DEPS.is_dir() and str(_LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DEPS))
try:
    import clarabel
except ImportError:
    clarabel = None


_EPS = np.finfo(float).eps


def _threshold_ratio(a, ratio):
    """Find t>=0 with ||(a-t)+||1 / ||(a-t)+||2 = ratio."""
    a = np.sort(np.asarray(a, dtype=float))[::-1]
    if not len(a) or a[0] == 0:
        return 0.0
    if a.sum() <= ratio * np.linalg.norm(a):
        return 0.0
    # Shift before computing prefix variances: near-tied gradients are common.
    d = a-a[0]
    k = np.arange(1, len(a)+1, dtype=float)
    sd, sd2 = np.cumsum(d), np.cumsum(d*d)
    variance = np.maximum(sd2-sd*sd/k, 0.0)
    eligible = k > ratio*ratio
    t = np.full(len(a), -np.inf)
    t[eligible] = a[0]+sd[eligible]/k[eligible] - ratio*np.sqrt(
        variance[eligible]/(k[eligible]*(k[eligible]-ratio*ratio)))
    next_a = np.r_[a[1:], 0.0]
    tol = 32*_EPS*a[0]
    good = eligible & (t >= next_a-tol) & (t <= a+tol) & (t >= 0)
    if np.any(good):
        return float(np.clip(t[np.flatnonzero(good)[0]], 0, a[0]))
    # A flat top can put the minimizer at the last kink.
    if np.count_nonzero(a >= a[0]-tol) >= ratio*ratio:
        return float(a[0])
    lo, hi = 0.0, float(a[0])
    for _ in range(60):
        mid = (lo+hi)/2
        q = np.maximum(a-mid, 0)
        if q.sum() > ratio*np.linalg.norm(q):
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def project_intersection(x, l2, l1):
    """Euclidean projection onto ||v||2<=l2, ||v||1<=l1."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    n2, n1 = np.linalg.norm(x), a.sum()
    if n2 <= l2 and n1 <= l1:
        return x.copy()
    radial = x*min(1.0, l2/max(n2, np.finfo(float).tiny))
    if np.abs(radial).sum() <= l1:
        return radial
    order = np.sort(a)[::-1]
    tau = (np.cumsum(order)-l1)/np.arange(1, len(order)+1)
    active = np.flatnonzero(order > tau)
    threshold = max(0.0, tau[active[-1]]) if len(active) else 0.0
    one = np.sign(x)*np.maximum(a-threshold, 0)
    if np.linalg.norm(one) <= l2:
        return one
    threshold = _threshold_ratio(a, l1/l2)
    q = np.sign(x)*np.maximum(a-threshold, 0)
    if np.linalg.norm(q) == 0:  # limiting tied-coordinate projection
        tied = a >= a.max()-32*_EPS*max(a.max(), 1e-300)
        q = np.sign(x)*tied
    q *= l2/np.linalg.norm(q)
    return _feasible(q, l2, l1)


def _feasible(v, l2, l1):
    return np.asarray(v, float)*min(
        1.0, l2/max(np.linalg.norm(v), 1e-300),
        l1/max(np.abs(v).sum(), 1e-300))


def _support_upper(g, l2, l1):
    """Upper bound on sup_{feasible v} g.v, tight to scalar-solve accuracy.

    For ANY t>=0, t*l1+l2*||soft(g,t)||2 is an upper bound. Thus an
    imperfect threshold solve cannot reverse the lower-certificate inequality.
    """
    a = np.abs(g)
    if not np.any(a):
        return 0.0
    if l1 >= l2*np.sqrt(len(a)) or a.sum() <= (l1/l2)*np.linalg.norm(a):
        return float(l2*np.linalg.norm(a))
    t = _threshold_ratio(a, l1/l2)
    bound = t*l1+l2*np.linalg.norm(np.maximum(a-t, 0))
    return float(min(bound, l2*np.linalg.norm(a), l1*a.max()))


def _ball_quadratic(evals, evecs, c, radius):
    """Minimize v'Hv-2c'v on an l2 ball using all eigenvalues."""
    if radius <= 0:
        return np.zeros_like(c), 0.0
    a = evecs.T@c
    floor = max(0.0, -float(evals[0]))

    def spectral(lam):
        den = evals+lam
        if np.any((den == 0) & (a != 0)):
            return None
        return np.divide(a, den, out=np.zeros_like(a), where=den != 0)

    q = spectral(floor)
    if q is not None and np.linalg.norm(q) <= radius:
        if floor > 0:
            # Classical trust-region hard case; no spectral mode is discarded.
            j = int(np.argmin(evals))
            q[j] += np.sqrt(max(radius*radius-q@q, 0))
        return evecs@q, floor

    def equation(lam):
        v = spectral(lam)
        return np.inf if v is None else np.linalg.norm(v)-radius

    upper = floor+max(np.linalg.norm(c)/radius, 1e-15)
    while equation(upper) > 0:
        upper = floor+2*(upper-floor)
    lam = brentq(equation, floor, upper, xtol=1e-18,
                 rtol=8*_EPS, maxiter=150)
    return evecs@spectral(lam), float(lam)


def _face_candidate(H, c, v, l2, l1):
    """Solve the quadratic on the current signed l1 face and l2 ball."""
    active = np.flatnonzero(np.abs(v) > max(1e-10, 1e-9*np.linalg.norm(v)))
    k = len(active)
    if k < 2 or l1*l1 > l2*l2*k*(1+32*_EPS):
        return None
    signs = np.sign(v[active])
    base = (l1/k)*signs
    Z = np.linalg.qr(signs[:, None], mode='complete')[0][:, 1:]
    A = H[np.ix_(active, active)]
    reduced = Z.T@A@Z
    vals, vecs = np.linalg.eigh((reduced+reduced.T)/2)
    rhs = Z.T@(c[active]-A@base)
    radius = np.sqrt(max(l2*l2-l1*l1/k, 0))
    q, _ = _ball_quadratic(vals, vecs, rhs, radius)
    face = base+Z@q
    if np.any(face*signs < -1e-9):
        return None
    candidate = np.zeros_like(v)
    candidate[active] = face
    return _feasible(candidate, l2, l1)


def _active_face_refine(H, c, v, l2, l1, gap_tol, max_faces=80):
    """Add/drop signed l1-face coordinates, solving each face spectrally."""
    active = list(np.flatnonzero(np.abs(v) > max(1e-10, 1e-9*np.linalg.norm(v))))
    signs = np.sign(v[active])
    best, best_value = v.copy(), float(v@H@v-2*c@v)
    seen = set()
    faces = 0
    for faces in range(1, max_faces+1):
        key = tuple(zip(active, signs))
        if key in seen:
            break
        seen.add(key)
        k = len(active)
        if k < 2 or l1*l1 > l2*l2*k*(1+32*_EPS):
            break
        base = (l1/k)*signs
        Z = np.linalg.qr(signs[:, None], mode='complete')[0][:, 1:]
        A = H[np.ix_(active, active)]
        reduced = Z.T@A@Z
        vals, vecs = np.linalg.eigh((reduced+reduced.T)/2)
        rhs = Z.T@(c[active]-A@base)
        radius = np.sqrt(max(l2*l2-l1*l1/k, 0))
        q, lam = _ball_quadratic(vals, vecs, rhs, radius)
        face = base+Z@q
        wrong = np.flatnonzero(face*signs < -1e-9)
        if len(wrong):
            drop = wrong[np.argmin((face*signs)[wrong])]
            active.pop(int(drop))
            signs = np.delete(signs, drop)
            continue
        candidate = np.zeros_like(v)
        candidate[active] = face
        candidate = _feasible(candidate, l2, l1)
        objective = float(candidate@H@candidate-2*c@candidate)
        if objective < best_value:
            best, best_value = candidate, objective
        gradient = 2*(H@candidate-c)
        if gradient@candidate+_support_upper(gradient, l2, l1) <= gap_tol:
            return candidate, faces
        residual = c-H@candidate
        multiplier = signs@(residual[active]-lam*candidate[active])/k
        violations = np.abs(residual)-multiplier
        violations[active] = -np.inf
        add = int(np.argmax(violations))
        if violations[add] <= 1e-12:
            break
        active.append(add)
        signs = np.r_[signs, np.sign(residual[add])]
    return best, faces


def _conic_candidate(H, c, l2, l1):
    """Direct SOC quadratic formulation; coefficients and budgets unchanged."""
    if clarabel is None:
        return None, {'available': False}
    n = len(c)
    identity = sparse.eye(n, format='csc')
    zero = sparse.csc_matrix((n, n))
    P = sparse.triu(sparse.block_diag((2*H, zero), format='csc'), format='csc')
    q = np.r_[-2*c, np.zeros(n)]
    linear = sparse.vstack((sparse.hstack((identity, -identity)),
                            sparse.hstack((-identity, -identity)),
                            sparse.csc_matrix(np.r_[np.zeros(n), np.ones(n)][None, :])), format='csc')
    soc = sparse.vstack((sparse.csc_matrix((1, 2*n)),
                         sparse.hstack((-identity, zero))), format='csc')
    A = sparse.vstack((linear, soc), format='csc')
    b = np.r_[np.zeros(2*n), l1, l2, np.zeros(n)]
    cones = [clarabel.NonnegativeConeT(2*n+1), clarabel.SecondOrderConeT(n+1)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 150
    settings.max_threads = 1
    settings.tol_gap_abs = 1e-11
    settings.tol_gap_rel = 1e-11
    settings.tol_feas = 1e-11
    settings.tol_ktratio = 1e-9
    try:
        conic = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = conic.solve()
        v = np.asarray(solution.x[:n], float)
        info = {'available': True, 'version': clarabel.__version__,
                'status': str(solution.status), 'iterations': int(solution.iterations),
                'solve_seconds': float(solution.solve_time),
                'reported_primal_residual': float(solution.r_prim),
                'reported_dual_residual': float(solution.r_dual)}
        return (_feasible(v, l2, l1) if np.all(np.isfinite(v)) else None), info
    except Exception as exc:
        return None, {'available': True, 'version': clarabel.__version__,
                      'status': 'exception', 'error': str(exc)}


def solve_refit(H, c, alpha=0.0, r=2, previous=None, *, gap_tol=1e-7,
                max_iter=4000, check_every=20):
    """Return (feasible risk, coefficients, numerical certificate metadata).

    Objective: 1-2*c@v+v@H@v. Budgets are exactly 32/(1-alpha) and
    64*sqrt(r)/(1-alpha). ``gap_tol`` controls the first-order optimality gap;
    the reported full objective gap additionally includes negative-curvature
    and floating-point guards. No solver status alone is an accuracy claim.
    """
    started = time.perf_counter()
    H, c = np.asarray(H, float), np.asarray(c, float)
    if H.shape != (len(c), len(c)) or not 0 <= alpha < 1 or r < 1:
        raise ValueError('Invalid quadratic dimensions, leak, or rank')
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(c)):
        raise ValueError('Nonfinite refit moments')
    H = (H+H.T)/2
    vals, vecs = np.linalg.eigh(H)
    spectral_norm = float(np.max(np.abs(vals)))
    l2, l1 = 32/(1-alpha), 64*np.sqrt(r)/(1-alpha)
    eig_guard = 32*_EPS*max(len(c), 1)*max(spectral_norm, 1e-300)
    curvature = min(0.0, float(vals[0])-eig_guard)

    def value(v):
        return float(1-2*c@v+v@H@v)

    def certificate(v):
        risk = value(v)
        gradient = 2*(H@v-c)
        support = _support_upper(gradient, l2, l1)
        first_gap = max(0.0, float(gradient@v+support))
        curvature_penalty = -curvature*(l2+np.linalg.norm(v))**2
        roundoff = 128*_EPS*(1+abs(risk)+spectral_norm*(l2+np.linalg.norm(v))**2
                           +2*np.linalg.norm(c)*(l2+np.linalg.norm(v)))
        gap = first_gap+curvature_penalty+roundoff
        return risk, risk-gap, first_gap, curvature_penalty, roundoff

    ball, _ = _ball_quadratic(vals, vecs, c, l2)
    candidates = [np.zeros_like(c), project_intersection(ball, l2, l1)]
    if previous is not None:
        previous = np.asarray(previous, float)
        if previous.shape == c.shape and np.all(np.isfinite(previous)):
            candidates.append(project_intersection(previous, l2, l1))
    v = min(candidates, key=value)
    method = 'spectral l2 solution' if np.abs(ball).sum() <= l1*(1+1e-12) else 'projected acceleration'
    iterations = 0
    active_steps = 0
    y, momentum = v.copy(), 1.0
    L = max(2*float(vals[-1]), 2*spectral_norm*_EPS, 1e-30)
    risk, lower, first_gap, penalty, roundoff = certificate(v)
    conic_info = {'available': clarabel is not None, 'used': False}
    if first_gap > gap_tol:
        conic_v, conic_info = _conic_candidate(H, c, l2, l1)
        conic_info['used'] = True
        if conic_v is not None:
            candidates.insert(1, conic_v)
            if value(conic_v) < risk:
                v = conic_v
                risk, lower, first_gap, penalty, roundoff = certificate(v)
                method = 'clarabel SOC quadratic'
    if first_gap > gap_tol:
        for start in candidates[1:]:
            refined, steps = _active_face_refine(H, c, start, l2, l1, gap_tol)
            active_steps += steps
            if value(refined) < risk:
                v = refined
                risk, lower, first_gap, penalty, roundoff = certificate(v)
                method = 'spectral active faces'
            if first_gap <= gap_tol:
                break
        y = v.copy()
    for iterations in range(1, max_iter+1):
        if first_gap <= gap_tol:
            iterations -= 1
            break
        new = project_intersection(y-(2/L)*(H@y-c), l2, l1)
        newrisk = value(new)
        if newrisk > risk+1e-14*max(1.0, abs(risk)):
            y, momentum = v.copy(), 1.0
            new = project_intersection(v-(2/L)*(H@v-c), l2, l1)
            newrisk = value(new)
        old = v
        v = new
        if iterations % check_every == 0:
            face = _face_candidate(H, c, v, l2, l1)
            if face is not None and value(face) < newrisk:
                v = face
                newrisk = value(v)
                momentum = 1.0
            risk, lower, first_gap, penalty, roundoff = certificate(v)
        else:
            risk = newrisk
        next_momentum = (1+np.sqrt(1+4*momentum*momentum))/2
        extrapolated = v+((momentum-1)/next_momentum)*(v-old)
        if (y-v)@(v-old) > 0:
            y, momentum = v.copy(), 1.0
        else:
            y, momentum = extrapolated, next_momentum
    v = _feasible(v, l2, l1)
    risk, lower, first_gap, penalty, roundoff = certificate(v)
    success = first_gap <= gap_tol
    info = {
        'l1': float(np.abs(v).sum()), 'l2': float(np.linalg.norm(v)),
        'l1_cap': float(l1), 'l2_cap': float(l2), 'success': bool(success),
        'message': 'first-order gap tolerance met' if success else 'iteration limit; feasible bound retained',
        'method': method, 'iterations': int(iterations), 'gram_min': float(vals[0]),
        'active_face_steps': int(active_steps),
        'conic': conic_info,
        'convex_lower_bound': float(lower), 'numerical_lower_bound': float(lower),
        'objective_gap_bound': float(risk-lower), 'first_order_gap': float(first_gap),
        'negative_curvature_allowance': float(penalty), 'roundoff_allowance': float(roundoff),
        'eigenvalue_guard': float(eig_guard), 'gap_tolerance': float(gap_tol),
        'elapsed_seconds': float(time.perf_counter()-started),
    }
    return float(risk), v, info


def normalized_refit(K, C, W, b, alpha, r, previous=None, **kwargs):
    """Compatibility wrapper; zero augmented rows represent zero features."""
    rownorm = np.sqrt(np.sum(np.asarray(W)**2, axis=1)+np.asarray(b)**2)
    inverse = np.divide(1.0, rownorm, out=np.zeros_like(rownorm), where=rownorm > 0)
    H = np.asarray(K)*np.outer(inverse, inverse)
    c = np.asarray(C)*inverse
    return solve_refit(H, c, alpha, r, previous, **kwargs)
