"""Population frame geometry and interventions; copied from the reviewed gated study.

A snap preserves the component perpendicular to U, the biases, and feature order.
"""
import numpy as np
from scipy.special import ndtr
from population import moments,pdf
from refit import normalized_refit
ACUTE_BINS_DEG=np.arange(0.,95.,5.)
LINE_BINS_DEG=np.arange(0.,185.,5.)

def snap_to_frame(W, U, theta_deg):
    """Project onto the nearer unoriented frame axis; preserve U-perp exactly.

    Ties choose the first axis. Biases and feature row order are untouched.
    """
    if U.shape != (W.shape[1], 2):
        raise ValueError("Frame intervention requires a two-column teacher basis")
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    e1, e2 = c * U[:, 0] + s * U[:, 1], -s * U[:, 0] + c * U[:, 1]
    p1, p2 = W @ e1, W @ e2
    remove = np.where((np.abs(p1) >= np.abs(p2))[:, None],
                      p2[:, None] * e2, p1[:, None] * e1)
    return W - remove


def bank_moments(W, b, teacher, alpha):
    """Population K,Q,C, including exact constant/zero spatial rows.

    The standard kernel rejects zero spatial rows. Their features are constants,
    so all moments involving such rows can instead be filled analytically.
    """
    W, b = np.asarray(W, float), np.asarray(b, float)
    n = np.linalg.norm(W, axis=1)
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
        raise ValueError("Nonfinite feature bank")
    if np.any((n > 0) & (n < 1e-150)):
        raise FloatingPointError("Nonzero spatial row below stable kernel scale")
    live = n > 0
    K, Q, C = np.zeros((len(b), len(b))), np.zeros((len(b), len(b))), np.zeros(len(b))
    feature_mean = alpha * b + (1 - alpha) * np.maximum(b, 0)
    derivative_mean = alpha + (1 - alpha) * (b > 0)
    if np.any(live):
        idx = np.flatnonzero(live)
        Klive, Qlive, _, _ = moments(W[live], b[live], alpha)
        K[np.ix_(idx, idx)], Q[np.ix_(idx, idx)] = Klive, Qlive
        C[live] = teacher.cross(W[live], b[live], alpha, False)[0]
        beta = b[live] / n[live]
        feature_mean[live] = (alpha * b[live]
                             + (1 - alpha) * (n[live] * pdf(beta) + b[live] * ndtr(beta)))
        derivative_mean[live] = alpha + (1 - alpha) * ndtr(beta)
    for j in np.flatnonzero(~live):
        K[j, :] = K[:, j] = feature_mean[j] * feature_mean
        Q[j, :] = Q[:, j] = derivative_mean[j] * derivative_mean
        C[j] = feature_mean[j] * teacher.mean
    if not all(np.all(np.isfinite(z)) for z in (K, Q, C)):
        raise FloatingPointError("Nonfinite population moments")
    return K, Q, C


def moment_report(K, C, W, b, teacher, refined_teacher, alpha, normalization_W=None):
    """Audit analytic Gram and compare teacher cross moments at two orders."""
    norms = np.linalg.norm(W, axis=1)
    beta = np.divide(b, norms, out=np.zeros_like(b), where=norms > 0)
    relu2 = (norms**2 + b**2) * ndtr(beta) + b * norms * pdf(beta)
    relu2 = np.where(norms > 0, relu2, np.maximum(b, 0)**2)
    # sigma_alpha(z)^2 = alpha^2 z^2 + (1-alpha^2) ReLU(z)^2.
    diagonal = alpha**2 * (norms**2 + b**2) + (1 - alpha**2) * relu2
    base_W = W if normalization_W is None else normalization_W
    augmented = np.sqrt(np.sum(base_W**2, axis=1) + b**2)
    inverse = np.divide(1., augmented, out=np.zeros_like(augmented), where=augmented > 0)
    if refined_teacher is None:
        delta = np.zeros_like(C)
    else:
        live = norms > 0
        refined_C = np.maximum(b, 0) * teacher.mean
        if alpha:
            refined_C = (alpha * b + (1 - alpha) * np.maximum(b, 0)) * teacher.mean
        if np.any(live):
            refined_C[live] = refined_teacher.cross(W[live], b[live], alpha, False)[0]
        delta = refined_C - C
    normalized_delta = delta * inverse
    l2, l1 = 32., 64. * np.sqrt(2.)
    sensitivity = 2 * min(l2 * np.linalg.norm(normalized_delta),
                          l1 * np.max(np.abs(normalized_delta), initial=0.))
    scale = max(float(np.max(np.abs(K), initial=0.)), np.finfo(float).tiny)
    return dict(
        analytic_gram=True,
        gram_max_asymmetry=float(np.max(np.abs(K - K.T), initial=0.)),
        gram_max_relative_diagonal_error=float(np.max(np.abs(np.diag(K) - diagonal), initial=0.) / scale),
        cross_cauchy_schwarz_violation=float(max(0., np.max(C**2 - np.diag(K), initial=0.))),
        zero_spatial_rows=int(np.sum(norms == 0)),
        zero_augmented_rows=int(np.sum(augmented == 0)),
        cross_moment_method="analytic" if refined_teacher is None else "deterministic 1D quadrature",
        quadrature_orders=None if refined_teacher is None else [teacher.order, refined_teacher.order],
        cross_refinement_max_abs=float(np.max(np.abs(delta), initial=0.)),
        normalized_cross_refinement_l2=float(np.linalg.norm(normalized_delta)),
        normalized_cross_refinement_max_abs=float(np.max(np.abs(normalized_delta), initial=0.)),
        uniform_refit_objective_refinement_difference_bound=float(sensitivity),
        refinement_scope="Bound on objective change between these two computed cross-moment vectors; not a bound on quadrature error. Teacher normalization is unchanged.",
    )


def geometry(A, W, Q, U):
    norms = np.linalg.norm(W, axis=1)
    plane = W @ U
    plane_norms = np.linalg.norm(plane, axis=1)
    weights = np.abs(A) * norms
    mass = float(weights.sum())
    projected_mass = float((np.abs(A) * plane_norms).sum())
    defined = plane_norms > 0
    acute = np.rad2deg(np.arctan2(np.abs(plane[:, 1]), np.abs(plane[:, 0])))
    line = np.mod(np.rad2deg(np.arctan2(plane[:, 1], plane[:, 0])), 180.)
    acute_hist = np.histogram(acute[defined], ACUTE_BINS_DEG, weights=weights[defined])[0]
    line_hist = np.histogram(line[defined], LINE_BINS_DEG, weights=weights[defined])[0]
    fractions = np.divide(plane_norms**2, norms**2, out=np.zeros_like(norms), where=norms > 0)
    As = A / max(float(np.max(np.abs(A), initial=0.)), 1e-150)
    Ws = W / max(float(norms.max(initial=0.)), 1e-150)
    G = Ws.T @ (Q * np.outer(As, As)) @ Ws / len(A)**2
    G = (G + G.T) / 2
    vals, vectors = np.linalg.eigh(G)
    r = U.shape[1]
    top = vectors[:, -r:]
    cos2 = np.linalg.svd(U.T @ top, compute_uv=False)**2
    top_value = max(float(vals[-1]), 1e-300)
    gap = float((vals[-r] - vals[-r - 1]) / top_value)
    relative_lambda = float(vals[-r] / top_value)
    spatial_energy = float(np.sum(W * W))
    return dict(
        A_min=float(cos2.min()), A_mean=float(cos2.mean()),
        A_sub=float(np.sum(plane**2) / spatial_energy) if spatial_energy else None,
        agop_valid=bool(gap > 1e-9 and relative_lambda > 1e-9),
        relative_gap=gap, relative_lambda_r=relative_lambda,
        agop_eigen_residual=float(np.linalg.norm(G @ top - top * vals[-r:]) / top_value),
        importance_mass=mass,
        projected_importance_mass=projected_mass,
        weighted_in_plane_norm_fraction=projected_mass / mass if mass else None,
        weighted_in_plane_energy_fraction=float(weights @ fractions / mass) if mass else None,
        angle_defined_importance_fraction=float(weights[defined].sum() / mass) if mass else None,
        acute_angle_importance_fraction=(acute_hist / mass).tolist() if mass else np.zeros_like(acute_hist).tolist(),
        line_angle_importance_fraction=(line_hist / mass).tolist() if mass else np.zeros_like(line_hist).tolist(),
        zero_plane_rows=int(np.sum(~defined)),
    )


def plateau_from_losses(losses):
    if not len(losses) or not np.all(np.isfinite(losses)):
        raise ValueError("Empty or nonfinite loss history")
    low, high = np.minimum.accumulate(losses), np.maximum.accumulate(losses)
    valid = (low >= .95) & (high <= 1.05) & (high <= 1.05 * low)
    failures = np.flatnonzero(~valid)
    if len(failures) and failures[0] == 0:
        return -1, False
    return (int(failures[0] - 1), False) if len(failures) else (len(losses) - 1, True)


def refit_record(K, C, W, b, previous, gap_tol, max_iter):
    risk, vector, info = normalized_refit(K, C, W, b, 0., 2, previous=previous,
                                          gap_tol=gap_tol, max_iter=max_iter)
    return dict(feasible_risk=float(risk), numerical_lower=float(info["numerical_lower_bound"]),
                numerical_gap=float(info["objective_gap_bound"]), info=info), vector

