"""Stable diagnostics for f(x)=sum_j A_j sigma_alpha(W_j x+b_j)/m.

No training, centering of AGOP, PSD projection, ridge, or rank substitution.
Population AGOP is assembled as E[J] E[J]^T + Cov(J), algebraically the
same uncentered second moment. Rare gate events avoid subtracting 1-p from
one when a gate is almost surely on. Signed contractions use longdouble
where the platform supports it. Adaptive one-dimensional integration is
deterministic, but its reported error is an estimate, not a certificate.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import ndtr, log_ndtr, ndtri_exp, owens_t
from scipy.spatial.distance import cdist

EPS = np.finfo(float).eps
LD = np.longdouble


def _joint_negative(a, b, rho, root, tolerance):
    """P(Z<=a,Y<=b), a,b<=0, without complementary-CDF subtraction.

    Integrate conditional probability over a uniformly parameterized
    truncated Gaussian: Phi(a) int_0^1 Phi((b-rho Phi^-1(u Phi(a)))/root) du.
    The smaller marginal is integrated. Extreme underflow is reported.
    """
    if a > b:
        a, b = b, a
    logp = float(log_ndtr(a)); p = float(np.exp(logp))
    if p == 0:
        return 0., 0., False
    if root == 0:
        return (p if rho >= 0 else 0.), 0., False
    if rho == 0:
        q = p*ndtr(b)
        return float(q), float(8*EPS*q), False
    def integrand(u):
        z = ndtri_exp(logp+np.log(u))
        return float(ndtr((b-rho*z)/root))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always', IntegrationWarning)
        val, error = quad(integrand, 0., 1., epsabs=tolerance,
                          epsrel=tolerance, limit=200,
                          points=[.001,.01,.1,.5,.9,.99,.999])
    return p*val, p*error+16*EPS*p, bool(observed)


def _rare_joint_matrix(threshold,unit,p,tolerance,backend):
    """Vectorized Owen-T fast route, adaptive cancellation fallback.

    Error estimates track cancellation in the explicit formula. They do not
    constitute interval guarantees on scipy special-function evaluation.
    """
    m=len(p);dm=cdist(unit,unit,'sqeuclidean')/2;dp=cdist(unit,-unit,'sqeuclidean')/2
    rho=np.where(dm<=dp,1-dm,dp-1);rho=np.clip(rho,-1,1)
    root=np.sqrt(np.maximum(dm*dp,0.));safe=np.maximum(root,1e-150)
    a=np.where(p>0,threshold,0.)[:,None];b=a.T
    aa=np.where(np.abs(a)<1e-14,-1e-14,a);bb=aa.T
    offab=np.where(rho>=0,(b-a)+dm*a,(b+a)-dp*a)
    offba=np.where(rho>=0,(a-b)+dm*b,(a+b)-dp*b)
    o1=owens_t(a,offab/(aa*safe));o2=owens_t(b,offba/(bb*safe))
    joint=(p[:,None]+p[None,:])/2-o1-o2
    errors=32*EPS*((p[:,None]+p[None,:])/2+np.abs(o1)+np.abs(o2))
    za=np.abs(a)<1e-14;zb=np.abs(b)<1e-14
    joint=np.where(za,ndtr(b)/2+owens_t(b,rho/safe),joint)
    joint=np.where(zb,ndtr(a)/2+owens_t(a,rho/safe),joint)
    joint=np.where(za&zb,np.arctan2(root,-rho)/(2*np.pi),joint)
    joint=np.where((root==0)&(rho>=0),np.minimum(p[:,None],p[None,:]),joint)
    joint=np.where((root==0)&(rho<0),0.,joint)
    live=(p[:,None]>0)&(p[None,:]>0)
    joint=np.where(live,joint,0.);errors=np.where(live,errors,0.)
    adaptive=live&((joint<128*errors)|(joint>np.minimum(p[:,None],p[None,:])+errors))
    if backend=='adaptive':adaptive=live.copy()
    elif backend!='auto':raise ValueError("pair_backend must be 'auto' or 'adaptive'")
    count=0;warning_count=0
    for i,j in np.argwhere(np.tril(adaptive,-1)):
        val,error,warning=_joint_negative(threshold[i],threshold[j],rho[i,j],root[i,j],tolerance)
        joint[i,j]=joint[j,i]=val;errors[i,j]=errors[j,i]=error
        count+=1;warning_count+=warning
    joint=(joint+joint.T)/2
    np.fill_diagonal(joint,p);np.fill_diagonal(errors,8*EPS*p)
    return joint,errors,count,warning_count


def spectral_summary(G, U, assembly_error_estimate=0., safety=10.):
    """Raw principal-angle scores plus explicitly noncertified resolution screen."""
    G=np.asarray(G,float); U=np.asarray(U,float); d=G.shape[0]; r=U.shape[1]
    if G.shape!=(d,d) or U.shape[0]!=d or not 1<=r<d:
        raise ValueError('Require square G and 1<=teacher rank<dimension')
    if np.linalg.norm(U.T@U-np.eye(r))>1e-9:
        raise ValueError('U must be orthonormal; do not silently change teacher basis')
    S=(G+G.T)/2
    vals,vecs=np.linalg.eigh(S); top=vecs[:,-r:]
    scores=np.linalg.svd(U.T@top,compute_uv=False)**2
    scale=float(np.max(np.abs(vals)))
    arithmetic=64*EPS*max(scale,np.finfo(float).tiny)
    perturbation=max(float(assembly_error_estimate),arithmetic)
    gap=float(vals[-r]-vals[-r-1]); threshold=safety*perturbation
    substantial_negative=bool(vals[0]<-threshold)
    resolved=bool(vals[-r]>threshold and gap>threshold and not substantial_negative)
    report=dict(A_min=float(scores.min()),A_mean=float(scores.mean()),
        eigenvalues_ascending=vals.tolist(),lambda_r=float(vals[-r]),
        lambda_next=float(vals[-r-1]),boundary_gap=gap,
        numerical_rank=int(np.sum(vals>threshold)),
        substantial_negative_eigenvalue=substantial_negative,
        assembly_error_estimate=float(assembly_error_estimate),
        arithmetic_scale_estimate=float(arithmetic),resolution_threshold=float(threshold),
        numerical_screen_passed=resolved,
        status='numerical_diagnostic_only_not_population_certificate',
        raw_scores_retained=True,
        eigen_residual_operator=float(np.linalg.norm(S@top-top*vals[-r:],2)))
    return report,top


def population_agop(A,W,b,U,alpha=0.,*,pair_tolerance=2e-11,pair_backend='auto',safety=10.):
    """Return (same-definition population G, metadata, top-r raw basis).

    Auto uses vectorized Owen-T probabilities and adaptive quadratures only
    for cancellation-prone pairs. 'adaptive' audits every live pair at higher
    cost. No old event flags are changed.
    """
    A=np.asarray(A,float);W=np.asarray(W,float);b=np.asarray(b,float)
    m,d=W.shape
    if A.shape!=(m,) or b.shape!=(m,) or not 0<=alpha<1:
        raise ValueError('Invalid shapes/leak')
    if not all(np.isfinite(v).all() for v in (A,W,b)):
        raise ValueError('Nonfinite checkpoint')
    norms=np.linalg.norm(W,axis=1); live=norms>0
    beta=np.divide(b,norms,out=np.zeros(m),where=live)
    # A zero spatial row contributes identically zero to the input gradient.
    beta=np.where(live,beta,np.where(b>=0,np.inf,-np.inf))
    positive=beta>=0
    orientation=np.where(positive,1.,-1.)
    threshold=-np.abs(beta)
    logs=log_ndtr(threshold); p=ndtr(threshold)
    base=np.where(positive,1.,alpha)
    correction=-(1-alpha)*orientation
    unit=np.divide(W,norms[:,None],out=np.zeros_like(W),where=live[:,None])
    rare_unit=unit*orientation[:,None]
    joint,errors,adaptive_count,warnings_count=_rare_joint_matrix(threshold,rare_unit,p,pair_tolerance,pair_backend)
    cov=joint.astype(LD)-np.outer(p.astype(LD),p.astype(LD))
    B=(A.astype(LD)*correction.astype(LD))[:,None]*W.astype(LD)/LD(m)
    mean=((A.astype(LD)*base.astype(LD))[:,None]*W.astype(LD)).sum(axis=0)/LD(m)
    mean+=(p.astype(LD)[:,None]*B).sum(axis=0)
    Gld=np.outer(mean,mean)+B.T@cov@B
    G=np.asarray((Gld+Gld.T)/2,float)
    # Propagate quadrature estimates and roundoff with absolute contractions.
    absB=np.abs(B)
    matrix_error=absB.T@(errors.astype(LD)+LD(32*EPS)*np.outer(p.astype(LD),p.astype(LD)))@absB
    absolute_terms=np.outer(np.abs(mean),np.abs(mean))+absB.T@np.abs(cov)@absB
    roundoff=LD(64*m*np.finfo(LD).eps)*absolute_terms
    # Crucial when mean-gradient terms cancel: its error scales with the
    # absolute summands, NOT with the tiny resulting mean. This prevents a
    # nearly constant saturated model from acquiring a spurious precision claim.
    mean_absolute=(np.abs(A.astype(LD)*base.astype(LD))[:,None]*np.abs(W.astype(LD))).sum(axis=0)/LD(m)
    mean_absolute+=(p.astype(LD)[:,None]*absB).sum(axis=0)
    mean_error=LD(16*m*np.finfo(LD).eps)*mean_absolute
    mean_error+=LD(8*EPS)*(p.astype(LD)[:,None]*absB).sum(axis=0)
    mean_matrix_error=np.outer(np.abs(mean),mean_error)+np.outer(mean_error,np.abs(mean))+np.outer(mean_error,mean_error)
    err=float(np.linalg.norm(np.asarray(matrix_error+roundoff+mean_matrix_error,float),2))
    report,top=spectral_summary(G,U,err,safety)
    report.update(method='modal_rare_gate_mean_plus_covariance',
        alpha=float(alpha),width=m,dimension=d,pair_tolerance=pair_tolerance,pair_backend=pair_backend,adaptive_pair_count=adaptive_count,
        quadrature_warning_count=warnings_count,
        longdouble_mantissa_bits=int(np.finfo(LD).nmant),
        beta_min=float(np.min(beta[live])) if live.any() else None,beta_max=float(np.max(beta[live])) if live.any() else None,
        constant_spatial_row_count=int(np.sum(~live)),
        rare_probability_min=float(p.min()),rare_probability_max=float(p.max()),
        underflowed_rare_probability_count=int(np.sum((p==0)&live)),
        largest_underflowed_log_probability=float(np.max(logs[(p==0)&live])) if np.any((p==0)&live) else None,
        mean_input_gradient_squared=float(mean@mean),
        mean_input_gradient_error_estimate=float(np.linalg.norm(np.asarray(mean_error,float))),
        quadrature_covariance_error_estimate=float(np.linalg.norm(np.asarray(matrix_error,float),2)),
        mean_input_gradient=[float(x) for x in mean],
        uncertainty='Quadrature/error estimates are empirical numerical diagnostics, not rigorous bounds. No PSD clipping, centering, ridge, or rank substitution is used.')
    if warnings_count:
        report['numerical_screen_passed']=False
    return G,report,top


def analytic_jacobian(A,W,b,X,alpha=0.):
    """Float64 gates, extended-precision signed sums; no gradient centering."""
    A=np.asarray(A,float);W=np.asarray(W,float);b=np.asarray(b,float);X=np.asarray(X,float)
    gates=alpha+(1-alpha)*(X@W.T+b>0)
    return np.asarray((gates.astype(LD)*A.astype(LD))@W.astype(LD)/LD(len(A)),float)


def fixed_checkpoint_svd(A,W,b,U,*,alpha=0.,sample_sizes=(2048,8192),seeds=(8101,8102),population_G=None):
    """Independent Gaussian sample/SVD checks; deliberately no certified claim."""
    results=[];d=np.asarray(W).shape[1];r=np.asarray(U).shape[1]
    for n in sample_sizes:
        for seed in seeds:
            # Distinct seeds across sizes avoid pretending nested samples independent.
            X=np.random.default_rng(seed+1000003*n).normal(size=(n,d))
            J=analytic_jacobian(A,W,b,X,alpha);B=J/np.sqrt(n)
            left,s,vt=np.linalg.svd(B,full_matrices=False)
            top=vt[:r].T; scores=np.linalg.svd(U.T@top,compute_uv=False)**2
            G=B.T@B;gram,Q=spectral_summary(G,U)
            item=dict(n=n,seed=seed+1000003*n,A_min_svd=float(scores.min()),
                A_mean_svd=float(scores.mean()),singular_values=s.tolist(),
                rank_at_svd_roundoff=int(np.sum(s>max(n,d)*EPS*s[0])) if len(s) and s[0]>0 else 0,
                svd_reconstruction_relative=float(np.linalg.norm(B-(left*s)@vt)/max(np.linalg.norm(B),np.finfo(float).tiny)),
                gram=gram,svd_gram_projector_difference=float(np.linalg.norm(top@top.T-Q@Q.T,2)),
                population_matrix_difference_operator=float(np.linalg.norm(G-population_G,2)) if population_G is not None else None)
            results.append(item)
    return dict(method='independent_float64_Gaussian_jacobian_SVD',samples=results,
        uncertainty='Finite diagnostic samples can miss rare gates and cannot certify population recovery. Full-r raw scores at rank-deficient checkpoints are arbitrary.')


def refit_uncertainty_report(info,scientific_tolerance=.001):
    """Separate optimizer accuracy from moment/roundoff allowances.

    Does not change the refit problem, coefficients, original flags or bounds.
    A gap larger than 1e-7 can still be negligible for a 0.1 MSE effect.
    """
    gap=float(info['objective_gap_bound']);first=float(info['first_order_gap'])
    return dict(first_order_gap=first,negative_curvature_allowance=float(info['negative_curvature_allowance']),
        roundoff_allowance=float(info['roundoff_allowance']),total_numerical_gap=gap,
        original_requested_tolerance=float(info['gap_tolerance']),
        optimizer_tolerance_met=first<=float(info['gap_tolerance']),
        scientific_tolerance=float(scientific_tolerance),
        gap_small_relative_to_scientific_tolerance=gap<=scientific_tolerance,
        interpretation='Preserves original fixed-budget objective and numerical interval; the scientific tolerance is an explicitly separate descriptive comparison, not a relaxed certificate.')


def stable_geometry(A,W,b,U,Q=None,alpha=0.,**kwargs):
    """Runner adapter; preserves raw metric names and adds nested diagnostics.

    ``agop_valid`` means the explicitly noncertified numerical screen passed,
    not that population recovery has been proved. Q is used only to compare
    the old direct assembly; it never controls the new output.
    """
    G,report,top=population_agop(A,W,b,U,alpha,**kwargs)
    vals=np.asarray(report['eigenvalues_ascending']);r=U.shape[1]
    scale=max(float(vals[-1]),np.finfo(float).tiny)
    scalesq=max(float(np.max(np.abs(A))),1e-150)**2*max(float(np.max(np.linalg.norm(W,axis=1))),1e-150)**2
    spatial=float(np.sum(W*W));trace=float(np.trace(G));restricted=U.T@G@U
    _,V=np.linalg.eigh(G)
    if Q is not None:
        old=W.T@(np.asarray(Q)*np.outer(A,A))@W/len(A)**2
        report['old_direct_assembly_difference_operator']=float(np.linalg.norm(G-old,2))
        report['old_direct_eigenvalues_ascending']=np.linalg.eigvalsh((old+old.T)/2).tolist()
    result=dict(A_min=report['A_min'],A_mean=report['A_mean'],
        A_sub=float(np.sum((W@U)**2)/spatial) if spatial else None,
        A_top=float(np.sum((U.T@V[:,-1])**2)),
        principal_cosines_sq_ascending=np.sort(np.linalg.svd(U.T@top,compute_uv=False)**2).tolist(),
        teacher_direction_captures=np.sum((U.T@top)**2,axis=1).tolist(),
        agop_teacher_energy_fraction=float(np.trace(restricted)/trace) if trace>0 else None,
        balanced_weak_direction_energy=float(r*np.linalg.eigvalsh((restricted+restricted.T)/2)[0]/trace) if trace>0 else None,
        agop_energy_valid=trace>0,
        agop_valid=bool(report['numerical_screen_passed'] and report['boundary_gap']/scale>1e-9 and report['lambda_r']/scale>1e-9),
        relative_gap=report['boundary_gap']/scale,relative_lambda_r=report['lambda_r']/scale,
        relative_lambda_2=float(vals[-2]/scale),
        agop_eigen_residual=report['eigen_residual_operator']/scale,
        agop_scaled_eigenvalues=(vals/scalesq).tolist(),agop_rank=r,
        agop_diagnostic=report)
    result['agop_diagnostic']['original_ratio_screens_passed']=bool(report['boundary_gap']/scale>1e-9 and report['lambda_r']/scale>1e-9)
    result['agop_diagnostic']['adapter_validity_rule']='New numerical-error screen AND both original 1e-9 rank/gap ratio screens; no old threshold loosened.'
    return result
