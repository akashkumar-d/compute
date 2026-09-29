"""Pure-array numerical checks. No evaluator/model imports or calls."""
import numpy as np
from controller import NumericalFailure, directions, norm

CUTOFFS=(1e-8,1e-10,1e-12,1e-14)


def validate_ev(ev):
    arrays=[ev[k] for k in ('L','V','G','t','gP','gV','ga')]
    arrays += list(ev['Cp'].values())+list(ev['D'].values())
    if not all(np.isfinite(x).all() for x in arrays) or ev['V']<=0:
        raise NumericalFailure('Nonfinite model moments/gradients/loss or nonpositive teacher variance')
    if ev['L'] < -1e-10*ev['V']:
        raise NumericalFailure('Material negative raw population MSE; retain raw signed value')
    hd=ev.get('hermite_diagnostics')
    if hd is not None and (not hd['all_scalar_estimates_valid'] or
            np.any(hd['scalar_bank']['material_negative_energy'])):
        raise NumericalFailure('Material negative scalar residual energy; signed residuals retained')


def point_screen(m):
    if m.get('status')=='failed':
        return dict(agop_resolved=False,refit_stable=False,reason=m.get('error'))
    risks=[m['pinv'][str(c)]['actual_mse'] for c in CUTOFFS]
    residuals=[m['pinv'][str(c)]['relative_normal_residual'] for c in CUTOFFS]
    return dict(agop_resolved=bool(m['agop_full_rank_resolved']),
        refit_stable=bool(np.isfinite(risks).all() and np.isfinite(residuals).all()
            and m['equilibrated_lambda_ratio']>0 and min(risks)>=-1e-8
            and max(risks)-min(risks)<=1e-7 and max(residuals)<=1e-7),
        cutoff_spread=float(max(risks)-min(risks)),max_normal_residual=float(max(residuals)))


def weights(ev):
    G=(ev['G']+ev['G'].T)/2; scale=np.sqrt(np.maximum(np.diag(G),1e-300))
    vals,Q=np.linalg.eigh(G/scale[:,None]/scale[None,:]); q=Q.T@(ev['t']/scale)
    out={}; masks={}
    for cut in CUTOFFS:
        keep=vals>cut*vals[-1]; z=np.zeros_like(q); z[keep]=q[keep]/vals[keep]
        out[str(cut)]=Q@z/scale; masks[str(cut)]=keep
    return out,vals,masks


def risk(ev,w):
    G=(ev['G']+ev['G'].T)/2
    return float((ev['V']-2*ev['t']@w+w@G@w)/ev['V'])


def compare(reference,candidate,theta,cfg,settings):
    """Records both failures and nonapplicable resolution-dependent screens.

    reference/candidate=(ev, AGOP, canonical_cached_metrics). Missing natural
    eigengaps/refit stability do not halt otherwise valid training. They do
    prevent a qualified sequence. The full matrix check always applies.
    """
    er,Mr,mr=reference; eh,Mh,mh=candidate
    validate_ev(er); validate_ev(eh)
    if not np.isfinite(Mr).all() or not np.isfinite(Mh).all():
        raise NumericalFailure('Nonfinite AGOP')
    a=settings['audits']; r=len(cfg['c']); var=float(er['V'])
    sr,sh=point_screen(mr),point_screen(mh)
    Dr,Dh=directions(er,cfg),directions(eh,cfg)
    dn=norm(Dr); scale=max(1.,norm(theta)); derr=norm(tuple(x-y for x,y in zip(Dh,Dr)))
    if dn>0:
        unit=tuple(x/dn for x in Dr)
        slope_r=-sum(float(np.sum(er[k]*u)) for k,u in zip(('gP','gV','ga'),unit))
        slope_h=-sum(float(np.sum(eh[k]*u)) for k,u in zip(('gP','gV','ga'),unit))
    else: slope_r=slope_h=0.
    dM=Mh-Mr; frel=float(np.linalg.norm(dM)/max(np.linalg.norm(Mr),1e-300))
    err2=float(np.linalg.norm(dM,2)); vals=np.linalg.eigvalsh((Mr+Mr.T)/2)[::-1]
    gap=float(vals[r-1]-vals[r]); both_resolved=sr['agop_resolved'] and sh['agop_resolved']
    passes=dict(loss=bool(abs(eh['L']-er['L'])<=a['loss_over_variance_tolerance']*var),
        variance=bool(eh['V']==er['V']),
        direction=bool(derr<=a['weighted_direction_relative']*dn+a['gradient_absolute_factor']*cfg['m']*var/scale),
        common_slope=bool(abs(slope_h-slope_r)<=a['common_slope_relative']*abs(slope_r)+a['gradient_absolute_factor']*var/scale),
        scalar_energy=True,AGOP_matrix=bool(frel<=a['agop_F_relative']),
        AGOP_resolution_agreement=sr['agop_resolved']==sh['agop_resolved'],
        refit_stability_agreement=sr['refit_stable']==sh['refit_stable'])
    values=dict(loss_abs=float(abs(eh['L']-er['L'])),weighted_direction_relative=derr/max(dn,1e-300),
        common_slope_abs=abs(slope_h-slope_r),agop_F_relative=frel,agop_spectral_error=err2,
        reference_rank_gap=gap,agop_error_over_gap=err2/gap if gap>0 else None)
    passes['AGOP_gap']=bool(err2<=a['agop_error_over_gap']*gap) if both_resolved else None
    passes['AGOP_alignment_agreement']=bool(abs(mh['agop_Amin']-mr['agop_Amin'])<=a['alignment_absolute']
        and abs(mh['agop_A']-mr['agop_A'])<=a['alignment_absolute']) if both_resolved else None
    refits={}; both_stable=sr['refit_stable'] and sh['refit_stable']
    if both_stable:
        wr,_,_=weights(er); wh,_,_=weights(eh)
        for cut in CUTOFFS:
            k=str(cut); rr=risk(er,wr[k]); rh=risk(eh,wh[k])
            refits[k]=dict(reference_self=rr,candidate_self=rh,candidate_on_reference=risk(er,wh[k]),
                reference_on_candidate=risk(eh,wr[k]),rank_match=mr['pinv'][k]['rank']==mh['pinv'][k]['rank'])
        passes['refit_rank_agreement']=all(x['rank_match'] for x in refits.values())
        passes['refit_cross_risk']=all(max(abs(x['candidate_self']-x['reference_self']),
            abs(x['candidate_on_reference']-x['candidate_self']),
            abs(x['reference_on_candidate']-x['reference_self']))<=a['refit_normalized_risk'] for x in refits.values())
    else: passes.update(refit_rank_agreement=None,refit_cross_risk=None)
    return dict(passes=passes,passed=all(x is not False for x in passes.values()),
        numerical_comparisons=values,refit=refits,reference_screen=sr,candidate_screen=sh,
        unresolved_diagnostics_retained=bool(not both_resolved or not both_stable),
        meaning='Finite sampled numerical agreement, not a rigorous trajectory or integral certificate')
