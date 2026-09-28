#!/usr/bin/env python3
"""Posthoc frozen-state diagnostics. Never retrains or changes source metrics."""
import argparse,json,pathlib,sys,time,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
X=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(X/'engine'))
import swsmall,swpop,vlab
CUTOFFS=[1e-8,1e-10,1e-12,1e-14]
DIAGNOSTIC_SCHEMA_VERSION = 'higher-rank-agop-v1'
AGOP_RESOLUTION_GUARD = 1e-9


def agop_geometry(M, rank, guard=AGOP_RESOLUTION_GUARD):
    """Numerical geometry for U=span(e_1,...,e_rank); no model evaluations.

    Spectral fields use the same formulas as the existing frozen-state
    high_metrics audit. Resolution flags are numerical guards, not certificates.
    Energy diagnostics require no choice of leading eigenvectors.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or not np.isfinite(M).all():
        raise ValueError('AGOP must be a finite square matrix')
    if not 1 <= rank < M.shape[0]:
        raise ValueError('Require 1 <= rank < dimension')
    M = 0.5 * (M + M.T)
    _, vectors = np.linalg.eigh(M)
    eigenvalues = np.linalg.eigvalsh(M)[::-1]
    basis = vectors[:, ::-1][:, :rank]
    cos2 = np.linalg.svd(basis[:rank, :], compute_uv=False) ** 2
    # Exactly the existing high_metrics normalizations, including the floor.
    scale = max(abs(eigenvalues[0]), 1e-300)
    following = eigenvalues[rank] if rank < len(eigenvalues) else 0.
    relative_r = float(eigenvalues[rank - 1] / scale)
    relative_gap = float((eigenvalues[rank - 1] - following) / scale)
    top_gap = float((eigenvalues[0] - eigenvalues[1]) / scale)
    psd_ratio = float(eigenvalues[-1] / scale)
    psd_ok = bool(eigenvalues[0] > 0 and psd_ratio >= -guard)
    teacher_block = M[:rank, :rank]
    teacher_values = np.linalg.eigvalsh(teacher_block)
    trace = float(np.trace(M))
    trace_positive = trace > 0
    axis_capture = np.sum(basis[:rank, :] ** 2, axis=1)
    return dict(
        agop_eigenvalues_descending=eigenvalues.tolist(),
        agop_principal_cosines_squared_descending=cos2.tolist(),
        agop_teacher_axis_captures=axis_capture.tolist(),
        agop_teacher_axis_energy_fractions=(np.diag(teacher_block) / trace).tolist() if trace_positive else None,
        agop_trace=trace,
        agop_teacher_trace_energy_fraction=float(np.trace(teacher_block) / trace) if trace_positive else None,
        agop_balanced_weakest_direction_energy=float(rank * teacher_values[0] / trace) if trace_positive else None,
        agop_teacher_block_eigenvalues_descending=teacher_values[::-1].tolist(),
        agop_rank_r_eigenvalue_relative=relative_r,
        agop_rank_r_relative_gap=relative_gap,
        agop_resolution_guard=float(guard),
        agop_psd_resolved=psd_ok,
        agop_leading_resolved=bool(psd_ok and top_gap > guard),
        agop_full_rank_resolved=bool(psd_ok and relative_r > guard and relative_gap > guard),
        agop_resolution_note='Numerical 1e-9 relative spectral/PSD guard; not an integration-error or eigenspace certificate',
    )


def metrics(E,P,V,a):
 ev=E.evaluate(P,V,a,need_grad=True)
 G0=ev['G'];G=(G0+G0.T)/2;t=ev['t'];vy=ev['V'];m=len(t)
 lam=1e-10*max(np.trace(G)/m,1e-300);wr=np.linalg.solve(G0+lam*np.eye(m),t)
 risk=lambda w:float((vy-2*t@w+w@G@w)/vy)
 scale=np.sqrt(np.maximum(np.diag(G),1e-300));C=G/scale[:,None]/scale[None,:];Ct=t/scale
 vals,Q=np.linalg.eigh(C);q=Q.T@Ct;mx=float(vals[-1]);mn=float(vals[0])
 pinv={}
 for cut in CUTOFFS:
  keep=vals>cut*mx;v=np.zeros_like(q);v[keep]=q[keep]/vals[keep];w=Q@v/scale
  pinv[str(cut)]=dict(actual_mse=risk(w),rank=int(keep.sum()),coefficient_norm=float(np.linalg.norm(w)),relative_normal_residual=float(np.linalg.norm(G@w-t)/max(np.linalg.norm(t),1e-300)))
 M=E.agop(P,V,a,ev);al=vlab.top_r_alignment(M,E.T.r);me,mq=np.linalg.eigh(M);top=mq[:,-1]
 out=dict(L=ev['L']/vy,refit_source=E.refit(ev)/vy,ridge_actual=risk(wr),ridge_lambda=float(lam),ridge_coefficient_norm=float(np.linalg.norm(wr)),pinv=pinv,equilibrated_lambda_min=mn,equilibrated_lambda_max=mx,equilibrated_lambda_ratio=mn/mx,relative_asymmetry=float(np.linalg.norm(G0-G0.T)/max(np.linalg.norm(G),1e-300)),agop_Atop=float((top[:E.T.r]**2).sum()),agop_top_relative_gap=float((me[-1]-me[-2])/max(abs(me[-1]),1e-300)),agop_psd_ratio=float(me[0]/max(abs(me[-1]),1e-300)),agop_A=al['A'],agop_Amin=al['cos2_min'],normalization_V=float(vy))
 out.update(agop_geometry(M,E.T.r))
 return out
def make_engine(cfg,mult=1):
 f,breaks=swsmall.make_link(cfg['link']);T=swpop.Teacher(f,np.array(cfg['c'],float),breaks=breaks,n_x=cfg['n_x']*mult)
 return swpop.SwiGLUPop(cfg['d'],T,alpha=cfg['alpha'],intercept='refit',bias=True,n_pair=cfg['n_pair']*mult,n_diag=cfg['n_diag']*mult,n_z=cfg['n_z']*mult)
def prefix_index(j,delta=.001):
 lh=np.array(j['Lhist']);v=np.flatnonzero((np.abs(lh[:,1]-lh[0,1])>delta)|~np.isfinite(lh[:,1]));t=lh[v[0]-1,0] if len(v) and v[0]>0 else lh[-1,0] if not len(v) else -1
 return max([i for i,r in enumerate(j['rows']) if r['t']<=t],default=0)
def run(path):
 j=json.loads(path.read_text());dest=X/'diagnostics'/path.name
 if dest.exists():print('skip',dest.name,flush=True);return
 s=np.load(path.with_name(path.stem+'_snaps.npz'));assert len(s['t'])==len(j['rows'])
 E=make_engine(j['cfg']);rows=[];start=time.time()
 failure_controls=X/'numerical_failure_controls.json';failure=json.loads(failure_controls.read_text()).get(path.stem,{}) if failure_controls.exists() else {}
 failure_step=failure.get('first_safeguard_step',float('inf'))
 for i,r in enumerate(j['rows']):
  assert abs(float(s['t'][i])-r['t'])<1e-9
  if r['step']>=failure_step:
   rows.append(dict(index=i,t=r['t'],step=r['step'],status='skipped_numerical_failure_tail',error='preserved raw tail after nominal fixed-step safeguard activated; no corrected-risk evaluation',refit_source=r.get('refit'),ridge_actual=None,pinv={}))
   continue
  try:
   z=dict(status='ok',**metrics(E,s['P'][i],s['V'][i],s['a'][i]))
  except Exception as exc:
   z=dict(status='failed',error=repr(exc),ridge_actual=None,refit_source=None,pinv={})
  rows.append(dict(index=i,t=r['t'],step=r['step'],**z))
  if i%10==0:print(path.stem,i,len(j['rows']),round(time.time()-start,1),flush=True)
 out=dict(tag=path.stem,cfg=j['cfg'],rows=rows,cutoffs=CUTOFFS,definition='prediction MSE normalized by teacher variance; source field preserved separately',wall_seconds=time.time()-start)
 if j['cfg']['seed']>=3 or j['cfg']['m']==256 or path.stem=='h3_d16m64_seed0' or path.stem.startswith('r2_') or path.stem in ['fixed_h3_dt0p1_d16m64_seed0','fixed_h3_dt0p05_d16m64_seed0']:
  order2=make_engine(j['cfg'],2);ix=(sorted(set([len(rows)-1,max(i for i,r in enumerate(j['rows']) if r['step']<=3000 and r['t']<=500)])) if path.stem.startswith('r2_') else [len(rows)-1] if path.stem.startswith('fixed_') else sorted(set([0,prefix_index(j),len(rows)-1])));out['quadrature_double']=[]
  for i in ix:
   try:
    z=dict(status='ok',**metrics(order2,s['P'][i],s['V'][i],s['a'][i]))
   except Exception as exc:
    z=dict(status='failed',error=repr(exc),ridge_actual=None,refit_source=None,pinv={})
   out['quadrature_double'].append(dict(index=i,t=j['rows'][i]['t'],step=j['rows'][i]['step'],**z))
 dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.rename(dest)
 print('done',path.name,'rows',len(rows),'seconds',time.time()-start,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('tags',nargs='*');a=ap.parse_args()
 paths=[X/'runs'/(n if n.endswith('.json') else n+'.json') for n in a.tags] if a.tags else sorted((X/'runs').glob('*.json'))
 import fcntl
 (X/'diagnostics').mkdir(exist_ok=True)
 for p in paths:
  with (X/'diagnostics'/(p.stem+'.lock')).open('a') as lock:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
   run(p)
