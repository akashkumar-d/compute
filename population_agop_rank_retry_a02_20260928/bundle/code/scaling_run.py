"""Bounded explicitly authorized local population follow-up; never overwrites old runs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib,json,signal,socket,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from population import Teacher,loss_force,pdf
from diagnostic_priority import diagnostic_order
from refit import normalized_refit
from frames import bank_moments,plateau_from_losses
ROOT=Path(__file__).resolve().parents[1]
KERNELS=['population.py','teachers_extra.py','teachers_product.py','teachers_smooth.py','diagnostic_priority.py','canonical14_moments.json','refit.py','frames.py']
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'diagnostics'))
from stable_relu import stable_geometry,refit_uncertainty_report


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def npz(filename,**values):
    tmp=filename.with_name(filename.name+'.tmp')
    with tmp.open('wb') as f:np.savez(f,**values)
    os.replace(tmp,filename)
def source_hashes():return {**{name:sha(CODE/name) for name in KERNELS+['scaling_run.py']},'diagnostics/stable_relu.py':sha(ROOT/'diagnostics/stable_relu.py')}
def checkpoints(end):return sorted(set(np.geomspace(1,max(end,1),32).astype(int))|{0,end})
def require_server():
    if os.environ.get('AGOP_EXECUTION_SITE') not in ['LOCAL_AUTHORIZED','SERVER']:
        raise RuntimeError('Explicit LOCAL_AUTHORIZED or SERVER execution site required')
def config_check(config):
    c=dict(config)
    for k in ['id','teacher','seed','m','d','r','scale','h','steps']:
        if k not in c:raise ValueError('Missing config field '+k)
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c['id']):raise ValueError('Unsafe id')
    if c['teacher'] not in ['h3','damped_2_3','three_atom','h3_plus_h5','h3_plus_sine','h2','h4','h5','sine','relu','abs','leaky_relu','softplus','silu','gelu','tanh','erf','gaussian_rbf']:raise ValueError('Teacher is outside the declared additive breadth pilot')
    if c.get('student','relu')!='relu' or c.get('optimizer','gd')!='gd':raise ValueError('ReLU simultaneous GD only')
    for k in ['seed','m','d','r','steps']:
        if isinstance(c[k],bool) or not isinstance(c[k],int) or c[k]<1:raise ValueError('Invalid '+k)
    if c['r'] not in [2,4,8,16] or c['d']<2*c['r'] or c['m']<c['r']:raise ValueError('Require r in2,4,8,16; d>=2r; m>=r')
    if not all(np.isfinite(c[k]) and c[k]>0 for k in ['scale','h']):raise ValueError('Positive finite scale,h required')
    for key in ['force_time_max','loss_stop']:
        if key in c and (not np.isfinite(c[key]) or c[key]<=0):raise ValueError('Positive finite '+key+' required')
    if 'diagnostic_stride' in c and (not isinstance(c['diagnostic_stride'],int) or c['diagnostic_stride']<1):raise ValueError('Positive integer diagnostic stride required')
    if 'diagnostic_priority_fractions' in c:
        fs=c['diagnostic_priority_fractions']
        if not isinstance(fs,list) or any(isinstance(f,bool) or not isinstance(f,(int,float)) or not np.isfinite(f) or not 0<f<1 for f in fs):raise ValueError('diagnostic_priority_fractions must be a list of finite fractions strictly between0and1')
    return c


def geometry(A,W,Q,U):
    """Full cross-neuron AGOP; rank-r eigenspace, identical scalar scaling to reviewed code."""
    norms=np.linalg.norm(W,axis=1)
    As=A/max(float(np.max(np.abs(A),initial=0.)),1e-150)
    Ws=W/max(float(norms.max(initial=0.)),1e-150)
    G=Ws.T@(Q*np.outer(As,As))@Ws/len(A)**2
    G=(G+G.T)/2
    vals,vecs=np.linalg.eigh(G);r=U.shape[1];top=vecs[:,-r:]
    cos2=np.linalg.svd(U.T@top,compute_uv=False)**2
    scale=max(float(vals[-1]),1e-300)
    gap=float((vals[-r]-vals[-r-1])/scale);relative=float(vals[-r]/scale)
    # Direction captures refer to the supplied teacher basis; the principal
    # cosines and energy summaries are invariant to rotations within that basis.
    captures=np.sum((U.T@top)**2,axis=1)
    restricted=U.T@G@U;restricted=(restricted+restricted.T)/2
    trace=float(np.trace(G));energy_valid=bool(np.isfinite(trace) and trace>0.)
    teacher_energy=float(np.trace(restricted)/trace) if energy_valid else None
    weak_energy=float(r*np.linalg.eigvalsh(restricted)[0]/trace) if energy_valid else None
    return dict(A_min=float(cos2.min()),A_mean=float(cos2.mean()),
        principal_cosines_sq_ascending=np.sort(cos2).tolist(),teacher_direction_captures=captures.tolist(),
        agop_teacher_energy_fraction=teacher_energy,balanced_weak_direction_energy=weak_energy,
        agop_energy_valid=energy_valid,
        A_sub=float(np.sum((W@U)**2)/max(float(np.sum(W*W)),1e-300)),
        A_top=float(np.sum((U.T@vecs[:,-1])**2)),
        agop_valid=bool(gap>1e-9 and relative>1e-9),relative_gap=gap,relative_lambda_r=relative,
        relative_lambda_2=float(vals[-2]/scale),agop_eigen_residual=float(np.linalg.norm(G@top-top*vals[-r:])/scale),
        agop_scaled_eigenvalues=vals.tolist(),agop_rank=r)


class BudgetExpired(Exception):pass


def execute(config,out,max_seconds=120.,deadline_utc=None,diagnostic_reserve=20.,resume=False):
    require_server();c=config_check(config);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config_hash=digest(c);sources=source_hashes();started=time.monotonic();wall=time.time()
    budget=float(max_seconds)
    if not np.isfinite(budget) or budget<=0 or not np.isfinite(diagnostic_reserve) or diagnostic_reserve<0:raise ValueError('Invalid wall-time budget/reserve')
    if deadline_utc:
        absolute=datetime.fromisoformat(deadline_utc.replace('Z','+00:00')).timestamp()
        budget=min(budget,absolute-wall)
    if budget<=0:raise ValueError('Run budget/deadline already exhausted')
    hard=started+budget;train_until=max(started,hard-min(diagnostic_reserve,budget*.4))
    state={'phase':'training','stop':None}
    def on_signal(signum,frame):
        state['stop']='signal_'+str(signum)
        if state['phase']=='diagnostics':raise BudgetExpired(state['stop'])
    previous_handlers={s:signal.signal(s,on_signal) for s in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]}
    attempt=str(time.time_ns())
    try:
        meta_path=out/'meta.json'
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta['config_sha256']!=config_hash or meta['source_sha256']!=sources:raise ValueError('Existing source/config mismatch')
            if (out/'DONE.json').exists():
                done=json.loads((out/'DONE.json').read_text())
                if done['complete']:
                    if any(sha(out/name)!=h for name,h in done['output_sha256'].items()):raise ValueError('Completed output changed')
                    return done
            if not resume:raise ValueError('Existing partial run; use --resume to preserve and continue it')
            for name in ['DONE.json','TRAIN_RESULT.json','rows.jsonl','FAILED.json']:
                if (out/name).exists():
                    dest=out/'attempts'/attempt/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((out/name).read_bytes())
        else:
            meta=dict(config=c,config_sha256=config_hash,source_sha256=sources,execution_site=os.environ['AGOP_EXECUTION_SITE'],
                hostname=socket.gethostname(),started_utc=utc(),student='relu',optimizer='gd',
                teacher=c['teacher'],seed=c['seed'],m=c['m'],d=c['d'],r=c['r'],scale=c['scale'],h=c['h'],steps=c['steps'],
                initialization='All raw A,W,b IID N(0,scale^2/d); all blocks jointly trained',
                profile_intercept=bool(c.get('profile_intercept',False)),
                plateau_rule='All-update initial positive-loss prefix: max/min<=1.05. Original near-unit rule retained separately; baseline is Var(Y) with profiled intercept, otherwise E[Y^2]=1',
                refit_caps={'l2':32.,'l1':float(64*np.sqrt(c['r']))},refit_gap_tolerance=1e-7,refit_max_iterations=4000,
                armijo_c=1e-4,armijo_factor=.5,armijo_roundoff_tolerance=1e-13,max_backtracks=40,
                declared_force_time_max=c.get('force_time_max'),declared_loss_stop=c.get('loss_stop'),diagnostic_stride=c.get('diagnostic_stride'),additional_strict_prefix_ratio=1.01,python_version=sys.version.split()[0],numpy_version=np.__version__)
            atomic(meta_path,meta)
        teacher=Teacher(c['teacher'],c['r'],c['d']);m=c['m']
        meta['teacher_moment_method']=getattr(teacher,'numerical_method',{'method':'existing exact Gaussian cross kernel'})
        atomic(meta_path,meta)
        profiled=bool(c.get('profile_intercept',False))
        baseline=1-teacher.mean**2 if profiled else 1.
        def evaluate(A,W,b,teacher,alpha):
            return loss_force(A,W,b,teacher,alpha,profile_intercept=profiled)
        def prefix(values):
            values=np.asarray(values)
            good=(values>0)&np.isfinite(values)&(np.maximum.accumulate(values)<=1.05*np.minimum.accumulate(values))
            bad=np.flatnonzero(~good)
            return (int(bad[0]-1),False) if len(bad) else (len(values)-1,True)
        rng=np.random.default_rng(c['seed']);sd=c['scale']/np.sqrt(c['d'])
        A,W,b=rng.normal(0,sd,m),rng.normal(0,sd,(m,c['d'])),rng.normal(0,sd,m)
        history=[];path=[];accepted=[];backtracks=[];states={};trial_h=c['h'];displacement=0.;start=0;prev=None
        state_dir=out/'states';state_dir.mkdir(exist_ok=True)
        def save_state(step,values):
            file=state_dir/f'step{step:08d}.npz'
            if not file.exists():npz(file,**dict(zip(['A','W','b'],values)))
            states[step]=file
        def load_state(step):
            with np.load(states[step],allow_pickle=False) as z:return tuple(z[k].copy() for k in ['A','W','b'])
        if resume and (out/'resume.npz').exists():
            with np.load(out/'resume.npz',allow_pickle=False) as z:
                if str(z['config_sha256'])!=config_hash:raise ValueError('Resume hash mismatch')
                A,W,b=(z[k].copy() for k in ['A','W','b']);start=int(z['step'])
                history=z['history'].tolist()[:-1];path=z['path'].tolist()[:-1]
                accepted=z['accepted'].tolist();backtracks=z['backtracks'].tolist()
                trial_h=float(z['trial_h']);displacement=float(z['displacement'])
                states={int(n):state_dir/f'step{int(n):08d}.npz' for n in z['saved_steps']}
                if start:prev=tuple(z['prev_'+k].copy() for k in ['A','W','b'])
        first_exit=None
        if history:
            prior_end,prior_censored=prefix(history)
            first_exit=None if prior_censored else prior_end+1
        lo=min(history,default=np.inf);hi=max(history,default=-np.inf);cached=None;stop_reason=None
        def snapshot(n):
            arrays={}
            if prev is not None:arrays.update({'prev_'+k:v for k,v in zip(['A','W','b'],prev)})
            npz(out/'resume.npz',A=A,W=W,b=b,step=n,config_sha256=config_hash,history=np.array(history),path=np.array(path),
                accepted=np.array(accepted),backtracks=np.array(backtracks),trial_h=trial_h,displacement=displacement,
                saved_steps=np.array(sorted(states)),**arrays)
        required=set(checkpoints(c['steps']))
        if c.get('diagnostic_stride'):required.update(range(0,c['steps']+1,c['diagnostic_stride']))
        # Capture actual loss-only 1% and 5% boundaries, without feature selection.
        strict_exit=None
        if history:
            bad=np.flatnonzero(np.maximum.accumulate(history)>1.01*np.minimum.accumulate(history))
            if len(bad):strict_exit=int(bad[0])
        force_clock=float(np.sum(accepted))
        for n in range(start,c['steps']+1):
            if cached is None:cached=evaluate(A,W,b,teacher,0.)
            L,forces,*_=cached
            if not np.isfinite(L) or any(not np.isfinite(z).all() for z in [A,W,b]):raise FloatingPointError('Nonfinite training state')
            history.append(float(L));path.append(displacement);lo=min(lo,L);hi=max(hi,L)
            crossed=first_exit is None and (L<=0 or hi>1.05*lo)
            if crossed:
                first_exit=n
                if n:save_state(n-1,prev)
            if strict_exit is None and (L<=0 or hi>1.01*lo):
                strict_exit=n
                if n:save_state(n-1,prev)
                save_state(n,(A,W,b))
            horizon_stop=('force_time_horizon' if force_clock>=c.get('force_time_max',float('inf')) else 'loss_target' if L<=c.get('loss_stop',-float('inf')) else None)
            stop_reason=state['stop'] or horizon_stop or ('training_budget' if time.monotonic()>=train_until and n<c['steps'] else None)
            if n in required or crossed or stop_reason:save_state(n,(A,W,b))
            if n%100==0 or stop_reason or n==c['steps']:snapshot(n)
            if n%500==0:print(json.dumps(dict(phase='training',step=n,loss=float(L),elapsed=time.monotonic()-started)),flush=True)
            if n==c['steps'] or stop_reason:break
            prev=(A.copy(),W.copy(),b.copy());norm2=sum(float(np.sum(z*z)) for z in forces)
            for bt in range(41):
                updates=[trial_h*z for z in forces]
                candidate=evaluate(A+updates[0],W+updates[1],b+updates[2],teacher,0.)
                allowance=1e-13*max(1.,abs(float(L)))
                if np.isfinite(candidate[0]) and candidate[0]<=L-1e-4*(2*trial_h/m)*norm2+allowance:
                    cached=candidate;accepted.append(trial_h);backtracks.append(bt);break
                trial_h*=.5
            else:raise FloatingPointError('Armijo exhausted40backtracks')
            force_clock+=trial_h
            displacement+=np.sqrt(sum(float(np.sum(z*z)) for z in updates))
            A+=updates[0];W+=updates[1];b+=updates[2]
        observed=n;endpoint,censored=prefix(history);save_state(n,(A,W,b))
        atomic(out/'checkpoint_manifest.json',{str(step):dict(path=str(file.relative_to(out)),sha256=sha(file)) for step,file in sorted(states.items())})
        for name,value in [('loss_every_step',history),('parameter_path_length',path),('accepted_h',accepted),('backtracking_counts',backtracks),('gd_cumulative_time',np.r_[0.,np.cumsum(accepted)*m/2])]:np.save(out/(name+'.npy'),np.asarray(value))
        train=dict(complete=observed==c['steps'] or stop_reason in ['force_time_horizon','loss_target'],observed_steps=observed,planned_steps=c['steps'],
            training_censored=observed<c['steps'] and stop_reason not in ['force_time_horizon','loss_target'],declared_horizon_or_loss_target_reached=stop_reason in ['force_time_horizon','loss_target'],force_clock=force_clock,strict_prefix_end=strict_exit-1 if strict_exit is not None else observed,strict_prefix_right_censored=strict_exit is None,stop_reason=stop_reason,plateau_end=endpoint,
            plateau_right_censored=censored,first_exit_step=first_exit,final_loss=history[-1],elapsed_seconds=time.monotonic()-started)
        atomic(out/'TRAIN_RESULT.json',train)
        state['phase']='diagnostics';rows=[];previous=None;diag_stop=None
        order=diagnostic_order(states,observed,endpoint,strict_exit,c.get('diagnostic_priority_fractions'))
        meta['diagnostic_priority_policy']={'boundary_order':'initial, strict1percent last-valid and first-crossing,5percent last-valid and first-crossing,terminal',
            'interior_fractions':c.get('diagnostic_priority_fractions',[]),'interior_rule':'For1percent then5percent all-update initial loss-prefix endpoints, fraction times endpoint snapped to nearest already-saved step withinprefix; ties earlier. No feature values used.',
            'remaining_rule':'All saved checkpoints in ascendingstep order; none discarded',
            'evaluation_order':order}
        atomic(meta_path,meta)
        target_linear=teacher.cross(np.eye(c['d']),np.zeros(c['d']),1.,False)[0]
        target_nonlinear_variance=float(1-teacher.mean**2-target_linear@target_linear)
        for step in order:
            remaining=hard-time.monotonic()
            if remaining<=.1 or state['stop']:diag_stop=state['stop'] or 'wall_budget';break
            signal.setitimer(signal.ITIMER_REAL,remaining)
            try:
                sa,sw,sb=load_state(step);K,Q,C=bank_moments(sw,sb,teacher,0.)
                norms=np.linalg.norm(sw,axis=1);prob=ndtr(sb/norms)
                feature_mean=norms*pdf(sb/norms)+sb*prob
                fitK=K-np.outer(feature_mean,feature_mean) if profiled else K
                fitC=C-teacher.mean*feature_mean if profiled else C
                risk,previous,info=normalized_refit(fitK,fitC,sw,sb,0.,c['r'],previous=previous,gap_tol=1e-7,max_iter=4000)
                if profiled:
                    risk-=teacher.mean**2
                    info=dict(info,numerical_lower_bound=max(0.,info['numerical_lower_bound']-teacher.mean**2),convex_lower_bound=max(0.,info['convex_lower_bound']-teacher.mean**2))
                    info['objective_gap_bound']=risk-info['numerical_lower_bound']
                norms=np.linalg.norm(sw,axis=1);prob=ndtr(sb/norms)
                mean=float(sa@feature_mean/m);intercept=teacher.mean-mean if profiled else 0.;linear=(sa*prob)@sw/m
                row=dict(step=step,loss=float(history[step]),time_eta=float(np.sum(accepted[:step])*m/2),
                    parameter_path_length=float(path[step]),in_plateau=step<=endpoint,
                    refit=float(risk),refit_lower=float(info['numerical_lower_bound']),refit_gap=float(info['objective_gap_bound']),refit_info=info,
                    mean_error_sq=float((teacher.mean-mean-intercept)**2),profile_intercept=profiled,intercept=intercept,loss_baseline=baseline,loss_normalized=float(history[step]/baseline),refit_normalized=float(risk/baseline),linear_error_sq=float(np.sum((target_linear-linear)**2)),
                    teacher_mean=float(teacher.mean),target_linear_norm_sq=float(target_linear@target_linear),**stable_geometry(sa,sw,sb,teacher.U,Q=Q))
                row['target_variance']=float(1-teacher.mean**2)
                row['loss_over_target_variance']=row['loss']/row['target_variance']
                row['refit_over_target_variance']=row['refit']/row['target_variance']
                row['refit_uncertainty']=refit_uncertainty_report(info)
                row['loss_nonlinear']=row['loss']-row['mean_error_sq']-row['linear_error_sq']
                row['target_nonlinear_variance']=target_nonlinear_variance
                row['loss_nonlinear_relative']=row['loss_nonlinear']/max(target_nonlinear_variance,1e-30)
                row['A_mean_chance_baseline']=c['r']/c['d']
                initial=next((v for v in rows if v['step']==0),row)
                row['A_min_gain']=row['A_min']-initial['A_min'];row['A_mean_gain']=row['A_mean']-initial['A_mean']
                row['refit_gain_interval']=[initial['refit_lower']-row['refit'],initial['refit']-row['refit_lower']]
                row['q_reference']=min(row['A_min_gain']/.5,(row['refit_gain_interval'][0]-1e-5)/.399)
                row['q_reference_resolved']=bool(initial['agop_valid'] and row['agop_valid'])
                rows.append(row)
                tmp=out/'rows.jsonl.tmp';tmp.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in sorted(rows,key=lambda v:v['step'])));os.replace(tmp,out/'rows.jsonl')
            except BudgetExpired as exc:diag_stop=str(exc);break
            finally:signal.setitimer(signal.ITIMER_REAL,0)
        rows.sort(key=lambda v:v['step']);initial=next((v for v in rows if v['step']==0),None);end=next((v for v in rows if v['step']==endpoint),None)
        qualified=[v for v in rows if v['in_plateau'] and initial and initial['agop_valid'] and v['agop_valid'] and v['A_min_gain']>=.5 and v['refit_gain_interval'][0]>1e-5]
        prefix_required={s for s in states if s<=endpoint}
        prefix_diagnostics_complete=prefix_required.issubset({v['step'] for v in rows})
        prefix_resolved=bool(initial and initial['agop_valid'] and all(v['agop_valid'] for v in rows if v['in_plateau']))
        event_status=('observed' if qualified else 'plateau_right_censored' if censored else
                      'diagnostics_censored' if not prefix_diagnostics_complete else
                      'unresolved_grid' if not prefix_resolved else 'not_observed_resolved_grid')
        done=dict(complete=train['complete'] and len(rows)==len(order),training_complete=train['complete'],
            diagnostic_complete=len(rows)==len(order),config_sha256=config_hash,source_sha256=sources,
            teacher=c['teacher'],seed=c['seed'],m=c['m'],d=c['d'],r=c['r'],scale=c['scale'],h=c['h'],
            planned_steps=c['steps'],observed_steps=observed,training_censored=train['training_censored'],
            plateau_end=endpoint,plateau_right_censored=censored,plateau_exit_observed=not censored,
            true_plateau_endpoint_diagnosed=bool(not censored and end is not None),
            endpoint_is_censored_boundary=censored,prefix_diagnostics_complete=prefix_diagnostics_complete,
            primary_event_status=event_status,allocated_seconds=budget,deadline_utc=deadline_utc,
            diagnostic_reserve_seconds=min(diagnostic_reserve,budget*.4),diagnostic_censored=len(rows)<len(order),
            diagnostic_stop_reason=diag_stop,training_stop_reason=stop_reason,
            diagnostic_steps=[v['step'] for v in rows],required_diagnostic_steps=sorted(order),diagnostic_evaluation_order=order,diagnostic_priority_fractions=c.get('diagnostic_priority_fractions',[]),
            initial_A_min=None if initial is None else initial['A_min'],initial_refit=None if initial is None else initial['refit'],
            endpoint_observed=end is not None,endpoint_A_min=None if end is None else end['A_min'],
            endpoint_A_min_gain=None if end is None else end['A_min_gain'],endpoint_A_mean=None if end is None else end['A_mean'],
            endpoint_refit=None if end is None else end['refit'],endpoint_refit_gain=None if end is None else end['refit_gain_interval'][0]-1e-5,
            endpoint_agop_valid=None if end is None else end['agop_valid'],
            joint_observed=bool(qualified),joint_status=event_status,strict_prefix_end=train['strict_prefix_end'],strict_prefix_right_censored=train['strict_prefix_right_censored'],declared_horizon_or_loss_target_reached=train['declared_horizon_or_loss_target_reached'],force_clock=train['force_clock'],
            primary_event_rule='same saved checkpoint in ratio<=1.05 prefix: min gain>=0.5, initial/current AGOP numerical screens pass, numerical lower refit gain>1e-5',
            legacy_near_unit_plateau=plateau_from_losses(np.array(history)/baseline),
            legacy_threshold_0399_on_primary_prefix=any(v['refit_gain_interval'][0]-1e-5>=.399 for v in qualified),
            loss_baseline=baseline,profile_intercept=profiled,
            max_resolved_plateau_q=max((v['q_reference'] for v in rows if v['in_plateau'] and v['q_reference_resolved']),default=None),
            full_grid_event_negative_resolved=event_status=='not_observed_resolved_grid',
            first_joint_step=min((v['step'] for v in qualified),default=None),final_loss=history[-1],elapsed_seconds=time.monotonic()-started,
            attempt=attempt,finished_utc=utc())
        done['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name in ['meta.json','TRAIN_RESULT.json','rows.jsonl','checkpoint_manifest.json']+[n+'.npy' for n in ['loss_every_step','parameter_path_length','accepted_h','backtracking_counts','gd_cumulative_time']]}
        atomic(out/'DONE.json',done)
        with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(done)+'\n')
        return done
    except Exception as exc:
        failure=dict(error=str(exc),traceback=traceback.format_exc(),phase=state['phase'],utc=utc(),config_sha256=config_hash)
        atomic(out/'FAILED.json',failure)
        with (out/'failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--max-seconds',type=float,default=120.);p.add_argument('--deadline-utc');p.add_argument('--diagnostic-reserve',type=float,default=20.);p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(execute(json.loads(a.config.read_text()),a.out,a.max_seconds,a.deadline_utc,a.diagnostic_reserve,a.resume)),flush=True)
if __name__=='__main__':main()
