"""SERVER-only rank/width/dimension scaling with explicit wall-time censoring."""
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
from refit import normalized_refit
from frames import bank_moments,plateau_from_losses
ROOT=Path(__file__).resolve().parents[1]
KERNELS=['population.py','teachers_extra.py','teachers_product.py','refit.py','frames.py']


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
def source_hashes():return {name:sha(ROOT/'code'/name) for name in KERNELS+['scaling_run.py']}
def checkpoints(end):return sorted(set(range(0,min(end,1000)+1,10))|set(range(1100,end+1,100))|{end})
def require_server():
    if sys.platform=='darwin' or os.environ.get('AGOP_EXECUTION_SITE')!='SERVER':
        raise RuntimeError('SERVER execution required; set AGOP_EXECUTION_SITE=SERVER on authorized Linux server')
def config_check(config):
    c=dict(config)
    for k in ['id','teacher','seed','m','d','r','scale','h','steps']:
        if k not in c:raise ValueError('Missing config field '+k)
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c['id']):raise ValueError('Unsafe id')
    if c['teacher'] not in ['h3','damped_2_3','three_atom','h3_plus_h5','h3_plus_sine','h2','h4','sine','relu','abs']:raise ValueError('Teacher is outside the declared additive breadth pilot')
    if c.get('student','relu')!='relu' or c.get('optimizer','gd')!='gd':raise ValueError('ReLU simultaneous GD only')
    for k in ['seed','m','d','r','steps']:
        if isinstance(c[k],bool) or not isinstance(c[k],int) or c[k]<1:raise ValueError('Invalid '+k)
    if c['r'] not in [2,4,8,16] or c['d']<2*c['r'] or c['m']<c['r']:raise ValueError('Require r in2,4,8,16; d>=2r; m>=r')
    if not all(np.isfinite(c[k]) and c[k]>0 for k in ['scale','h']):raise ValueError('Positive finite scale,h required')
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
            meta=dict(config=c,config_sha256=config_hash,source_sha256=sources,execution_site='SERVER',
                hostname=socket.gethostname(),started_utc=utc(),student='relu',optimizer='gd',
                teacher=c['teacher'],seed=c['seed'],m=c['m'],d=c['d'],r=c['r'],scale=c['scale'],h=c['h'],steps=c['steps'],
                initialization='All raw A,W,b IID N(0,scale^2/d); all blocks jointly trained',
                plateau_rule='All-update initial prefix in [.95,1.05] with running max/min<=1.05',
                refit_caps={'l2':32.,'l1':float(64*np.sqrt(c['r']))},refit_gap_tolerance=1e-7,refit_max_iterations=4000,
                armijo_c=1e-4,armijo_factor=.5,armijo_roundoff_tolerance=1e-13,max_backtracks=40,
                python_version=sys.version.split()[0],numpy_version=np.__version__)
            atomic(meta_path,meta)
        teacher=Teacher(c['teacher'],c['r'],c['d']);m=c['m']
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
            prior_end,prior_censored=plateau_from_losses(np.asarray(history))
            first_exit=None if prior_censored else prior_end+1
        lo=min(history,default=np.inf);hi=max(history,default=-np.inf);cached=None;stop_reason=None
        def snapshot(n):
            arrays={}
            if prev is not None:arrays.update({'prev_'+k:v for k,v in zip(['A','W','b'],prev)})
            npz(out/'resume.npz',A=A,W=W,b=b,step=n,config_sha256=config_hash,history=np.array(history),path=np.array(path),
                accepted=np.array(accepted),backtracks=np.array(backtracks),trial_h=trial_h,displacement=displacement,
                saved_steps=np.array(sorted(states)),**arrays)
        required=set(checkpoints(c['steps']))
        for n in range(start,c['steps']+1):
            if cached is None:cached=loss_force(A,W,b,teacher,0.)
            L,forces,*_=cached
            if not np.isfinite(L) or any(not np.isfinite(z).all() for z in [A,W,b]):raise FloatingPointError('Nonfinite training state')
            history.append(float(L));path.append(displacement);lo=min(lo,L);hi=max(hi,L)
            crossed=first_exit is None and (not .95<=L<=1.05 or hi>1.05*lo)
            if crossed:
                first_exit=n
                if n:save_state(n-1,prev)
            stop_reason=state['stop'] or ('training_budget' if time.monotonic()>=train_until and n<c['steps'] else None)
            if n in required or crossed or stop_reason:save_state(n,(A,W,b))
            if n%100==0 or stop_reason or n==c['steps']:snapshot(n)
            if n%500==0:print(json.dumps(dict(phase='training',step=n,loss=float(L),elapsed=time.monotonic()-started)),flush=True)
            if n==c['steps'] or stop_reason:break
            prev=(A.copy(),W.copy(),b.copy());norm2=sum(float(np.sum(z*z)) for z in forces)
            for bt in range(41):
                updates=[trial_h*z for z in forces]
                candidate=loss_force(A+updates[0],W+updates[1],b+updates[2],teacher,0.)
                allowance=1e-13*max(1.,abs(float(L)))
                if np.isfinite(candidate[0]) and candidate[0]<=L-1e-4*(2*trial_h/m)*norm2+allowance:
                    cached=candidate;accepted.append(trial_h);backtracks.append(bt);break
                trial_h*=.5
            else:raise FloatingPointError('Armijo exhausted40backtracks')
            displacement+=np.sqrt(sum(float(np.sum(z*z)) for z in updates))
            A+=updates[0];W+=updates[1];b+=updates[2]
        observed=n;endpoint,censored=plateau_from_losses(np.array(history));save_state(n,(A,W,b))
        atomic(out/'checkpoint_manifest.json',{str(step):dict(path=str(file.relative_to(out)),sha256=sha(file)) for step,file in sorted(states.items())})
        for name,value in [('loss_every_step',history),('parameter_path_length',path),('accepted_h',accepted),('backtracking_counts',backtracks),('gd_cumulative_time',np.r_[0.,np.cumsum(accepted)*m/2])]:np.save(out/(name+'.npy'),np.asarray(value))
        train=dict(complete=observed==c['steps'],observed_steps=observed,planned_steps=c['steps'],
            training_censored=observed<c['steps'],stop_reason=stop_reason,plateau_end=endpoint,
            plateau_right_censored=censored,first_exit_step=first_exit,final_loss=history[-1],elapsed_seconds=time.monotonic()-started)
        atomic(out/'TRAIN_RESULT.json',train)
        state['phase']='diagnostics';rows=[];previous=None;diag_stop=None
        priority=[0,endpoint,endpoint+1,observed]+sorted(states)
        order=list(dict.fromkeys(s for s in priority if s in states))
        target_linear=teacher.cross(np.eye(c['d']),np.zeros(c['d']),1.,False)[0]
        target_nonlinear_variance=float(1-teacher.mean**2-target_linear@target_linear)
        for step in order:
            remaining=hard-time.monotonic()
            if remaining<=.1 or state['stop']:diag_stop=state['stop'] or 'wall_budget';break
            signal.setitimer(signal.ITIMER_REAL,remaining)
            try:
                sa,sw,sb=load_state(step);K,Q,C=bank_moments(sw,sb,teacher,0.)
                risk,previous,info=normalized_refit(K,C,sw,sb,0.,c['r'],previous=previous,gap_tol=1e-7,max_iter=4000)
                norms=np.linalg.norm(sw,axis=1);prob=ndtr(sb/norms)
                mean=float(sa@(norms*pdf(sb/norms)+sb*prob)/m);linear=(sa*prob)@sw/m
                row=dict(step=step,loss=float(history[step]),time_eta=float(np.sum(accepted[:step])*m/2),
                    parameter_path_length=float(path[step]),in_plateau=step<=endpoint,
                    refit=float(risk),refit_lower=float(info['numerical_lower_bound']),refit_gap=float(info['objective_gap_bound']),refit_info=info,
                    mean_error_sq=float((teacher.mean-mean)**2),linear_error_sq=float(np.sum((target_linear-linear)**2)),
                    teacher_mean=float(teacher.mean),target_linear_norm_sq=float(target_linear@target_linear),**geometry(sa,sw,Q,teacher.U))
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
        qualified=[v for v in rows if v['in_plateau'] and initial and initial['agop_valid'] and v['agop_valid'] and v['A_min_gain']>=.5 and v['refit_gain_interval'][0]-1e-5>=.399]
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
            diagnostic_steps=[v['step'] for v in rows],required_diagnostic_steps=sorted(order),
            initial_A_min=None if initial is None else initial['A_min'],initial_refit=None if initial is None else initial['refit'],
            endpoint_observed=end is not None,endpoint_A_min=None if end is None else end['A_min'],
            endpoint_A_min_gain=None if end is None else end['A_min_gain'],endpoint_A_mean=None if end is None else end['A_mean'],
            endpoint_refit=None if end is None else end['refit'],endpoint_refit_gain=None if end is None else end['refit_gain_interval'][0]-1e-5,
            endpoint_agop_valid=None if end is None else end['agop_valid'],
            joint_observed=bool(qualified),joint_status=event_status,
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
