"""Server-only scientific plumbing; imported only after run.py resource/review gates.

The base evaluator is unchanged. Its population identities approximate the
intended gradient, not the exact derivative of its finite-series loss. The
controller therefore verifies actual finite-evaluator decrease and audits.
"""
import json
import time
from pathlib import Path
import numpy as np
from controller import (NumericalFailure,NumericalStall,initialize,state_hash,norm,
                        directions,backtrack,Checkpoints,Events)
from comparisons import validate_ev,point_screen,compare,weights


class BudgetStop(RuntimeError):
    pass


def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray): return clean(obj.tolist())
    if isinstance(obj,np.generic): return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return str(obj)
    return obj


def write_json(path,obj):
    p=Path(path); tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n');tmp.replace(p)


def append(path,obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(clean(obj),allow_nan=False)+'\n');f.flush()


class Cached:
    def __init__(self,E,theta,ev): self.E,self.theta,self.ev=E,theta,ev; self.calls=0
    def __getattr__(self,name): return getattr(self.E,name)
    def evaluate(self,P,V,a,need_grad=True):
        if not need_grad or not all(np.array_equal(x,y) for x,y in zip((P,V,a),self.theta)):
            raise ValueError('Cached diagnostic request changed state')
        self.calls+=1
        return self.ev


class Evaluators:
    def __init__(self,cfg):
        import swsmall_periodic as engine
        from student_transition_fast import BatchedTransitionSwiGLUPop as Reference
        from hermite_pairs import make_hermite_evaluator_class
        from hermite_audit_only import make_hermite_evaluator_class as make_high
        link,breaks=engine.make_link(cfg['link'])
        self.T=engine.swpop.Teacher(link,np.asarray(cfg['c']),breaks=breaks,n_x=cfg['n_x'])
        options=dict(alpha=cfg['alpha'],intercept='refit',bias=True,quad_order=16,quad_limit=12.)
        self.base=make_hermite_evaluator_class(Reference)(cfg['d'],self.T,
            hermite_degree=128,hermite_quad_order=24,**options)
        self.high=make_high(Reference)(cfg['d'],self.T,
            hermite_degree=256,hermite_quad_order=32,**options)
        self.reference=Reference(cfg['d'],self.T,**options)


def diagnostic(E,theta,ev):
    import diagnostics
    cached=Cached(E,theta,ev)
    M=E.agop(*theta,ev=ev)
    if not np.isfinite(M).all(): raise NumericalFailure('Nonfinite cached AGOP')
    try:
        metrics=dict(status='ok',**diagnostics.metrics(cached,*theta))
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
        # Natural numerical nonresolution remains visible and unqualified.
        metrics=dict(status='failed',error=repr(exc),agop_full_rank_resolved=False,pinv={})
    if cached.calls != 1: raise RuntimeError('Canonical metrics must request exactly one cached evaluation')
    return ev,M,metrics


def arrays(theta,ev):
    out={k:v for k,v in zip(('P','V','a'),theta)}
    out.update({k:ev[k] for k in ('L','G','t','gP','gV','ga')});out['teacher_variance']=ev['V']
    out.update({'Cp_'+k:v for k,v in ev['Cp'].items()});out.update({'D_'+k:v for k,v in ev['D'].items()})
    hd=ev.get('hermite_diagnostics')
    if hd:
        for group,prefix in [('scalar_bank','bank_'),('per_pair_tail_estimates','tail_'),('geometry','geometry_')]:
            out.update({prefix+k:v for k,v in hd[group].items() if isinstance(v,np.ndarray)})
    return out


class Session:
    def __init__(self,root,out,settings,started,stopped,report):
        self.root,self.out,self.settings=root,out,settings
        self.started,self.stopped,self.report=started,stopped,report
        self.snapshots={};self.audited=set();self.deadline=settings['runtime']['smoke_admission_seconds'] if report['mode']=='smoke' else settings['runtime']['arm_admission_seconds']
        self.maxcalls=settings['runtime']['smoke_max_model_calls'] if report['mode']=='smoke' else float('inf')
    def save(self):
        self.report['elapsed_seconds']=time.monotonic()-self.started
        write_json(self.out/'RESULTS.json',self.report)
    def check(self):
        if self.stopped[0] or time.monotonic()-self.started>=self.deadline or self.report['model_evaluations']>=self.maxcalls:
            raise BudgetStop('Declared admission/deadline/call cap reached')
    def evaluate(self,E,theta,label):
        self.check();self.report['model_evaluations']+=1; call=self.report['model_evaluations']
        append(self.out/'EVALUATIONS.jsonl',dict(call=call,label=label,status='started',state_hash=state_hash(theta)))
        self.save();start=time.monotonic()
        try:
            ev=E.evaluate(*theta,need_grad=True)
            validate_ev(ev)
        except Exception as exc:
            payload={k:v for k,v in zip(('P','V','a'),theta)}
            if 'ev' in locals(): payload.update(arrays(theta,ev))
            if hasattr(exc,'diagnostics'):
                write_json(self.out/f'failure_call{call}_diagnostics.json',exc.diagnostics)
            np.savez_compressed(self.out/f'failure_call{call}.npz',**payload)
            append(self.out/'EVALUATIONS.jsonl',dict(call=call,label=label,status='failed',error=repr(exc)))
            raise
        append(self.out/'EVALUATIONS.jsonl',dict(call=call,label=label,status='returned',L=ev['L'],
            wall_seconds=time.monotonic()-start,scalar_energy_valid=True))
        self.save()
        return ev
    def snapshot(self,label,theta,record,step,t,roles):
        ev,M,metrics=record; key=(label,state_hash(theta))
        if key not in self.snapshots:
            filename=f'state_{len(self.snapshots):05d}_{label}.npz'
            payload=arrays(theta,ev);payload['AGOP']=M
            try:
                ws,vals,masks=weights(ev);payload['equilibrated_gram_eigenvalues']=vals
                payload.update({'refit_w_'+k:v for k,v in ws.items()})
                payload.update({'refit_mask_'+k:v for k,v in masks.items()})
            except np.linalg.LinAlgError: pass
            np.savez_compressed(self.out/filename,**payload)
            self.snapshots[key]=filename
        filename=self.snapshots[key]
        append(self.out/'SNAPSHOTS.jsonl',dict(file=filename,label=label,state_hash=key[1],step=step,t=t,
            roles=roles,metrics=metrics,screen=point_screen(metrics)))
        return filename
    def audit(self,engines,theta,base,cfg,step,t,roles,full_reference=False):
        self.snapshot('base',theta,base,step,t,roles)
        for kind,E in [('high',engines.high)]+([('reference',engines.reference)] if full_reference else []):
            key=(kind,state_hash(theta))
            if key in self.audited: continue
            item=dict(kind=kind,step=step,t=t,roles=roles,state_hash=key[1],status='started')
            append(self.out/'AUDITS.jsonl',item)
            ev=self.evaluate(E,theta,f'{step}/{kind}')
            other=diagnostic(E,theta,ev)
            self.snapshot(kind,theta,other,step,t,roles)
            result=compare(other,base,theta,cfg,self.settings)
            append(self.out/'AUDITS.jsonl',dict(item,status='complete',comparison=result))
            if not result['passed']: raise NumericalFailure(kind+' comparison failed')
            self.audited.add(key)


def loaded_reference(path):
    with np.load(path,allow_pickle=False) as f:
        theta=tuple(f[k].copy() for k in ('P','V','a'))
        ev={k:f[k].copy() for k in ('L','G','t','gP','gV','ga')};ev['V']=float(f['teacher_variance'])
        ev['Cp']={k[3:]:f[k].copy() for k in f.files if k.startswith('Cp_')}
        ev['D']={k[2:]:f[k].copy() for k in f.files if k.startswith('D_')}
        return theta,ev


def run_smoke(S,plan):
    report=S.report; report['scope']='Engineering smoke only; two checked updates are not a scientific trajectory'
    report['anchors']=[dict(step=k,status='unstarted') for k in S.settings['smoke']['saved_states']]
    report['arms']=[dict(id=a['id'],status='unstarted') for a in plan['arms']]
    report['pairing']=[];S.save()
    initial_states={}
    for arm in plan['arms']:
        cfg=arm['configuration'];theta,receipt=initialize(cfg);initial_states[arm['id']]=(theta,receipt)
    for item in report['arms']: item['initialization']=initial_states[item['id']][1]
    for seed in plan['development_seeds']:
        small,rs=initial_states[f'h3_scale_0.3_seed{seed}'];large,rl=initial_states[f'h3_scale_1.0_seed{seed}']
        paired=bool(rs['gaussian_hashes']==rl['gaussian_hashes'] and np.array_equal(small[2],large[2]))
        error=max(float(np.max(np.abs(s-.3*l))) for s,l in zip(small[:2],large[:2]))
        passed=paired and error<=8*np.finfo(float).eps*max(1.,norm(large[:2]))
        report['pairing'].append(dict(seed=seed,passed=passed,actual_heads_bitwise_equal=np.array_equal(small[2],large[2]),
            gaussian_hashes_equal=rs['gaussian_hashes']==rl['gaussian_hashes'],scaled_inner_max_abs=error))
        if not passed: raise NumericalFailure('Matched initialization identity failed')
    # Independent old-source head rate for old-state direction comparisons.
    oldcfg=json.loads((S.root/'smoke_saved_config.json').read_text())[0]['args'];old=Evaluators(oldcfg)
    for item in report['anchors']:
        step=item['step'];item['status']='started';S.save()
        try:
            theta,ev=loaded_reference(S.root/'runtime_inputs'/f'step{step}_reference.npz')
            validate_ev(ev);reference=diagnostic(old.reference,theta,ev)
            item['comparisons']={}
            for kind,E in [('base',old.base),('high',old.high)]:
                tested=diagnostic(E,theta,S.evaluate(E,theta,f'saved{step}/{kind}'))
                S.snapshot(kind,theta,tested,step,None,['smoke_saved_reference'])
                item['comparisons'][kind]=compare(reference,tested,theta,oldcfg,S.settings)
            item['status']='passed' if all(x['passed'] for x in item['comparisons'].values()) else 'failed'
        except BudgetStop: item['status']='capped';S.save();raise
        except Exception as exc: item.update(status='failed',error=repr(exc))
        S.save()
    for arm,item in zip(plan['arms'],report['arms']):
        cfg=arm['configuration'];theta,receipt=initial_states[arm['id']]
        item.update(status='started',initialization=receipt,finite_differences=[],checked_updates=[]);S.save()
        try:
            E=Evaluators(cfg);ev=S.evaluate(E.base,theta,arm['id']+'/initial')
            base=diagnostic(E.base,theta,ev);S.snapshot('base',theta,base,0,0.,[arm['id'],'smoke_initial'])
            S.audit(E,theta,base,cfg,0,0.,['smoke_initial'],full_reference=False)
            D=directions(ev,cfg);dn=norm(D)
            if dn==0: raise NumericalStall('Zero smoke direction')
            unit=tuple(x/dn for x in D)
            predicted=-sum(float(np.sum(ev[k]*u)) for k,u in zip(('gP','gV','ga'),unit))
            for rel in S.settings['smoke']['initial_directional_epsilon_relative']:
                epsilon=rel*max(1.,norm(theta))
                plus=tuple(x+epsilon*u for x,u in zip(theta,unit));minus=tuple(x-epsilon*u for x,u in zip(theta,unit))
                lp=S.evaluate(E.base,plus,arm['id']+'/fd_plus')['L'];lm=S.evaluate(E.base,minus,arm['id']+'/fd_minus')['L']
                observed=-(lp-lm)/(2*epsilon)
                error=abs(observed-predicted);allowed=S.settings['smoke']['slope_relative_tolerance']*abs(predicted)+1e-10*E.T.V/max(1.,norm(theta))
                item['finite_differences'].append(dict(epsilon=epsilon,observed=observed,predicted=predicted,absolute_error=error,passed=bool(error<=allowed)))
                if error>allowed: raise NumericalFailure('Smoke directional finite difference failed')
            t=0.
            for step in range(1,S.settings['smoke']['checked_updates_per_initial']+1):
                def log(row): append(S.out/'TRIALS.jsonl',dict(arm=arm['id'],from_step=step-1,**row))
                theta,ev,dt,rel=backtrack(theta,ev,cfg,t,S.settings['controller'],
                    lambda x,j:S.evaluate(E.base,x,f'{arm["id"]}/smoke{step}/trial{j}'),log)
                t+=dt;base=diagnostic(E.base,theta,ev)
                S.audit(E,theta,base,cfg,step,t,[arm['id'],'smoke_update'],full_reference=False)
                item['checked_updates'].append(dict(step=step,t=t,dt=dt,L=float(ev['L']),state_hash=state_hash(theta)))
            item['status']='passed'
        except BudgetStop: item['status']='capped';S.save();raise
        except Exception as exc: item.update(status='failed',error=repr(exc))
        S.save()
    report['all_checks_passed']=all(x['status']=='passed' for x in report['anchors']+report['arms']) and all(x['passed'] for x in report['pairing'])
    report['status']='complete' if report['all_checks_passed'] else 'failed'


def run_arm(S,arm,criteria):
    cfg=arm['configuration'];report=S.report;E=Evaluators(cfg)
    theta,receipt=initialize(cfg);report.update(initialization=receipt,arm=arm,
        training_method='Fixed Hermite128/q24 approximate population gradient with safeguarded Euler; sampled numerical checks')
    events=Events(criteria);checkpoints=Checkpoints(cfg);step=0;t=0.;previous=None;current=None
    failure=False;S.save()
    try:
        ev=S.evaluate(E.base,theta,'initial');current=diagnostic(E.base,theta,ev)
        # Actual full reference gate BEFORE any update. All work counts in cap.
        S.audit(E,theta,current,cfg,step,t,['initial'],full_reference=True)
        while True:
            ev,M,metric=current;L=float(ev['L']/E.T.V)
            roles=events.observe(step,t,L,metric,point_screen(metric))
            record=dict(step=step,t=t,L=L,loss_raw=float(ev['L']),teacher_variance=float(E.T.V),
                state_hash=state_hash(theta),metrics=metric,screen=point_screen(metric),roles=roles,
                event_claim_status='unverified_until_audits_and_independent_saved_output_review')
            append(S.out/'ACCEPTED.jsonl',record)
            natural=checkpoints.natural(t,L)
            if natural: S.snapshot('base',theta,current,step,t,['natural'])
            if any(x.startswith('prefix_crossing:') for x in roles) and previous is not None:
                pt,pr,ps,pclock=previous
                S.audit(E,pt,pr,cfg,ps,pclock,['prefix_previous'])
            if roles or step%S.settings['audits']['accepted_step_cadence']==0:
                full=any(x.startswith(('first_candidate:','first_release:')) for x in roles)
                S.audit(E,theta,current,cfg,step,t,roles or ['cadence'],full_reference=full)
            report.update(accepted_updates=step,physical_time=t,normalized_loss=L,events=events.summary());S.save()
            if t>=cfg['t_max']: report['stop_reason']='physical_horizon';break
            if L<=0: raise NumericalStall('Nonpositive loss within roundoff tolerance is not an ordinary loss-stop completion')
            if L<=cfg['L_stop']: report['stop_reason']='loss_stop';break
            if step>=cfg['max_steps']: report['stop_reason']='step_cap';break
            previous=(theta,current,step,t)
            def log(row): append(S.out/'TRIALS.jsonl',dict(from_step=step,from_t=t,**row))
            newtheta,newev,dt,rel=backtrack(theta,ev,cfg,t,S.settings['controller'],
                lambda x,j:S.evaluate(E.base,x,f'{step+1}/trial{j}'),log)
            try:
                newrecord=diagnostic(E.base,newtheta,newev)
            except Exception:
                np.savez_compressed(S.out/f'failure_uncommitted_step{step+1}.npz',**arrays(newtheta,newev))
                append(S.out/'TRIALS.jsonl',dict(from_step=step,from_t=t,
                    status='accepted_trial_not_committed_diagnostic_failure',state_hash=state_hash(newtheta)))
                raise
            theta=newtheta;step+=1;t+=dt;current=newrecord
    except BudgetStop as exc: report.update(stop_reason='wall_censored',stop_detail=str(exc))
    except NumericalStall as exc: report.update(stop_reason='numerical_stall',stop_detail=str(exc))
    except Exception as exc:
        failure=True;report.update(stop_reason='numerical_or_runtime_failure',error=repr(exc))
    finally:
        report.update(accepted_updates=step,physical_time=t,events=events.summary(),
            events_numerical_status='unverified_until_independent_saved_output_review; failed/capped audits invalidate affected event claims',
            status='failed' if failure else 'censored' if report.get('stop_reason') in ('wall_censored','numerical_stall') else 'complete',
            final_numerical_audit='not_completed')
        if current is not None:
            S.snapshot('base',theta,current,step,t,['final'])
            if not failure:
                S.deadline=S.settings['runtime']['arm_final_diagnostic_deadline_seconds']
                try:
                    S.audit(E,theta,current,cfg,step,t,['final'],full_reference=True)
                    report['final_numerical_audit']='passed'
                except BudgetStop as exc: report.update(final_numerical_audit='censored',final_audit_error=str(exc),status='censored')
                except Exception as exc: report.update(final_numerical_audit='failed',final_audit_error=repr(exc),status='failed')
        S.save()
