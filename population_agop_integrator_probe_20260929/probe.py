"""SERVER-only frozen-state probe. No trajectory, initialization generation or training."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


class Capped(Exception):pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();output=args.output.resolve()
    if output.parent!=HERE or output.exists():raise ValueError('Use a fresh direct-child output directory')
    plan_path=HERE/'PROBE_MANIFEST.json';plan=read(plan_path);plan_hash=sha(plan_path)
    for name,digest in plan['input_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Pinned probe input changed: '+name)
    review=read(HERE/'PROBE_REVIEW.json')
    if review.get('status')!='approved_for_execution' or review.get('manifest_sha256')!=plan_hash:
        raise RuntimeError('Independent exact-manifest probe review required')
    # These imported functions are the unchanged stdlib-only scheduler guards.
    guard=module(HERE/'probe_resources.py','reviewed_probe_resource_guards')
    guard.require_server()
    resources=guard.require_capacity(dict(workers=1,reserved_cpus=31,min_available_memory_gib=8,min_free_disk_gib=5))
    guard.require_priority()
    allocated=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allocated[-1]})
    for key in guard.THREAD_VARIABLES:os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic();deadline=start+170.;stopped=[False];evaluations=[0]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    output.mkdir()
    report=dict(status='started',manifest_sha256=plan_hash,resources=resources,
                active_affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0),model_evaluations=0,
                new_training_updates=0,input_sha256=plan['input_sha256'],
                slots=[dict(state=s['name'],order_multiplier=k,status='not_attempted') for k in (1,2) for s in plan['states']])

    def write():
        report['model_evaluations']=evaluations[0];report['elapsed_seconds']=time.monotonic()-start
        tmp=output/'RESULTS.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output/'RESULTS.json')

    def check():
        if stopped[0] or time.monotonic()>=deadline or evaluations[0]>=64:raise Capped('signal, 170-second soft deadline, or 64-evaluation cap')

    write()
    try:
        # No NumPy or scientific module is imported until all pins and guards pass.
        import numpy as np
        sys.path.insert(0,str(HERE/'fork/swiglu/code/engine'))
        import swsmall_armijo as engine
        from armijo_step import Settings, take_step, raw_metric_slope
        cfg=read(HERE/'probe_inputs/config.json')[0]['args']
        arrays=np.load(HERE/'probe_inputs/states.npz',allow_pickle=False)

        def finite_value(value):
            if isinstance(value,dict):return all(finite_value(v) for v in value.values())
            return bool(np.isfinite(value).all())

        def eval_compare(left,right):
            if isinstance(left,dict):return all(eval_compare(left[k],right[k]) for k in left) and set(left)==set(right)
            return bool(np.allclose(left,right,rtol=1e-10,atol=1e-12,equal_nan=False))

        for multiplier in (1,2):
            for spec in plan['states']:
                slot=next(s for s in report['slots'] if s['state']==spec['name'] and s['order_multiplier']==multiplier)
                try:
                    check();slot['status']='running';write()
                    order={key:cfg[key]*multiplier for key in ('n_pair','n_diag','n_z','n_x')}
                    link,breaks=engine.make_link(cfg['link'])
                    teacher=engine.swpop.Teacher(link,np.asarray(cfg['c'],float),breaks=breaks,n_x=order['n_x'])
                    E=engine.swpop.SwiGLUPop(cfg['d'],teacher,alpha=cfg['alpha'],intercept='refit',bias=True,
                        n_pair=order['n_pair'],n_diag=order['n_diag'],n_z=order['n_z'])
                    index=spec['extracted_index'];theta=tuple(arrays[key][index].copy() for key in ('P','V','a'))
                    original_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in theta]

                    def evaluate(state,label,need_grad=True):
                        check();evaluations[0]+=1
                        value=E.evaluate(*state,need_grad=need_grad)
                        event=dict(index=evaluations[0],state=spec['name'],order_multiplier=multiplier,
                                   label=label,loss_raw=float(value['L']) if np.isfinite(value['L']) else None,
                                   normalized_loss=float(value['L']/teacher.V) if np.isfinite(value['L']/teacher.V) else None,
                                   need_grad=need_grad,elapsed_seconds=time.monotonic()-start)
                        with (output/'EVALUATIONS.jsonl').open('a') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')
                        check();return value

                    base=evaluate(theta,'base');D=engine.update_directions(base,cfg['m'],cfg['head_lr'])
                    slope=raw_metric_slope([float(np.sum(base[k]**2)) for k in ('gP','gV','ga')],cfg['m'],cfg['head_lr'],teacher.V)
                    dt=spec['old_proposed_dt'];F0=float(base['L']/teacher.V)
                    gn=np.sqrt((D[0]**2).sum(0)+(D[1]**2).sum(0)+D[2]**2)
                    norms=np.sqrt((theta[0]**2).sum(0)+(theta[1]**2).sum(0)+theta[2]**2)
                    rel=float((gn/np.maximum(norms,1e-300)).max())
                    if not finite_value(base) or not np.isfinite(slope) or not np.isfinite(F0):
                        raise ValueError('Nonfinite base evaluation or nominal slope')
                    slot.update(orders=order,base_loss=F0,target_variance=float(teacher.V),nominal_slope=slope,
                                old_proposed_dt=dt,recomputed_relative_cap_dt=min(cfg['dt_max'],cfg['h']/max(rel,1e-300)),
                                directional_differences=[],old_proposal={'status':'not_evaluated'},armijo={'status':'not_evaluated'},cache_check={'status':'not_evaluated'})
                    for fraction in (1e-3,1e-4):
                        eps=dt*fraction
                        lower=evaluate(tuple(x-eps*g for x,g in zip(theta,D)),f'fd_minus_{fraction}',False)['L']/teacher.V
                        upper=evaluate(tuple(x+eps*g for x,g in zip(theta,D)),f'fd_plus_{fraction}',False)['L']/teacher.V
                        fd=float((lower-upper)/(2*eps));floor=128*np.finfo(float).eps*max(1.,abs(F0))/eps
                        if not np.isfinite([lower,upper,fd,floor]).all():raise ValueError('Nonfinite directional difference')
                        slot['directional_differences'].append(dict(epsilon=eps,minus_loss=float(lower),plus_loss=float(upper),
                            derivative=fd,nominal_slope=slope,absolute_difference=abs(fd-slope),
                            relative_difference=abs(fd-slope)/max(abs(fd),abs(slope),1e-300),
                            roundoff_scale=floor,negative=fd<0,agreement=abs(fd-slope)<=1e-3*max(abs(fd),abs(slope))+floor))
                        write()
                    differences=slot['directional_differences'];f0,f1=[d['derivative'] for d in differences]
                    floor=max(d['roundoff_scale'] for d in differences)
                    slot['nominal_slope_check']=dict(all_negative=all(d['negative'] for d in differences),
                        both_agree=all(d['agreement'] for d in differences),
                        finite_difference_absolute_change=abs(f0-f1),
                        finite_difference_stable=abs(f0-f1)<=1e-3*max(abs(f0),abs(f1))+floor,
                        roundoff_dominated=any(abs(d['derivative'])<=d['roundoff_scale'] for d in differences))
                    old_state=tuple(x-dt*g for x,g in zip(theta,D));old_ev=evaluate(old_state,'old_proposal')
                    slot['old_proposal']=dict(status='evaluated',loss=float(old_ev['L']/teacher.V) if np.isfinite(old_ev['L']/teacher.V) else None,strict_descent=bool(old_ev['L']<base['L']))
                    if multiplier==1:
                        saved=tuple(arrays[key][spec['next_extracted_index']] for key in ('P','V','a'))
                        slot['old_proposal'].update(saved_next_step=spec['next_step'],saved_next_loss=spec['next_saved_loss'],
                            arrays_bitwise_equal=all(np.array_equal(x,y) for x,y in zip(old_state,saved)),
                            arrays_close=all(np.allclose(x,y,rtol=1e-10,atol=1e-12) for x,y in zip(old_state,saved)),
                            array_max_abs=[float(np.max(np.abs(x-y))) for x,y in zip(old_state,saved)],
                            loss_close=bool(np.isclose(old_ev['L']/teacher.V,spec['next_saved_loss'],rtol=1e-10,atol=1e-12)))
                    first=[True];trials=[]
                    def trial_evaluate(state):
                        if first[0]:
                            first[0]=False
                            if all(np.array_equal(x,y) for x,y in zip(state,old_state)):return old_ev
                        return evaluate(state,'backtrack_trial')
                    result=take_step(theta,base,dt=dt,time=spec['t'],slope=slope,
                        make_trial=lambda state,amount:tuple(x-amount*g for x,g in zip(state,D)),evaluate=trial_evaluate,
                        loss=lambda ev:ev['L']/teacher.V,valid_evaluation=finite_value,
                        valid_state=lambda state:all(np.isfinite(x).all() for x in state),
                        same_state=lambda x,y:all(np.array_equal(a,b) for a,b in zip(x,y)),
                        stop=lambda:stopped[0] or time.monotonic()>=deadline or evaluations[0]>=64,
                        record=lambda event:(trials.append(event),slot.update(trials=trials),write()),settings=Settings())
                    slot['armijo']=dict(status=result.reason,accepted=result.accepted,accepted_dt=result.accepted_dt,
                                        accepted_loss=float(result.evaluation['L']/teacher.V) if result.accepted else None)
                    write();check()
                    if result.accepted:
                        fresh=evaluate(result.state,'fresh_accepted_cache_check')
                        slot['cache_check']=dict(status='checked',all_fields_close=eval_compare(result.evaluation,fresh),
                            loss_absolute_difference=abs(float(result.evaluation['L']-fresh['L'])))
                    else:slot['cache_check']={'status':'not_applicable_no_accepted_trial'}
                    slot['base_arrays_unchanged']=original_hashes==[hashlib.sha256(x.tobytes()).hexdigest() for x in theta]
                    slot['status']='complete';write()
                except Capped as exc:
                    slot.update(status='capped',error=str(exc));write()
                except Exception as exc:
                    slot.update(status='failed',error=repr(exc));write()
        arrays.close()
        report['status']='complete' if all(s['status']=='complete' for s in report['slots']) else 'incomplete_or_failed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
    finally:
        for slot in report['slots']:
            if slot['status']=='not_attempted':slot['status']='not_attempted_capped_or_failed'
        report['inputs_unchanged']=all(sha(HERE/n)==h for n,h in plan['input_sha256'].items())
        write()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
