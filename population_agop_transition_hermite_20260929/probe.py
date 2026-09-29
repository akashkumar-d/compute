"""Server-only fixed-state Hermite evaluator comparison; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=HERE/args.output
    if out.parent!=HERE or out.exists():raise ValueError('Fresh direct-child output required')
    pp=HERE/'PROBE_MANIFEST.json';plan=read(pp)
    for n,h in plan['input_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('Changed pinned input: '+n)
    review=read(HERE/'PROBE_REVIEW.json')
    if review.get('status')!='approved_for_execution' or review.get('manifest_sha256')!=sha(pp):
        raise ValueError('Exact source review required')
    import probe_resources as guard
    guard.require_server()
    resources=guard.require_capacity(dict(workers=1,reserved_cpus=31,
        min_available_memory_gib=8,min_free_disk_gib=5))
    guard.require_priority()
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    for key in guard.THREAD_VARIABLES:os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    out.mkdir();start=time.monotonic();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    report=dict(status='started',resources=resources,manifest_sha256=sha(pp),model_evaluations=0,
        new_training_updates=0,slots=[],finite_differences=[],
        scope='Two fixed r8d64m64 states, against stored order16 reference; no AGOP/refit or trajectory validation')

    def write():
        report['elapsed_seconds']=time.monotonic()-start
        p=out/'RESULTS.json.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(out/'RESULTS.json')

    def evaluate(E,theta,label,grad):
        if stopped[0] or time.monotonic()-start>=170 or report['model_evaluations']>=14:
            raise RuntimeError('Diagnostic cap reached')
        report['model_evaluations']+=1;t0=time.monotonic()
        with (out/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'],label=label,status='started'))+'\n')
        ev=E.evaluate(*theta,need_grad=grad)
        elapsed=time.monotonic()-t0
        with (out/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'],label=label,status='returned',L=float(ev['L']),wall_seconds=elapsed))+'\n')
        if not np.isfinite(ev['L']):raise ValueError('Nonfinite loss')
        return ev,elapsed

    def norm(xs):return float(np.sqrt(sum(np.sum(x*x) for x in xs)))

    write()
    try:
        import numpy as np
        sys.path.insert(0,str(HERE/'engine'))
        import swsmall_periodic as engine
        from student_transition_fast import BatchedTransitionSwiGLUPop
        from hermite_pairs import make_hermite_evaluator_class
        H=make_hermite_evaluator_class(BatchedTransitionSwiGLUPop)
        cfg=read(HERE/'probe_inputs/config.json')[0]['args']
        if cfg['link']!='h3' or cfg['m']!=64:raise ValueError('Original pure h3 full-width configuration required')
        link,breaks=engine.make_link(cfg['link'])
        T=engine.swpop.Teacher(link,np.asarray(cfg['c']),breaks=breaks,n_x=cfg['n_x'])
        states=[];chosen=[]
        for name,index in [('last_valid_5pct',0),('old_bad_step',2)]:
            with np.load(HERE/'reference_saved'/(name+'_order16.npz'),allow_pickle=False) as f:
                ref={k:f[k].copy() for k in f.files}
            theta=tuple(ref[k] for k in ('P','V','a'))
            with np.load(HERE/'probe_inputs/states.npz',allow_pickle=False) as f:
                if not all(np.array_equal(ref[k],f[k][index]) for k in ('P','V','a')):
                    raise ValueError('Original full-width state mismatch')
            Dr=engine.update_directions(ref,cfg['m'],cfg['head_lr']);dn=norm(Dr)
            unit=tuple(x/dn for x in Dr)
            if not all(np.array_equal(x,ref[k]) for x,k in zip(unit,('DP','DV','Da'))):
                raise ValueError('Reference direction mismatch')
            states.append((name,theta,ref,Dr,unit))
            for degree,order in [(64,16),(128,16),(128,24)]:
                E=H(cfg['d'],T,alpha=cfg['alpha'],intercept='refit',bias=True,
                    quad_order=16,quad_limit=12.,hermite_degree=degree,hermite_quad_order=order)
                label=name+f'/N{degree}_q{order}'
                ev,elapsed=evaluate(E,theta,label+'/base',True)
                diag=ev['hermite_diagnostics'];bank=diag['scalar_bank']
                D=engine.update_directions(ev,cfg['m'],cfg['head_lr'])
                gref=tuple(ref[k] for k in ('gP','gV','ga'));g=tuple(ev[k] for k in ('gP','gV','ga'))
                slope_ref=float(-sum(np.sum(x*y) for x,y in zip(gref,unit)))
                slope=float(-sum(np.sum(x*y) for x,y in zip(g,unit)))
                scale=max(1,norm(theta));loss_error=float(abs(ev['L']-ref['L']))
                derror=norm(tuple(x-y for x,y in zip(D,Dr)))
                slope_error=abs(slope-slope_ref)
                scalar_valid=bool(diag['all_scalar_estimates_valid'])
                comparisons=dict(loss_abs=loss_error,G_max_abs=float(np.max(np.abs(ev['G']-ref['G']))),
                    t_max_abs=float(np.max(np.abs(ev['t']-ref['t']))),
                    gradient_relative=norm(tuple(x-y for x,y in zip(g,gref)))/max(norm(gref),1e-30),
                    weighted_direction_relative=derror/max(dn,1e-30),common_slope_abs=slope_error,
                    common_slope_relative=slope_error/max(abs(slope_ref),1e-30))
                passes=dict(loss=bool(loss_error<=1e-7*T.V),
                    weighted_direction=bool(derror<=1e-4*dn+1e-10*cfg['m']*T.V/scale),
                    common_slope=bool(slope_error<=1e-4*abs(slope_ref)+1e-10*T.V/scale),
                    scalar_energy=scalar_valid)
                slot=dict(case=name,degree=degree,scalar_order=order,loss=float(ev['L']),wall_seconds=elapsed,
                    comparisons=comparisons,passes=passes,passes_base_screen=bool(all(passes.values())),
                    material_negative_scalar_energies=int(np.count_nonzero(bank['material_negative_energy'])),
                    negative_scalar_energies=int(np.count_nonzero(bank['negative_energy'])),
                    raw_residual_min=float(np.min(bank['signed_residual_energy'])),
                    correlation_clips=int(np.count_nonzero(diag['geometry']['correlation_clipped'])),
                    scalar_banks=int(diag['marginal_banks']),pair_rows_returned=diag['pair_rows_returned'],
                    gradient_semantics=ev['gradient_semantics'])
                report['slots'].append(slot)
                arrays={k:ev[k] for k in ('L','G','t','gP','gV','ga')}
                arrays.update({k:ref[k] for k in ('P','V','a')})
                arrays.update({'Cp_'+k:v for k,v in ev['Cp'].items()})
                arrays.update({'tail_'+k:v for k,v in diag['per_pair_tail_estimates'].items()})
                arrays.update({'bank_'+k:v for k,v in bank.items() if isinstance(v,np.ndarray)})
                np.savez_compressed(out/(name+f'_N{degree}_q{order}.npz'),**arrays)
                if degree==128 and order==24:chosen.append((E,ev,slot))
                write()
        report['fixed_final_recipe']='N128/scalar_order24; alternatives are convergence checks, not outcome-based selection'
        report['final_recipe_base_passes_both']=bool(all(x[2]['passes_base_screen'] for x in chosen))
        if report['final_recipe_base_passes_both']:
            for (name,theta,ref,Dr,unit),(E,base,slot) in zip(states,chosen):
                nominal=float(-sum(np.sum(base[k]*u) for k,u in zip(('gP','gV','ga'),unit)))
                for fraction in (1e-4,1e-5):
                    epsilon=max(1,norm(theta))*fraction
                    lo,_=evaluate(E,tuple(x-epsilon*u for x,u in zip(theta,unit)),name+f'/minus{fraction}',False)
                    hi,_=evaluate(E,tuple(x+epsilon*u for x,u in zip(theta,unit)),name+f'/plus{fraction}',False)
                    fd=float((lo['L']-hi['L'])/(2*epsilon))
                    relative=abs(fd-nominal)/max(abs(fd),abs(nominal),1e-30)
                    scalar_valid=bool(lo['hermite_diagnostics']['all_scalar_estimates_valid'] and hi['hermite_diagnostics']['all_scalar_estimates_valid'])
                    report['finite_differences'].append(dict(case=name,epsilon=epsilon,minus_loss=float(lo['L']),
                        plus_loss=float(hi['L']),nominal_slope=nominal,fd_slope=fd,relative_mismatch=relative,
                        scalar_energy_valid=scalar_valid,passes_1e3=bool(relative<=1e-3 and scalar_valid)))
                    write()
        else:report['fd_not_run_reason']='Fixed final recipe failed at least one base comparison; no training or parameter changes'
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete_or_failed',error=repr(exc))
    finally:
        report['inputs_unchanged']=all(sha(HERE/n)==h for n,h in plan['input_sha256'].items());write()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
