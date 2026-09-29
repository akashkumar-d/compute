"""Server-only fixed-state subset quadrature diagnostic. No training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = HERE / args.output
    if output.parent != HERE or output.exists():
        raise ValueError('Fresh direct-child output name required')
    plan_path = HERE/'PROBE_MANIFEST.json'
    plan = read(plan_path)
    for name, digest in plan['input_sha256'].items():
        if sha(HERE/name) != digest:
            raise ValueError('Changed pinned input: '+name)
    review = read(HERE/'PROBE_REVIEW.json')
    if review.get('status') != 'approved_for_execution' or review.get('manifest_sha256') != sha(plan_path):
        raise ValueError('Independent exact-manifest review required')
    import probe_resources as guard
    guard.require_server()
    resources = guard.require_capacity(dict(workers=1, reserved_cpus=31,
        min_available_memory_gib=8, min_free_disk_gib=5))
    guard.require_priority()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    for key in guard.THREAD_VARIABLES:
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    output.mkdir()
    start = time.monotonic()
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    report = dict(status='started', resources=resources, manifest_sha256=sha(plan_path),
        new_training_updates=0, scope='four-neuron subsets, not original full-width model',
        model_evaluations=0, slots=[])

    def write():
        report['elapsed_seconds'] = time.monotonic()-start
        temp = output/'RESULTS.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(output/'RESULTS.json')

    def evaluate(E, state, label, need_grad):
        if stopped[0] or time.monotonic()-start >= 170 or report['model_evaluations'] >= 33:
            raise RuntimeError('Diagnostic call/time cap reached')
        report['model_evaluations'] += 1
        with (output/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'], label=label, status='started'))+'\n')
        value = E.evaluate(*state, need_grad=need_grad)
        with (output/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'], label=label, status='returned',
                                   L=float(value['L'])))+'\n')
        if not np.isfinite(value['L']):
            raise ValueError('Nonfinite evaluated loss')
        return value

    write()
    try:
        import numpy as np
        sys.path.insert(0, str(HERE/'engine'))
        import swsmall_periodic as engine
        from student_transition import TransitionSwiGLUPop
        cfg = read(HERE/'probe_inputs/config.json')[0]['args']
        if cfg['link'] != 'h3':
            raise ValueError('Pure h3 only')
        link, breaks = engine.make_link(cfg['link'])
        T = engine.swpop.Teacher(link, np.asarray(cfg['c']), breaks=breaks, n_x=cfg['n_x'])
        with np.load(HERE/'probe_inputs/states.npz', allow_pickle=False) as arrays:
            cases=[]
            for spec in plan['states']:
                full=tuple(arrays[k][spec['extracted_index']].copy() for k in ('P','V','a'))
                # Fixed data-only rule, no optimization or outcome-dependent selection.
                ids=np.argsort(-np.linalg.norm(full[0][:-1],axis=0), kind='stable')[:4]
                theta=(full[0][:,ids].copy(),full[1][:,ids].copy(),full[2][ids].copy())
                cases.append((spec['name'],ids.tolist(),theta))
            theta=tuple(x.copy() for x in cases[-1][2])
            theta[0][:-1,0]=0.
            theta[0][:-1,2]=-theta[0][:-1,1]
            theta[0][:-1,3]=theta[0][:-1,1]
            cases.append(('late_degenerate_gate_geometry', cases[-1][1],theta))
        for name, ids, theta in cases:
            orig=[hashlib.sha256(x.tobytes()).hexdigest() for x in theta]
            previous=None
            for order in (8,12):
                slot=dict(case=name, neuron_indices=ids, order=order, limit=12., status='running')
                report['slots'].append(slot); write()
                E=TransitionSwiGLUPop(cfg['d'],T,alpha=cfg['alpha']*4/cfg['m'],
                    intercept='refit',bias=True,quad_order=order,quad_limit=12.)
                base=evaluate(E,theta,name+f'/order{order}/base',True)
                g=tuple(base[k] for k in ('gP','gV','ga'))
                D=engine.update_directions(base,cfg['m'],cfg['head_lr'])
                dn=np.sqrt(sum(np.sum(x*x) for x in D))
                if not np.isfinite(dn) or dn<=0:
                    raise ValueError('Invalid nonzero direction')
                D=tuple(x/dn for x in D)
                nominal=float(-sum(np.sum(x*y) for x,y in zip(g,D)))
                scale=float(max(1,np.sqrt(sum(np.sum(x*x) for x in theta))))
                slot.update(loss=float(base['L']), nominal_slope=nominal,
                    asymmetry_G=float(np.max(np.abs(base['G']-base['G'].T))),
                    asymmetry_p11=float(np.max(np.abs(base['Cp']['p11VV']-base['Cp']['p11VV'].T))),
                    max_dropped_gate_residual=E.max_dropped_gate_residual, differences=[])
                np.savez_compressed(output/(name+f'_order{order}.npz'),P=theta[0],V=theta[1],a=theta[2],
                    L=base['L'],G=base['G'],t=base['t'],gP=g[0],gV=g[1],ga=g[2],DP=D[0],DV=D[1],Da=D[2])
                for fraction in (1e-4,1e-5):
                    epsilon=scale*fraction
                    lo=evaluate(E,tuple(x-epsilon*y for x,y in zip(theta,D)),name+f'/o{order}/minus{fraction}',False)['L']
                    hi=evaluate(E,tuple(x+epsilon*y for x,y in zip(theta,D)),name+f'/o{order}/plus{fraction}',False)['L']
                    fd=float((lo-hi)/(2*epsilon))
                    relative=float(abs(fd-nominal)/max(abs(fd),abs(nominal),1e-30))
                    slot['differences'].append(dict(epsilon=epsilon,minus_loss=lo,plus_loss=hi,
                        fd_slope=fd,relative_mismatch=relative,passes_1e3=bool(relative<=1e-3)))
                    write()
                if previous is not None:
                    slot['refinement']=dict(loss_abs=float(abs(base['L']-previous['L'])),
                        G_max_abs=float(np.max(np.abs(base['G']-previous['G']))),
                        t_max_abs=float(np.max(np.abs(base['t']-previous['t']))),
                        gradient_relative=float(np.sqrt(sum(np.sum((base[k]-previous[k])**2) for k in ('gP','gV','ga')))/
                            max(np.sqrt(sum(np.sum(base[k]**2) for k in ('gP','gV','ga'))),1e-30)))
                    tail=TransitionSwiGLUPop(cfg['d'],T,alpha=cfg['alpha']*4/cfg['m'],
                        intercept='refit',bias=True,quad_order=12,quad_limit=10.)
                    narrow=evaluate(tail,theta,name+'/tail_limit10',True)
                    slot['tail_sensitivity']=dict(loss_abs=float(abs(narrow['L']-base['L'])),
                        gradient_max_abs=float(max(np.max(np.abs(narrow[k]-base[k])) for k in ('gP','gV','ga'))))
                previous=base
                slot['base_unchanged']=orig==[hashlib.sha256(x.tobytes()).hexdigest() for x in theta]
                slot['status']='complete';write()
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete_or_failed',error=repr(exc))
    finally:
        report['inputs_unchanged']=all(sha(HERE/n)==h for n,h in plan['input_sha256'].items())
        write()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

