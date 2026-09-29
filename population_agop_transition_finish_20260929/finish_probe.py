"""Finish exactly two censored fixed-state checks; never run training."""
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
    out = HERE / args.output
    if out.parent != HERE or out.exists():
        raise ValueError('Fresh direct-child output required')
    plan_path = HERE / 'PROBE_MANIFEST.json'
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
    out.mkdir()
    start = time.monotonic()
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    report = dict(status='started', resources=resources,
                  manifest_sha256=sha(plan_path), new_training_updates=0,
                  model_evaluations=0, scope='Two missing checks from capped fast a01; original record retained')

    def write():
        report['elapsed_seconds'] = time.monotonic()-start
        temp = out/'RESULTS.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(out/'RESULTS.json')

    def evaluate(E, theta, label, need_grad):
        if stopped[0] or time.monotonic()-start >= 170 or report['model_evaluations'] >= 2:
            raise RuntimeError('Diagnostic cap reached')
        report['model_evaluations'] += 1
        t0 = time.monotonic()
        with (out/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'], label=label, status='started'))+'\n')
        result = E.evaluate(*theta, need_grad=need_grad)
        with (out/'EVALUATIONS.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=report['model_evaluations'], label=label, status='returned',
                                   L=float(result['L']), wall_seconds=time.monotonic()-t0))+'\n')
        if not np.isfinite(result['L']):
            raise ValueError('Nonfinite loss')
        return result

    write()
    try:
        import numpy as np
        sys.path.insert(0, str(HERE/'engine'))
        import swsmall_periodic as engine
        from student_transition_fast import BatchedTransitionSwiGLUPop
        cfg = read(HERE/'probe_inputs/config.json')[0]['args']
        if cfg['link'] != 'h3' or cfg['m'] != 64:
            raise ValueError('Original full-width h3 only')
        with np.load(HERE/'saved/old_bad_step_order16.npz', allow_pickle=False) as f:
            base = {k:f[k].copy() for k in f.files}
        theta = tuple(base[k] for k in ('P','V','a'))
        with np.load(HERE/'probe_inputs/states.npz', allow_pickle=False) as f:
            if not all(np.array_equal(base[k],f[k][2]) for k in ('P','V','a')):
                raise ValueError('Original state mismatch')
        computed_D = engine.update_directions(base,cfg['m'],cfg['head_lr'])
        dn = np.sqrt(sum(np.sum(x*x) for x in computed_D))
        computed_D = tuple(x/dn for x in computed_D)
        D = tuple(base[k] for k in ('DP','DV','Da'))
        if not all(np.array_equal(x,y) for x,y in zip(D,computed_D)):
            raise ValueError('Saved direction mismatch')
        epsilon = 1e-5*float(max(1,np.sqrt(sum(np.sum(x*x) for x in theta))))
        journal = [json.loads(x) for x in (HERE/'saved/EVALUATIONS.jsonl').read_text().splitlines()]
        minus = [x for x in journal if x['label']=='old_bad_step/o16/minus1e-05' and x['status']=='returned']
        if len(minus) != 1 or epsilon != plan['epsilon']:
            raise ValueError('Reused stencil mismatch')
        minus_loss = float(minus[0]['L'])
        nominal = float(-sum(np.sum(base[k]*d) for k,d in zip(('gP','gV','ga'),D)))
        report.update(epsilon=epsilon,reused_minus_loss=minus_loss,nominal_slope=nominal,
                      reused_base_loss=float(base['L']))
        link, breaks = engine.make_link(cfg['link'])
        T = engine.swpop.Teacher(link,np.asarray(cfg['c']),breaks=breaks,n_x=cfg['n_x'])
        E = BatchedTransitionSwiGLUPop(cfg['d'],T,alpha=cfg['alpha'],
             intercept='refit',bias=True,quad_order=16,quad_limit=12.)
        plus = evaluate(E,tuple(x+epsilon*d for x,d in zip(theta,D)),
                        'old_bad_step/o16/plus1e-05',False)
        plus_loss = float(plus['L'])
        fd = (minus_loss-plus_loss)/(2*epsilon)
        relative = abs(fd-nominal)/max(abs(fd),abs(nominal),1e-30)
        report['small_fd'] = dict(plus_loss=plus_loss,fd_slope=fd,
                                  relative_mismatch=relative,passes_1e3=bool(relative<=1e-3))
        write()
        E10 = BatchedTransitionSwiGLUPop(cfg['d'],T,alpha=cfg['alpha'],
              intercept='refit',bias=True,quad_order=16,quad_limit=10.)
        tail = evaluate(E10,theta,'old_bad_step/tail_limit10',True)
        np.savez_compressed(out/'old_bad_step_order16_limit10.npz',
            **{k:tail[k] for k in ('L','G','t','gP','gV','ga')},
            **{k:base[k] for k in ('P','V','a')})
        report['tail_sensitivity'] = dict(loss_abs=float(abs(tail['L']-base['L'])),
            gradient_max_abs=float(max(np.max(np.abs(tail[k]-base[k])) for k in ('gP','gV','ga'))))
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='incomplete_or_failed',error=repr(exc))
    finally:
        report['inputs_unchanged'] = all(sha(HERE/n)==h for n,h in plan['input_sha256'].items())
        write()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
