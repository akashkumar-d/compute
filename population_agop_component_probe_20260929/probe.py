"""SERVER-only component directional probe at frozen saved states; no training."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class Capped(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    output = args.output.resolve()
    if output.parent != HERE or output.exists():
        raise ValueError('Use a fresh direct-child output directory')
    plan_path = HERE / 'PROBE_MANIFEST.json'
    plan, plan_hash = read(plan_path), sha(plan_path)
    for name, digest in plan['input_sha256'].items():
        if sha(HERE / name) != digest:
            raise ValueError('Pinned probe input changed: ' + name)
    review = read(HERE / 'PROBE_REVIEW.json')
    if review.get('status') != 'approved_for_execution' or review.get('manifest_sha256') != plan_hash:
        raise RuntimeError('Independent exact-manifest component probe review required')
    guard = module(HERE / 'probe_resources.py', 'reviewed_component_resource_guards')
    guard.require_server()
    resources = guard.require_capacity(dict(workers=1, reserved_cpus=31,
        min_available_memory_gib=8, min_free_disk_gib=5))
    guard.require_priority()
    allocated = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allocated[-1]})
    for key in guard.THREAD_VARIABLES:
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    start = time.monotonic()
    deadline = start + 170.
    stopped = [False]
    counts = dict(evaluate=0, teacher_terms=0)
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    output.mkdir()
    report = dict(status='started', manifest_sha256=plan_hash, resources=resources,
        active_affinity=sorted(os.sched_getaffinity(0)), nice=os.nice(0),
        new_training_updates=0, input_sha256=plan['input_sha256'],
        slots=[dict(state=s['name'], step=s['step'], order_multiplier=k,
                    status='not_attempted', directional_differences=[
                        dict(fraction=f, status='not_attempted') for f in plan['epsilon_fractions']])
               for k in plan['order_multipliers'] for s in plan['states']])

    def write():
        report['model_evaluations'] = counts['evaluate']
        report['separate_teacher_term_calls'] = counts['teacher_terms']
        report['kernel_calls_started'] = sum(counts.values())
        report['elapsed_seconds'] = time.monotonic() - start
        tmp = output / 'RESULTS.json.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(output / 'RESULTS.json')

    def check():
        if stopped[0] or time.monotonic() >= deadline:
            raise Capped('Signal or 170-second soft deadline')

    def event(value):
        with (output / 'EVALUATIONS.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')

    write()
    arrays = None
    try:
        # All source/input pins and runtime gates precede scientific imports.
        import numpy as np
        sys.path.insert(0, str(HERE / 'engine'))
        import swsmall_periodic as engine
        from component_report import directional_report, stability, cancellation
        cfg = read(HERE / 'probe_inputs/config.json')[0]['args']
        if cfg['link'] != 'h3':
            raise ValueError('Component formula is restricted to the pinned pure h3 teacher')
        arrays = np.load(HERE / 'probe_inputs/states.npz', allow_pickle=False)
        for multiplier in plan['order_multipliers']:
            for spec in plan['states']:
                slot = next(s for s in report['slots']
                    if s['state'] == spec['name'] and s['order_multiplier'] == multiplier)
                theta = None
                try:
                    check()
                    slot['status'] = 'running'
                    write()
                    order = {key: cfg[key] * multiplier for key in ('n_pair', 'n_diag', 'n_z', 'n_x')}
                    link, breaks = engine.make_link(cfg['link'])
                    teacher = engine.swpop.Teacher(link, np.asarray(cfg['c'], float),
                        breaks=breaks, n_x=order['n_x'])
                    scope = dict(pure_h3=bool(link.terms == ((1.0, 'h3'),)),
                        raw_mean=float(link.gaussian_moments[0]), raw_second=float(link.gaussian_moments[1]),
                        EY=float(teacher.EY), variance=float(teacher.V),
                        zero_mean=bool(teacher.EY == 0.0), unit_variance=bool(teacher.V == 1.0))
                    slot.update(orders=order, teacher_scope=scope)
                    if not (scope['pure_h3'] and scope['raw_mean'] == 0.0 and scope['raw_second'] == 1.0
                            and scope['zero_mean'] and scope['unit_variance']):
                        raise ValueError('Pure h3 / EY=0 / VarY=1 scope gate failed')
                    E = engine.swpop.SwiGLUPop(cfg['d'], teacher, alpha=cfg['alpha'],
                        intercept='refit', bias=True, n_pair=order['n_pair'],
                        n_diag=order['n_diag'], n_z=order['n_z'])
                    theta = tuple(arrays[key][spec['extracted_index']].copy() for key in ('P', 'V', 'a'))
                    original_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in theta]
                    al = E.alpha / cfg['m']

                    def call(kind, label, function):
                        check()
                        if sum(counts.values()) >= 24:
                            raise Capped('24-kernel-call cap')
                        counts[kind] += 1
                        receipt = dict(index=sum(counts.values()), kind=kind, label=label,
                            state=spec['name'], order_multiplier=multiplier)
                        event(dict(receipt, status='started', elapsed_seconds=time.monotonic()-start))
                        value = function()
                        event(dict(receipt, status='returned', elapsed_seconds=time.monotonic()-start))
                        check()
                        return value

                    def components(state, ev):
                        a = state[2]
                        result = dict(total=float(ev['L']/teacher.V),
                            teacher=float(-2*al*(a @ ev['t'])/teacher.V),
                            student=float(al*al*(a @ ev['G'] @ a)/teacher.V))
                        if not all(np.isfinite(v) for v in result.values()):
                            raise ValueError('Nonfinite loss component')
                        result['reconstruction_residual'] = float(result['total']-1-result['teacher']-result['student'])
                        return result

                    base = call('evaluate', 'base', lambda: E.evaluate(*theta, need_grad=True))
                    D = engine.update_directions(base, cfg['m'], cfg['head_lr'])
                    if not all(np.isfinite(x).all() for x in D):
                        raise ValueError('Nonfinite weighted full-gradient direction')
                    total_slope = float(-sum(np.sum(base[k]*g) for k, g in zip(('gP','gV','ga'), D))/teacher.V)
                    t, gpt, gvt = call('teacher_terms', 'base_teacher_gradient',
                        lambda: E.teacher_terms(*E._split(theta[0], theta[1])))
                    gTeacher = (-2*al*theta[2][None,:]*gpt, -2*al*theta[2][None,:]*gvt, -2*al*t)
                    teacher_slope = float(-sum(np.sum(g*v) for g, v in zip(gTeacher, D))/teacher.V)
                    nominal = dict(total=total_slope, teacher=teacher_slope,
                        student=float(total_slope-teacher_slope))
                    if not all(np.isfinite(v) for v in nominal.values()):
                        raise ValueError('Nonfinite nominal component slope')
                    base_values = components(theta, base)
                    slot.update(base=base_values, nominal_slopes=nominal,
                        nominal_cancellation=cancellation(nominal), total_nominal_negative=bool(total_slope<0),
                        teacher_t_max_abs_difference=float(np.max(np.abs(t-base['t']))),
                        teacher_t_matches=bool(np.array_equal(t, base['t'])),
                        saved_native_loss=spec['saved_loss'],
                        saved_native_loss_difference=float(base_values['total']-spec['saved_loss']) if multiplier==1 else None,
                        old_proposed_dt=spec['old_proposed_dt'])
                    write()
                    for fdslot in slot['directional_differences']:
                        fraction = fdslot['fraction']
                        epsilon = float(spec['old_proposed_dt']*fraction)
                        fdslot.update(status='running', epsilon=epsilon)
                        write()
                        minus = tuple(x-epsilon*g for x,g in zip(theta,D))
                        evminus = call('evaluate', f'fd_minus_{fraction}', lambda: E.evaluate(*minus, need_grad=False))
                        lower = components(minus, evminus)
                        fdslot['minus'] = lower
                        write()
                        plus = tuple(x+epsilon*g for x,g in zip(theta,D))
                        evplus = call('evaluate', f'fd_plus_{fraction}', lambda: E.evaluate(*plus, need_grad=False))
                        upper = components(plus, evplus)
                        fdslot['plus'] = upper
                        fdslot.update(directional_report(base_values, lower, upper, nominal, epsilon))
                        fdslot['status'] = 'complete'
                        write()
                    slot['two_scale_checks'] = stability(slot['directional_differences'])
                    slot['status'] = 'complete'
                except Capped as exc:
                    slot.update(status='capped', error=str(exc))
                except Exception as exc:
                    slot.update(status='failed', error=repr(exc))
                finally:
                    if theta is not None:
                        slot['base_arrays_unchanged'] = bool(original_hashes ==
                            [hashlib.sha256(x.tobytes()).hexdigest() for x in theta])
                    for fdslot in slot['directional_differences']:
                        if fdslot['status'] != 'complete':
                            fdslot.update(status='capped' if slot['status']=='capped' else 'missing_after_failure')
                    write()
        report['status'] = 'complete' if all(s['status']=='complete' for s in report['slots']) else 'incomplete_or_failed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
    finally:
        if arrays is not None:
            arrays.close()
        for slot in report['slots']:
            if slot['status']=='not_attempted':
                slot['status']='not_attempted_after_failure'
                for fdslot in slot['directional_differences']:
                    fdslot['status']='not_attempted_after_failure'
        report['inputs_unchanged'] = bool(all(sha(HERE/n)==h for n,h in plan['input_sha256'].items()))
        write()
    return 0 if report['status']=='complete' and report['inputs_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
