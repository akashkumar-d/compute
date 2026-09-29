#!/usr/bin/env python3
"""One explicitly authorized, wall-censored run of the pinned adaptive SwiGLU engine.

No training runs at import. The parent scheduler owns concurrency and the global
deadline. Durable state snapshots and all observed update losses are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

CODE = Path(__file__).resolve().parent


def require_server():
    """This fork allows bounded user-authorized local runs."""
    if os.environ.get('AGOP_EXECUTION_SITE') not in ('LOCAL_AUTHORIZED','SERVER'):
        raise RuntimeError('Explicit LOCAL_AUTHORIZED or SERVER execution site required')


def validate_diagnostic_priority_fractions(value):
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or not 0 < x < 1 for x in value):
        raise ValueError('diagnostic_priority_fractions must be a list of finite fractions strictly between0 and1')
    return value


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Expected a nonempty list of configurations')
    tags = []
    for job in jobs:
        tag = job.get('tag')
        if not isinstance(tag, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag):
            raise ValueError('Each configuration requires a safe nonempty tag')
        tags.append(tag)
        cfg = job['args']
        validate_diagnostic_priority_fractions(cfg.get('diagnostic_priority_fractions', []))
        head_lr = cfg.get('head_lr', 1.0)
        if isinstance(head_lr, bool) or not isinstance(head_lr, (int, float)) or not math.isfinite(head_lr) or head_lr <= 0:
            raise ValueError(f'{tag}: head_lr must be a positive finite number')
        rank, d, width = job['rank'], cfg['d'], cfg['m']
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (rank, d, width)):
            raise ValueError(f'{tag}: rank, dimension and width must be integers')
        if rank < 1 or len(cfg['c']) != rank or d < 2 * rank or width < rank:
            raise ValueError(f'{tag}: require rank=len(c)>=1, d>=2*rank, and m>=rank')
        if not isinstance(job.get('cell'), str) or not job['cell']:
            raise ValueError(f'{tag}: requires a nonempty cell identifier')
    if len(set(tags)) != len(tags):
        raise ValueError('Configuration tags must be unique')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def atomic_json(path, value, finite=False):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value) if finite else value, allow_nan=not finite) + '\n')
    temp.replace(path)


def prefix_index(raw, delta):
    import numpy as np
    loss = np.asarray(raw['Lhist'], dtype=float)[:, 1]
    bad = np.flatnonzero(~np.isfinite(loss) | (np.abs(loss - loss[0]) > delta))
    end = int(bad[0] - 1) if len(bad) else len(loss) - 1
    checkpoint = max((i for i, row in enumerate(raw['rows']) if row['step'] <= end), default=None)
    return end, checkpoint

def ratio_prefix_index(raw, ratio):
    import numpy as np
    loss = np.asarray(raw['Lhist'], dtype=float)[:, 1]
    good = np.isfinite(loss) & (loss > 0) & (np.maximum.accumulate(loss) <= ratio * np.minimum.accumulate(loss))
    bad = np.flatnonzero(~good)
    end = int(bad[0] - 1) if len(bad) else len(loss) - 1
    index = max((i for i, row in enumerate(raw['rows']) if row['step'] <= end), default=None)
    crossing_index = next((i for i, row in enumerate(raw['rows']) if row['step'] == end+1), None) if len(bad) else None
    return dict(ratio=ratio,last_admissible_update=end,checkpoint_index=index,
                first_crossing_update=end+1 if len(bad) else None,
                crossing_checkpoint_index=crossing_index,right_censored=not len(bad))


def diagnostic_order(raw, prefix_data, ratio_prefixes, fractions=None):
    """Reorder existing snapshots only; omitted fractions preserve original order."""
    fractions = validate_diagnostic_priority_fractions([] if fractions is None else fractions)
    rows = raw['rows']
    if not rows:
        return [], [], []
    ratio_selected = [p[key] for p in ratio_prefixes.values()
                      for key in ('checkpoint_index', 'crossing_checkpoint_index') if p[key] is not None]
    selected = list(dict.fromkeys([0] + ratio_selected +
        [v[1] for v in prefix_data.values() if v[1] is not None] + [len(rows)-1]))
    interior, records = [], []
    for name, prefix in ratio_prefixes.items():
        end = prefix['last_admissible_update']
        eligible = [i for i, row in enumerate(rows) if 0 <= row['step'] <= end]
        for fraction in fractions:
            if not eligible:
                continue
            target = fraction * end
            index = min(eligible, key=lambda i: (abs(rows[i]['step']-target), rows[i]['step'], i))
            interior.append(index)
            records.append(dict(prefix=name, fraction=fraction, target_update=target,
                selected_index=index, selected_update=rows[index]['step'],
                inside_initial_prefix=True, ties='earlier saved update'))
    prioritized = list(dict.fromkeys(selected + interior))
    order = prioritized + [i for i in range(len(rows)) if i not in prioritized]
    return selected, order, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CODE / 'configs.json')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--tag')
    selection.add_argument('--index', type=int)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=1200,
                        help='Soft training wall ceiling; current engine evaluation finishes before stopping.')
    parser.add_argument('--diagnostic-seconds', type=float, default=90,
                        help='Soft frozen-diagnostic ceiling, checked before each evaluation.')
    parser.add_argument('--deadline-utc', required=True,
                        help='Global soft computation deadline; preserve time for diagnostics and serialization.')
    parser.add_argument('--double-quadrature', action='store_true',
                        help='Optional doubled-order checks at initialization, primary prefix and terminal state, within diagnostic budget.')
    args = parser.parse_args()
    require_server()
    if json.loads((CODE / 'SOURCE_MANIFEST.json').read_text()).get('status') != 'frozen_after_review':
        raise RuntimeError('Head-rate source bundle is a draft; root review and explicit manifest freeze are required before launch')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
        os.environ[key] = '1'
    import numpy as np
    sys.path.insert(0, str(CODE / 'engine'))
    import swsmall_periodic as engine
    import diagnostics

    jobs = json.loads(args.config.read_text())
    validate_jobs(jobs)
    job = jobs[args.index] if args.index is not None else next(j for j in jobs if j['tag'] == args.tag)
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'result.json'
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get('completed') and prior.get('tag') == job['tag']:
            print(json.dumps({'event': 'skip_completed', 'tag': job['tag']}), flush=True)
            return 0
        raise RuntimeError('Output directory contains a previous result; preserve it and use a new directory')

    deadline_value = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline_value.tzinfo is None:
        raise ValueError('--deadline-utc must include Z or an explicit UTC offset')
    deadline = deadline_value.timestamp()
    assert args.wall_seconds > 0 and args.diagnostic_seconds >= 0
    now = time.time()
    reserve = min(args.diagnostic_seconds, 60.) + 10.
    train_seconds = min(args.wall_seconds, max(0., deadline - now - reserve))
    state = {'stop_cause': None}
    report = {'tag': job['tag'], 'cell': job['cell'], 'rank': job['rank'], 'cfg': job['args'],
              'method': 'block-specific adaptive Euler population flow with finite-order quadrature; all P,V,a train; head multiplier declared separately',
              'head_lr': job['args'].get('head_lr', 1.0),
              'balance_invariant': 'a^2/head_lr - sum(V^2), including value bias',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'effective_training_wall_seconds': train_seconds,
              'global_soft_deadline_utc': args.deadline_utc,
              'diagnostic_budget_seconds': args.diagnostic_seconds,
              'config_sha256': sha(args.config),
              'source_manifest_sha256': sha(CODE / 'SOURCE_MANIFEST.json'),
              'runner_sha256': sha(Path(__file__)),
              'source_files_sha256': {str(p.relative_to(CODE)): sha(p) for p in sorted(CODE.rglob('*.py'))},
              'execution_site': os.environ['AGOP_EXECUTION_SITE'],
              'execution_platform': platform.system(),
              'diagnostic_schema_version': diagnostics.DIAGNOSTIC_SCHEMA_VERSION,
              'agop_resolution_guard': diagnostics.AGOP_RESOLUTION_GUARD, 'completed': False}
    atomic_json(output / 'launch.json', report, finite=True)
    if train_seconds <= 0:
        report.update(status='not_started_deadline', completed=True, wall_seconds=0.)
        atomic_json(result_path, report, finite=True)
        print(json.dumps({'event': 'not_started_deadline', 'tag': job['tag']}), flush=True)
        return 0

    def stop(cause):
        if state['stop_cause'] is None:
            state['stop_cause'] = cause
        engine._STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, lambda *_: stop('parent_SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop('parent_SIGINT'))
    engine._STOP_REQUESTED = False
    engine._LAST_PARTIAL_WALL = 0.
    # This variable controls recording paths after imports are resolved. Route
    # the unmodified engine's periodic partials into this job's output folder.
    recording_anchor = output / '_recording_location'
    recording_anchor.mkdir(exist_ok=True)
    engine._here = str(recording_anchor)
    raw_path = output / (job['tag'] + '.json')
    cfg = dict(job['args'], out=str(raw_path), verbose=True)
    timer = threading.Timer(train_seconds, lambda: stop('training_wall_budget'))
    timer.daemon = True
    timer.start()
    print(json.dumps({'event': 'training_started', 'tag': job['tag'], 'wall_seconds': train_seconds}), flush=True)
    training_start = time.time()
    raw = None
    snaps_path = raw_path.with_name(raw_path.stem + '_snaps.npz')
    try:
        raw = engine.run(cfg)
        timer.cancel()
        snapshots = raw.pop('_snaps')
        atomic_json(raw_path, raw)
        temporary = snaps_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, t=np.array([x[0] for x in snapshots]),
                            P=np.array([x[1] for x in snapshots]),
                            V=np.array([x[2] for x in snapshots]),
                            a=np.array([x[3] for x in snapshots]))
        temporary.replace(snaps_path)
    except Exception as exc:
        report['training_exception'] = repr(exc)
        (output / 'training_exception.txt').write_text(traceback.format_exc())
        partial = output / 'partials' / raw_path.name
        partial_snaps = partial.with_name(partial.stem + '_snaps.npz')
        if partial.exists() and partial_snaps.exists():
            raw = json.loads(partial.read_text())
            last = raw['step_history'][-1]
            raw['termination'] = dict(step=last['step'], t=last['t'], L=last['L'],
                                      reason='exception_durable_partial', wall_seconds=time.time()-training_start)
            raw['partial'] = True
            atomic_json(raw_path, raw)
            shutil.copyfile(partial_snaps, snaps_path)
    finally:
        timer.cancel()
    report['training_wall_seconds'] = time.time() - training_start
    report['wall_stop_cause'] = state['stop_cause']
    if raw is None:
        report.update(status='failed_without_durable_state', completed=True, wall_seconds=time.time()-now)
        atomic_json(result_path, report, finite=True)
        return 0

    report['termination'] = raw['termination']
    report['raw_partial'] = raw.get('partial', False)
    report['all_update_count'] = len(raw['Lhist'])
    report['checkpoint_count'] = len(raw['rows'])
    prefix_data = {str(delta): prefix_index(raw, delta) for delta in (.001, .0001)}
    report['prefixes'] = {k: {'last_admissible_update': v[0], 'checkpoint_index': v[1]}
                          for k, v in prefix_data.items()}
    report['ratio_prefixes'] = {str(ratio): ratio_prefix_index(raw, ratio) for ratio in (1.01, 1.05)}
    selected, order, interior_priority = diagnostic_order(raw, prefix_data, report['ratio_prefixes'],
        cfg.get('diagnostic_priority_fractions', []))
    if 'diagnostic_priority_fractions' in cfg:
        report['diagnostic_priority_policy'] = dict(fractions=cfg['diagnostic_priority_fractions'],
            original_priority_indices=selected, interior_choices=interior_priority, full_order=order,
            snapshot_grid_changed=False, omitted_states=False)
    diagnostics_start = time.time()
    diagnostic_end = min(diagnostics_start + args.diagnostic_seconds, deadline - 5.)
    computed = [dict(index=i, step=row['step'], t=row['t'], status='not_evaluated_runtime_budget',
                     original_L=row['L'], original_A_mean=row['A'], original_A_min=row['Amin'],
                     original_A_top=row['Atop'], original_top_r_gap_ratio=row['gap'])
                for i, row in enumerate(raw['rows'])]
    E = diagnostics.make_engine(cfg)
    with np.load(snaps_path) as snapshots:
        for i in order:
            if time.time() >= diagnostic_end or state['stop_cause'] in ('parent_SIGTERM', 'parent_SIGINT'):
                break
            try:
                metrics = diagnostics.metrics(E, snapshots['P'][i], snapshots['V'][i], snapshots['a'][i])
                computed[i].update(status='ok', **metrics)
            except Exception as exc:
                computed[i].update(status='failed', error=repr(exc))
            atomic_json(output / 'diagnostics_partial.json', {'tag': job['tag'], 'rows': computed}, finite=True)
        doubled = []
        if args.double_quadrature:
            high = diagnostics.make_engine(cfg, 2)
            for i in selected:
                if time.time() >= diagnostic_end or state['stop_cause'] in ('parent_SIGTERM', 'parent_SIGINT'):
                    break
                try:
                    metrics = diagnostics.metrics(high, snapshots['P'][i], snapshots['V'][i], snapshots['a'][i])
                    doubled.append(dict(index=i, step=raw['rows'][i]['step'], status='ok', **metrics))
                except Exception as exc:
                    doubled.append(dict(index=i, step=raw['rows'][i]['step'], status='failed', error=repr(exc)))
    report['diagnostics'] = computed
    report['quadrature_double'] = doubled
    report['diagnostic_wall_seconds'] = time.time() - diagnostics_start
    report['diagnostic_success_count'] = sum(row['status'] == 'ok' for row in computed)
    report['status'] = 'loss_stop' if raw['termination']['reason'] == 'loss_stop' else 'censored_or_failed'
    report['raw_files'] = {raw_path.name: sha(raw_path), snaps_path.name: sha(snaps_path)}
    report.update(completed=True, wall_seconds=time.time()-now, ended_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(result_path, report, finite=True)
    print(json.dumps({'event': 'completed', 'tag': job['tag'], 'status': report['status'],
                      'updates': raw['termination']['step'], 'diagnostic_success_count': report['diagnostic_success_count'],
                      'wall_seconds': report['wall_seconds']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
