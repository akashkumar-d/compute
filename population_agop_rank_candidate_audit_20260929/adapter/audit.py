"""Server-only fixed saved-state audit. No initialization, update or integration."""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.util
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'launcher'))
import launch


def verify_inputs(case):
    folder = Path(case['remote_source_folder']).resolve(strict=True)
    result = {}
    for name, expected in case['files'].items():
        path = (folder / name).resolve(strict=True)
        if Path(name).name != name or path.parent != folder:
            raise ValueError('Input escaped the pinned case directory')
        result[name] = dict(size=path.stat().st_size, sha256=launch.sha256(path))
        if result[name] != expected:
            raise ValueError('Pinned external input changed: ' + name)
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CachedEvaluation:
    """Give unchanged diagnostics.metrics the exact already-computed ev/AGOP."""
    def __init__(self, engine, parameters, ev, matrix):
        self.engine, self.parameters, self.ev, self.matrix = engine, parameters, ev, matrix
        self.T = engine.T

    def evaluate(self, P, V, a, need_grad=True):
        if not all(x is y for x, y in zip((P, V, a), self.parameters)) or not need_grad:
            raise ValueError('Cached diagnostic evaluation identity mismatch')
        return self.ev

    def agop(self, P, V, a, ev):
        if ev is not self.ev or not all(x is y for x, y in zip((P, V, a), self.parameters)):
            raise ValueError('Cached AGOP identity mismatch')
        return self.matrix

    def refit(self, ev):
        if ev is not self.ev:
            raise ValueError('Cached refit identity mismatch')
        return self.engine.refit(ev)


def order_record(np, diagnostics, training, engine, cfg, factor, P, V, a):
    ev = engine.evaluate(P, V, a, need_grad=True)
    raw = {k: ev[k] for k in ('gP', 'gV', 'ga')}
    direction = training.update_directions(ev, cfg['m'], cfg['head_lr'])
    GP, GV, Ga = direction
    theta = np.sqrt((P ** 2).sum(0) + (V ** 2).sum(0) + a ** 2)
    rate = np.sqrt((GP ** 2).sum(0) + (GV ** 2).sum(0) + Ga ** 2) / np.maximum(theta, 1e-300)
    rel = float(rate.max())
    dt = min(cfg['dt_max'], cfg['h'] / max(rel, 1e-300))
    vector = np.concatenate([x.ravel() for x in direction])
    raw_vector = np.concatenate([raw[k].ravel() for k in ('gP', 'gV', 'ga')])
    matrix = engine.agop(P, V, a, ev)
    if not all(np.isfinite(x).all() for x in [*raw.values(), vector, matrix]):
        raise ValueError('Nonfinite gradient/direction/AGOP')
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    rank = len(cfg['c']); Q = vectors[:, -rank:]
    gap = float(values[-rank] - values[-rank - 1])
    # Same frozen refit/guard formulas, with no second model evaluation.
    metrics = diagnostics.metrics(CachedEvaluation(engine, (P, V, a), ev, matrix), P, V, a)
    record = dict(quadrature={k: cfg[k] * factor for k in ('n_pair', 'n_diag', 'n_z', 'n_x')},
        raw_loss=float(ev['L']), normalization_variance=float(ev['V']),
        raw_gradient_norms={k: float(np.linalg.norm(v)) for k, v in raw.items()},
        raw_gradient_norm=float(np.linalg.norm(raw_vector)),
        actual_direction_norm=float(np.linalg.norm(vector)),
        actual_direction_block_norms={k: float(np.linalg.norm(v)) for k, v in zip(('P', 'V', 'a'), direction)},
        relative_rate_max=rel, relative_rate_per_neuron=rate.tolist(), cap_neuron=int(np.argmax(rate)),
        dt=dt, actual_step_norm=float(np.linalg.norm(dt * vector)),
        agop_rank_r_eigenvalue=float(values[-rank]), agop_next_eigenvalue=float(values[-rank - 1]),
        agop_gap=gap, agop_relative_gap=gap / max(abs(float(values[-1])), 1e-300),
        metrics=metrics)
    cache = dict(raw=raw, raw_vector=raw_vector, direction=vector, blocks=direction,
                 step=dt * vector, M=matrix, Q=Q, gap=gap, dt=dt)
    return record, cache


def compare_orders(np, compare, low, high):
    perturbation = float(np.linalg.norm(high['M'] - low['M'], ord=2))
    singular = np.clip(np.linalg.svd(low['Q'].T @ high['Q'], compute_uv=False), 0., 1.)
    return dict(raw_gradient=compare(low['raw_vector'], high['raw_vector']),
        raw_blocks={k: compare(low['raw'][k], high['raw'][k]) for k in low['raw']},
        actual_direction=compare(low['direction'], high['direction']),
        actual_direction_blocks={k: compare(x, y) for k, x, y in zip(('P', 'V', 'a'), low['blocks'], high['blocks'])},
        actual_step=compare(low['step'], high['step']),
        dt_relative_difference=abs(low['dt'] - high['dt']) / max(high['dt'], 1e-300),
        agop_operator_difference=perturbation,
        agop_operator_difference_over_high_gap=perturbation / max(high['gap'], 1e-300),
        learned_subspace_min_cosine_squared=float(singular[-1] ** 2),
        principal_angles_radians=np.arccos(singular).tolist(),
        learned_projector_operator_difference=float(np.linalg.norm(low['Q'] @ low['Q'].T - high['Q'] @ high['Q'].T, ord=2)))


def qualify(canonical, case, rows):
    """Apply existing per-arm thresholds with same-order initial diagnostics.

    Empty history intentionally prevents any new later-release/trajectory claim.
    """
    for factor in ('1', '2'):
        ready = [row for row in rows if factor in row['orders']]
        if not ready or ready[0]['step'] != 0:
            continue
        variance = ready[0]['orders'][factor]['normalization_variance']
        points = []
        for row in ready:
            metrics = row['orders'][factor]['metrics']
            points.append(canonical.swiglu_point(dict(step=row['step'], t=row['time'], L=metrics['L']),
                dict(metrics, status='ok', t=row['time']), variance))
        mini = dict(checkpoints=points, history=[], prefixes={}, target_variance=variance,
                    student='swiglu', issues=[], configuration_matches=True, snapshot_integrity_verified=True)
        canonical.evaluate_arm(mini)
        for row, point in zip(ready, mini['checkpoints']):
            row['orders'][factor]['fixed_state_qualification'] = {k: point[k] for k in (
                'A_min_gain', 'refit_gain_lower_raw', 'refit_gain_lower_over_variance',
                'refit_sensitivity_spread_normalized', 'refit_normal_residual_max',
                'agop_screen', 'refit_screen', 'same_checkpoint_material_candidate_original',
                'same_checkpoint_material_candidate_numerically_qualified')}
    for row in rows:
        if all('fixed_state_qualification' in row['orders'].get(k, {}) for k in ('1', '2')):
            row['native_order_fixed_candidate_qualifies'] = row['orders']['1']['fixed_state_qualification']['same_checkpoint_material_candidate_numerically_qualified']
            row['doubled_order_fixed_candidate_qualifies'] = row['orders']['2']['fixed_state_qualification']['same_checkpoint_material_candidate_numerically_qualified']
            row['fixed_candidate_thresholds_survive_doubling'] = bool(row['native_order_fixed_candidate_qualifies'] and row['doubled_order_fixed_candidate_qualifies'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--diagnostic-seconds', type=float, required=True); ap.add_argument('--deadline-utc', required=True)
    args = ap.parse_args(argv)
    launch.require_server(); manifest, digest = launch.validate(ROOT)
    launch.validate_protocol(manifest, ROOT); launch.require_review(ROOT, digest)
    launch.require_capacity(manifest['runtime']); launch.require_priority()
    if os.environ.get('AGOP_AUDIT_LAUNCHER_PID') != str(os.getppid()):
        raise RuntimeError('Use the reviewed scheduler parent')
    if any(os.environ.get(k) != '1' for k in launch.THREAD_VARIABLES) or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Single-thread CPU environment required')
    selection = launch.load_json(ROOT / 'SELECTION.json')
    case = next(c for c in selection['cases'] if c['id'] == args.tag)
    if args.config.resolve() != (ROOT / case['config_path']).resolve():
        raise ValueError('Pinned configuration required')
    if not math.isfinite(args.diagnostic_seconds) or not 0 < args.diagnostic_seconds <= 250:
        raise ValueError('Case soft budget must be in (0,250] seconds')
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None: raise ValueError('Timezone required')
    out = args.out_dir.resolve()
    if out.parent.name != 'data' or out.parent.parent.parent != ROOT or out.name != args.tag or out.exists():
        raise ValueError('Require fresh scheduler output inside this bundle')
    inputs = verify_inputs(case)
    cfg = case['config'][0]['args']; folder = Path(case['remote_source_folder'])
    raw = launch.load_json(folder / (args.tag + '.json'))
    receipt = launch.load_json(folder / 'result.json')
    if receipt['cfg'] != cfg or {k: v for k, v in raw['cfg'].items() if k not in ('out', 'verbose')} != cfg:
        raise ValueError('Original input configuration changed')
    if receipt['source_files_sha256'] != case['original_source_hashes']:
        raise ValueError('Original scientific source identity changed')
    for name, expected in case['original_source_hashes'].items():
        if launch.sha256(ROOT / 'swiglu/code' / name) != expected:
            raise ValueError('Scientific source differs from original')
    # Every admission/hash/review/priority gate precedes scientific imports.
    import numpy as np
    sys.path.insert(0, str(ROOT / 'swiglu/code'))
    import diagnostics
    import swsmall_periodic as training
    prior = module(ROOT / 'reference/weighted_update_audit.py', 'reviewed_weighted_audit')
    canonical = module(ROOT / 'reference/canonical_summary.py', 'canonical_saved_criteria')
    out.mkdir(exist_ok=False)
    rows = [dict(tag=args.tag, index=s['index'], step=s['step'], time=s['time'], roles=s['roles'],
                 head_lr=cfg['head_lr'], status='pending', orders={}, comparisons={}) for s in case['states']]
    record = dict(tag=args.tag, manifest_sha256=digest, selection_sha256=launch.sha256(ROOT / 'SELECTION.json'),
        source_sha256=manifest['source_sha256'], inputs_before=inputs, inputs_unchanged_after=None,
        actual_head_lr=cfg['head_lr'], actual_direction_formula='m*(gP,gV,head_lr*ga)',
        cap_definition='full per-neuron P,V,a norm including bias coordinates; dt=min(dt_max,h/max(relative_rate,1e-300))',
        comparison_denominator='higher-order norm; dt uses higher-order dt; errors are fractions, not percentages',
        scope='Selected development states only; no updates or full-trajectory/independent-seed certificate.',
        rows=rows, audit_flags=[], native_vs_saved_absolute_tolerance=1e-9,
        new_training_updates=0, completed=False, started_utc=launch.utc(), effective_nice=os.nice(0))
    start = time.monotonic(); end = start + min(args.diagnostic_seconds, max(0, deadline.timestamp() - time.time() - 5))
    stop = {'signal': None}
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda n, frame: stop.update(signal=n))
    def persist():
        record.update(elapsed_seconds=time.monotonic()-start, interruption_signal=stop['signal'])
        launch.atomic_json(out / 'FIXED_STATE_AUDIT.json', canonical.clean(record))
    persist()
    with np.load(folder / (args.tag + '_snaps.npz'), allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('t', 'P', 'V', 'a')}
    expected = {'t': (len(raw['rows']),), 'P': (len(raw['rows']), cfg['d']+1, cfg['m']),
                'V': (len(raw['rows']), cfg['d']+1, cfg['m']), 'a': (len(raw['rows']), cfg['m'])}
    for key, arr in arrays.items():
        if arr.shape != expected[key] or not np.isfinite(arr).all(): raise ValueError('Invalid saved array: '+key)
        arr.flags.writeable = False
    engines = {}
    for row, selected in zip(rows, case['states']):
        if stop['signal'] or time.monotonic() >= end: row['status']='not_evaluated_budget_or_signal'; persist(); continue
        try:
            i = row['index']; saved = raw['rows'][i]
            if saved['step'] != row['step'] or saved['t'] != row['time'] or float(arrays['t'][i]) != row['time']:
                raise ValueError('Pinned checkpoint identity mismatch')
            P, V, a = [arrays[k][i] for k in ('P', 'V', 'a')]; cache = {}
            for factor in (1, 2):
                if stop['signal'] or time.monotonic() >= end: raise TimeoutError('Fixed-state budget or signal')
                if factor not in engines: engines[factor] = diagnostics.make_engine(cfg, factor)
                row['orders'][str(factor)], cache[factor] = order_record(np, diagnostics, training, engines[factor], cfg, factor, P, V, a)
                row['status'] = 'partially_evaluated'; qualify(canonical, case, rows); persist()
            row['comparisons']['1_vs_2'] = compare_orders(np, prior.compare, cache[1], cache[2])
            native = row['orders']['1']['metrics']; old = selected['original_diagnostic']
            row['native_vs_saved'] = {k: native[k] - old[k] for k in ('L', 'agop_Amin', 'agop_A')}
            row['native_vs_saved']['cutoff_risk_differences'] = {k: native['pinv'][k]['actual_mse'] - old['pinv'][k]['actual_mse'] for k in old['pinv']}
            differences = [row['native_vs_saved'][k] for k in ('L','agop_Amin','agop_A')] + list(row['native_vs_saved']['cutoff_risk_differences'].values())
            row['native_vs_saved_mismatch'] = any(not math.isfinite(v) or abs(v)>1e-9 for v in differences)
            row['native_vs_saved_max_absolute_difference'] = max(abs(v) for v in differences)
            if row['native_vs_saved_mismatch']:
                record['audit_flags'].append(dict(flag='NATIVE_REPRODUCTION_MISMATCH',step=row['step'],absolute_tolerance=1e-9))
            row['status'] = 'ok'
        except Exception as exc:
            row.update(status='failed_or_censored', error=f'{type(exc).__name__}: {exc}')
        qualify(canonical, case, rows); persist()
    record['inputs_after'] = verify_inputs(case); launch.validate(ROOT, digest)
    record.update(inputs_unchanged_after=True, completed=all(r['status']=='ok' for r in rows),
        native_reproduction_all_pass=all(r.get('native_vs_saved_mismatch') is False for r in rows), ended_utc=launch.utc())
    persist()
    return 0 if record['completed'] else 124


if __name__ == '__main__':
    raise SystemExit(main())
