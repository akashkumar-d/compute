"""Compare actual SwiGLU update directions at fixed saved states, without training."""
import argparse
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, report):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def compare(a, b):
    a, b = np.ravel(a), np.ravel(b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    error = float(np.linalg.norm(a - b))
    cosine = float(np.clip(a @ b / (na * nb), -1, 1)) if na * nb > 0 else None
    return dict(low_norm=na, high_norm=nb, absolute_difference=error,
                relative_difference=error / max(nb, 1e-300),
                angle_radians=float(np.arccos(cosine)) if cosine is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', type=Path, required=True)
    ap.add_argument('--execution', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--multipliers', default='1,2,4', help='Increasing positive integer factors, e.g.1,2')
    args = ap.parse_args()
    factors = [int(x) for x in args.multipliers.split(',')]
    if len(factors) < 2 or factors != sorted(set(factors)) or any(x < 1 or x > 4 for x in factors):
        ap.error('Use at least two distinct increasing integer multipliers in1..4')
    if args.out.exists():
        raise FileExistsError('Use a new output path; previous audits are preserved.')
    code = args.code.resolve()
    sys.path.insert(0, str(code))
    import diagnostics
    import swsmall_periodic as training
    sources = {str(p): sha(p) for p in sorted(code.rglob('*.py'))}
    selection = json.loads(args.selection.read_text())
    record = dict(scope='Fixed saved states only; no update is applied and no trajectory is certified.',
                  no_training=True, multipliers=factors, inputs={str(args.selection): sha(args.selection)},
                  source_hashes=sources, rows=[], complete=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for tag, steps in selection.items():
        folder = args.execution / 'data' / tag
        raw, snap, receipt = folder / (tag + '.json'), folder / (tag + '_snaps.npz'), folder / 'result.json'
        for p in (raw, snap, receipt):
            record['inputs'][str(p)] = sha(p)
        assert json.loads(receipt.read_text())['completed'] is True
        data = json.loads(raw.read_text())
        cfg, states = data['cfg'], np.load(snap)
        engines = {k: diagnostics.make_engine(cfg, k) for k in record['multipliers']}
        for step in steps:
            indices = [i for i, row in enumerate(data['rows']) if row['step'] == step]
            assert len(indices) == 1
            i = indices[0]
            assert abs(float(states['t'][i]) - data['rows'][i]['t']) < 1e-9
            P, V, a = states['P'][i], states['V'][i], states['a'][i]
            theta = np.sqrt((P ** 2).sum(0) + (V ** 2).sum(0) + a ** 2)
            row = dict(tag=tag, index=i, step=step, time=float(states['t'][i]),
                       head_lr=cfg['head_lr'], orders={}, comparisons={})
            cache = {}
            for factor, engine in engines.items():
                ev = engine.evaluate(P, V, a, need_grad=True)
                raw_blocks = {k: ev[k] for k in ('gP', 'gV', 'ga')}
                direction = training.update_directions(ev, cfg['m'], cfg['head_lr'])
                GP, GV, Ga = direction
                gn = np.sqrt((GP ** 2).sum(0) + (GV ** 2).sum(0) + Ga ** 2)
                rel_per_neuron = gn / np.maximum(theta, 1e-300)
                rel = float(rel_per_neuron.max())
                dt = min(cfg['dt_max'], cfg['h'] / max(rel, 1e-300))
                vector = np.concatenate([b.ravel() for b in direction])
                M = engine.agop(P, V, a, ev)
                M = (M + M.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(M)
                r = len(cfg['c'])
                Q = eigenvectors[:, -r:]
                cos2 = np.linalg.svd(Q[:r], compute_uv=False) ** 2
                gap = float(eigenvalues[-r] - eigenvalues[-r-1])
                row['orders'][str(factor)] = dict(raw_loss=float(ev['L']),
                    quadrature={k: cfg[k] * factor for k in ('n_pair', 'n_diag', 'n_z', 'n_x')},
                    raw_gradient_norms={k: float(np.linalg.norm(v)) for k, v in raw_blocks.items()},
                    actual_direction_norm=float(np.linalg.norm(vector)), relative_rate_max=rel,
                    relative_rate_per_neuron=rel_per_neuron.tolist(), cap_neuron=int(np.argmax(rel_per_neuron)),
                    dt=dt, actual_step_norm=float(np.linalg.norm(dt * vector)),
                    agop_Amin=float(cos2.min()), agop_Amean=float(cos2.mean()),
                    agop_rank_r_eigenvalue=float(eigenvalues[-r]),
                    agop_next_eigenvalue=float(eigenvalues[-r-1]), agop_gap=gap,
                    agop_relative_gap=gap / max(abs(float(eigenvalues[-1])), 1e-300),
                    gram_asymmetry=float(np.linalg.norm(ev['G'] - ev['G'].T) / max(np.linalg.norm(ev['G']), 1e-300)))
                cache[factor] = dict(raw=raw_blocks, direction=vector, step=dt * vector,
                                     M=M, Q=Q, gap=gap, dt=dt)
            for low, high in combinations(factors, 2):
                l, h = cache[low], cache[high]
                perturbation = float(np.linalg.norm(h['M'] - l['M'], ord=2))
                singular = np.linalg.svd(l['Q'].T @ h['Q'], compute_uv=False)
                row['comparisons'][f'{low}_vs_{high}'] = dict(
                    raw_blocks={k: compare(l['raw'][k], h['raw'][k]) for k in l['raw']},
                    actual_direction=compare(l['direction'], h['direction']),
                    actual_step=compare(l['step'], h['step']),
                    dt_relative_difference=abs(l['dt'] - h['dt']) / max(h['dt'], 1e-300),
                    agop_operator_difference=perturbation,
                    agop_operator_difference_over_high_gap=perturbation / max(h['gap'], 1e-300),
                    learned_subspace_min_cosine_squared=float(singular[-1] ** 2),
                    learned_projector_operator_difference=float(np.linalg.norm(l['Q'] @ l['Q'].T - h['Q'] @ h['Q'].T, ord=2)))
            record['rows'].append(row)
            write(args.out, record)
            print(json.dumps(dict(tag=tag, step=step, elapsed_seconds=time.monotonic()-started)), flush=True)
        states.close()
    record.update(complete=True, wall_seconds=time.monotonic()-started,
                  inputs_unchanged=all(sha(Path(p)) == h for p, h in record['inputs'].items()),
                  sources_unchanged=all(sha(Path(p)) == h for p, h in sources.items()),
                  script_sha256=sha(Path(__file__)))
    assert record['inputs_unchanged'] and record['sources_unchanged']
    write(args.out, record)


if __name__ == '__main__':
    main()
