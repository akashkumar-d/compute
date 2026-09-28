"""Audit saved paired initializations without importing or evaluating a model.

API: audit(manifest_path, execution_path) returns a JSON-serializable dictionary.
Final result.raw_files metadata locates the JSON/NPZ pair; raw rows locate step 0.
Unlinked partials are unresolved, never inferred to match. Inputs are read only.
Only float64 arrays from the frozen writer are supported. NumPy loads saved NPZ
data with allow_pickle=False; only initial parameter slices are compared.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
QS = (1.0, 0.3, 0.1)
SEEDS = (641, 642)
HEAD_RTOL = 4 * sys.float_info.epsilon


class Unresolved(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Unresolved(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.missing = set()

    def track(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            self.missing.add(str(path))
            raise FileNotFoundError(str(path))
        value = digest(path)
        require(str(path) not in self.hashes or self.hashes[str(path)] == value,
                'input changed between reads: ' + str(path))
        self.hashes[str(path)] = value
        return value

    def json(self, path):
        self.track(path)
        return json.loads(Path(path).read_text())

    def changed(self):
        result = []
        for path, value in self.hashes.items():
            try:
                if digest(path) != value:
                    result.append(path)
            except OSError:
                result.append(path)
        result.extend(path for path in self.missing if Path(path).exists())
        return sorted(set(result))


def local_reference(folder, name):
    require(isinstance(name, str) and not Path(name).is_absolute(),
            'raw_files must contain relative file references')
    path = (folder / name).resolve()
    require(path.is_relative_to(folder.resolve()), 'raw_files reference escapes arm folder')
    return path


def load_initial(job, execution, inputs):
    ident = job['id']
    folder = execution / 'data' / ident
    result = inputs.json(folder / 'result.json')
    cfg = job['config'][0]['args']
    require(result.get('tag') == ident, 'result tag does not match planned ID')
    require(all(result.get('cfg', {}).get(k) == v for k, v in cfg.items()),
            'result configuration differs from frozen manifest')
    files = result.get('raw_files', {})
    require(isinstance(files, dict) and files, 'no recorded raw_files mapping; partials remain unresolved')
    json_names = [name for name in files if Path(name).suffix == '.json']
    npz_names = [name for name in files if Path(name).suffix == '.npz']
    require(len(json_names) == len(npz_names) == 1, 'ambiguous recorded JSON/NPZ pair')
    raw_path = local_reference(folder, json_names[0])
    snapshot_path = local_reference(folder, npz_names[0])
    for name, path in ((json_names[0], raw_path), (npz_names[0], snapshot_path)):
        require(inputs.track(path) == files[name], 'recorded raw file hash mismatch: ' + name)
    raw = inputs.json(raw_path)
    require(all(raw.get('cfg', {}).get(k) == v for k, v in cfg.items()),
            'raw configuration differs from frozen manifest')
    rows = raw.get('rows', [])
    require(isinstance(rows, list) and rows, 'no recorded checkpoints')
    initial = [i for i, row in enumerate(rows) if row.get('step') == 0]
    require(len(initial) == 1, 'initial step missing or duplicated')
    index = initial[0]
    require(index == 0 and rows[index].get('t') == 0.0,
            'recorded initial step/time/order does not match frozen writer')
    require(all(type(row.get('step')) is int and row['step'] >= 0 for row in rows),
            'invalid recorded checkpoint step')
    require(all(rows[i]['step'] < rows[i + 1]['step'] for i in range(len(rows) - 1)),
            'checkpoint steps are not strictly increasing')
    n, d, m = len(rows), cfg['d'], cfg['m']
    with np.load(snapshot_path, allow_pickle=False) as snapshots:
        require(sorted(snapshots.files) == ['P', 'V', 'a', 't'],
                'unexpected or duplicate NPZ members')
        shapes = {'t': (n,), 'P': (n, d + 1, m), 'V': (n, d + 1, m), 'a': (n, m)}
        arrays = {key: snapshots[key] for key in shapes}
        for key, array in arrays.items():
            require(array.shape == shapes[key] and array.dtype == np.dtype('<f8'),
                    'expected float64 shape for ' + key)
        times = arrays['t'].tolist()
        require(all(math.isfinite(t) for t in times), 'nonfinite saved time')
        require(times == [row.get('t') for row in rows], 'NPZ time/recorded row index disagreement')
        require(times[index] == 0.0 and all(times[i] < times[i + 1] for i in range(n - 1)),
                'invalid initial or nonmonotone checkpoint time')
        require(all(np.isfinite(arrays[key][index]).all() for key in ('P', 'V', 'a')),
                'nonfinite saved initial parameters')
        P = arrays['P'][index].tobytes(order='C')
        V = arrays['V'][index].tobytes(order='C')
        a_bytes = arrays['a'][index].tobytes(order='C')
        head = arrays['a'][index].tolist()
    q = job['q']
    normalized = [value / q for value in head]
    require(all(math.isfinite(x) for x in normalized), 'nonfinite normalized head')
    public = dict(id=ident, seed=cfg['seed'], q=q, status='ready', initial_index=index,
                  initial_step=rows[index]['step'], initial_time=times[index], snapshot_count=n,
                  raw_json=str(raw_path), snapshot_npz=str(snapshot_path), raw_partial=bool(raw.get('partial', False)),
                  source_result_completed=result.get('completed'), raw_head=head, normalized_head=normalized,
                  initial_shapes={'P': [d + 1, m], 'V': [d + 1, m], 'a': [m]}, dtype='<f8',
                  initial_array_sha256={key: hashlib.sha256(value).hexdigest()
                                        for key, value in [('P', P), ('V', V), ('a', a_bytes)]})
    return public, (P, V, a_bytes, head)


def audit(manifest_path, execution_path):
    """Read only: return all six arm statuses and four within-seed comparisons."""
    inputs = Inputs()
    report = dict(schema='headscale_paired_initialization_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  status='unresolved', **{'pass': False}, arms=[], comparisons=[], issues=[],
                  method='Recorded result.raw_files -> raw.rows step0/time0 index -> matching NPZ t/P/V/a initial slices; no RNG regeneration.',
                  head_tolerance={'rtol': HEAD_RTOL, 'atol': 0.0,
                                  'justification': 'a=(Gaussian*s)*q and a/q add one binary64 multiplication and division; 4 epsilon relative allowance covers their roundoff. Exact forward a_q=q*a_1 is separately required.'},
                  raw_head_equality_required=False, expected_nontrivial_pairs=4, no_model_imports=True,
                  no_model_evaluation=True, training_runs=0,
                  caveats=['Initialization pairing alone does not validate dynamics, numerical trajectories, AGOP/refit criteria, or population accuracy.',
                           'Two reused development seeds are not fresh independent confirmation.',
                           'Only linked, hashed saved arrays are assessed; missing/unlinked/ambiguous artifacts stay unresolved.',
                           'Only initial parameter slices and full time/index metadata are checked for finiteness; later parameter slices are not evaluated.'])
    try:
        manifest = inputs.json(manifest_path)
        jobs = manifest['configs']
        require(manifest.get('total_arms') == len(jobs) == 6, 'expected all six planned arms')
        ids = [job['id'] for job in jobs]
        require(len(set(ids)) == 6 and all(Path(x).name == x and x not in ('.', '..') for x in ids), 'invalid or duplicate arm IDs')
        expected = {(seed, q) for seed in SEEDS for q in QS}
        require({(job['seed'], job['q']) for job in jobs} == expected, 'planned seed/q grid differs')
        states = {}
        for job in jobs:
            cfg = job['config'][0]['args']
            require(job['engine'] == 'swiglu' and cfg['link'] == 'h3' and cfg['seed'] == job['seed']
                    and cfg['head_ratio'] == job['q'] and cfg['d'] == 64 and cfg['m'] == 64
                    and cfg['s'] == 0.3 and len(cfg['c']) == 8, 'unexpected frozen initialization design')
            try:
                arm, arrays = load_initial(job, Path(execution_path).resolve(), inputs)
                states[(job['seed'], job['q'])] = arrays
            except FileNotFoundError as exc:
                arm = dict(id=job['id'], seed=job['seed'], q=job['q'], status='missing', reason=str(exc))
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
                arm = dict(id=job['id'], seed=job['seed'], q=job['q'], status='unresolved', reason=str(exc))
            report['arms'].append(arm)
        for seed in SEEDS:
            for q in QS[1:]:
                pair = dict(seed=seed, q=q, reference_q=1.0, status='unresolved', **{'pass': False})
                if (seed, 1.0) in states and (seed, q) in states:
                    P1, V1, _, a1 = states[seed, 1.0]
                    Pq, Vq, aq_bytes, aq = states[seed, q]
                    expected_head = [q * x for x in a1]
                    normalized = [x / q for x in aq]
                    differences = [abs(x - y) for x, y in zip(normalized, a1)]
                    exact_forward = aq_bytes == np.asarray(expected_head, dtype='<f8').tobytes()
                    within_tolerance = all(diff <= HEAD_RTOL * abs(ref) for diff, ref in zip(differences, a1))
                    pair.update(P_exact=Pq == P1, V_exact=Vq == V1, raw_head_equals_q_times_reference=exact_forward,
                                normalized_head_exact=normalized == a1, normalized_head_within_tolerance=within_tolerance,
                                normalized_head_max_abs_difference=max(differences),
                                normalized_head_max_relative_difference=max((d / abs(r) if r else 0.0 if d == 0 else math.inf) for d, r in zip(differences, a1)))
                    # Avoid non-standard JSON Infinity even in a mismatched zero reference.
                    if not math.isfinite(pair['normalized_head_max_relative_difference']):
                        pair['normalized_head_max_relative_difference'] = None
                    pair['pass'] = Pq == P1 and Vq == V1 and exact_forward and within_tolerance
                    pair['status'] = 'matched' if pair['pass'] else 'mismatch'
                else:
                    pair['reason'] = 'reference or scaled arm lacks verified initial arrays'
                report['comparisons'].append(pair)
        report['pass'] = len(states) == 6 and all(pair['pass'] for pair in report['comparisons'])
        report['status'] = 'pass' if report['pass'] else 'mismatch' if any(p['status'] == 'mismatch' for p in report['comparisons']) else 'unresolved'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['issues'].append(str(exc))
    changed = inputs.changed()
    report.update(input_sha256=inputs.hashes, missing_input_paths=sorted(inputs.missing),
                  inputs_unchanged=not changed, changed_input_paths=changed,
                  planned_arms=6, assessed_arms=sum(a['status'] == 'ready' for a in report['arms']))
    if changed:
        report.update(status='input_changed', **{'pass': False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != '.json' or not output.is_relative_to(HERE) or output.exists():
        parser.error('--output must be a fresh JSON file inside headscale/analysis')
    result = audit(args.manifest, args.execution)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'pass': result['pass'], 'output': str(output)}))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
