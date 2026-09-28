"""Read-only diagnostic reuse/planning. Standard library; no scientific imports."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def verify_inputs(spec, input_root):
    root = Path(input_root).resolve(strict=True)
    receipt = {}
    for name, expected in spec['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Input must be a plain file name')
        path = (root / name).resolve(strict=True)
        if path.parent != root or not path.is_file():
            raise ValueError('Pinned input path escaped case directory')
        actual = dict(size=path.stat().st_size, sha256=sha256(path))
        if actual != expected: raise ValueError(f'Pinned input changed: {name}')
        receipt[name] = actual
    return receipt


def ratio_prefix(raw, ratio):
    lo, hi = math.inf, -math.inf
    first_bad = None
    for i, pair in enumerate(raw['Lhist']):
        t, loss = pair
        if not math.isfinite(t) or (i and t < raw['Lhist'][i-1][0]):
            raise ValueError('Invalid all-update clock')
        lo, hi = min(lo, loss), max(hi, loss)
        if not math.isfinite(loss) or loss <= 0 or hi > ratio * lo:
            first_bad = i
            break
    end = len(raw['Lhist']) - 1 if first_bad is None else first_bad - 1
    checkpoint = max((i for i, row in enumerate(raw['rows']) if row['step'] <= end), default=None)
    crossing = next((i for i, row in enumerate(raw['rows']) if row['step'] == first_bad), None)
    return dict(ratio=ratio,last_admissible_update=end,checkpoint_index=checkpoint,
                first_crossing_update=first_bad,crossing_checkpoint_index=crossing,
                right_censored=first_bad is None)


def make_plan(raw, result, partial, spec):
    """Preserve successful rows verbatim; prioritize loss-only prefix membership."""
    if result['tag'] != spec['tag'] or partial['tag'] != spec['tag']:
        raise ValueError('Input tag mismatch')
    if result['cfg'] != spec['cfg'] or {k:v for k,v in raw['cfg'].items()
            if k not in ('out','verbose')} != spec['cfg']:
        raise ValueError('Pinned scientific configuration mismatch')
    if result['source_files_sha256'] != spec['source_files_sha256']:
        raise ValueError('Original diagnostic kernel hash mismatch')
    if result['source_manifest_sha256'] != spec['source_manifest_sha256']:
        raise ValueError('Original frozen source manifest mismatch')
    if result['termination'] != spec['termination'] or raw['termination'] != spec['termination']:
        raise ValueError('Original training termination mismatch')
    if result['diagnostics'] != partial['rows']:
        raise ValueError('Original diagnostic sidecar and result disagree')
    rows = raw['rows']
    computed = copy.deepcopy(result['diagnostics'])
    if len(rows) != spec['expected_checkpoint_count'] or len(computed) != len(rows):
        raise ValueError('Snapshot grid length mismatch')
    if len(raw['Lhist']) != raw['termination']['step'] + 1:
        raise ValueError('All-update history length mismatch')
    reused = []
    for i, (row, diagnostic) in enumerate(zip(rows, computed)):
        if diagnostic['index'] != i or diagnostic['step'] != row['step'] or diagnostic['t'] != row['t']:
            raise ValueError('Diagnostic and saved checkpoint mismatch')
        if row['step'] < 0 or row['step'] >= len(raw['Lhist']):
            raise ValueError('Invalid checkpoint update index')
        if (row['t'],row['L']) != tuple(raw['Lhist'][row['step']]):
            raise ValueError('Checkpoint disagrees with all-update history')
        if i and row['step'] <= rows[i-1]['step']:
            raise ValueError('Checkpoint update grid must be strictly ascending')
        if diagnostic['status'] == 'ok': reused.append(i)
    if reused != spec['expected_success_indices']:
        raise ValueError('Original successful diagnostic set changed')
    prefixes = {str(r):ratio_prefix(raw,r) for r in (1.01,1.05)}
    if prefixes != spec['original_ratio_prefixes'] or prefixes != result['ratio_prefixes']:
        raise ValueError('Initial all-update ratio window mismatch')
    missing = [i for i in range(len(rows)) if i not in reused]
    first = [i for i in missing if rows[i]['step'] <= prefixes['1.01']['last_admissible_update']]
    second = [i for i in missing if rows[i]['step'] <= prefixes['1.05']['last_admissible_update'] and i not in first]
    remaining = [i for i in missing if i not in first and i not in second]
    order = first + second + remaining
    origins = {str(i):'original_interrupted_diagnostic_process' for i in reused}
    return computed, dict(checkpoint_count=len(rows),preserved_success_indices=reused,
        missing_original_indices=missing,initial_1pct_missing_indices=first,
        additional_initial_5pct_missing_indices=second,remaining_missing_indices=remaining,
        full_compute_order=order,compute_steps=[rows[i]['step'] for i in order],
        ratio_prefixes=prefixes,priority_basis='original all-update initial max/min loss only',
        grid_changed=False,new_training_updates=0,origins=origins)


def complete_rows(computed, plan, evaluate, persist, should_stop):
    """Bounded, interruption-safe core; backend injection permits purely mock tests."""
    attempted = []
    origins = copy.deepcopy(plan['origins'])
    persist(computed,origins,attempted)
    for index in plan['full_compute_order']:
        if should_stop(): break
        attempted.append(index)
        try:
            metrics = evaluate(index)
            if any(k in metrics for k in ('index','step','t','status')):
                raise ValueError('Metrics cannot overwrite diagnostic identity')
            computed[index].update(metrics)
            computed[index]['status'] = 'ok'
            computed[index].pop('error',None)
            origins[str(index)] = 'diagnostics_only_completion_pass'
        except Exception as exc:
            computed[index].update(status='failed',error=f'{type(exc).__name__}: {exc}')
            origins[str(index)] = 'diagnostics_only_completion_pass_failed'
        persist(computed,origins,attempted)
    return computed,origins,attempted
