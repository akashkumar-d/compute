"""Saved-state/control validation; no model evaluation or scientific import."""
import copy
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover_control(raw, cfg):
    for key, value in cfg.items():
        if key not in ('out', 'verbose') and raw['cfg'].get(key) != value:
            raise ValueError('Scientific configuration changed: ' + key)
    history, losses, rows = raw['step_history'], raw['Lhist'], raw['rows']
    terminal = raw['termination']
    if terminal['reason'] != 'external_stop' or not history or len(history) != len(losses):
        raise ValueError('Require a complete externally stopped segment')
    if history[-1]['step'] != terminal['step'] or rows[-1]['step'] != terminal['step']:
        raise ValueError('Final saved snapshot must be the terminal evaluated state')
    if terminal['step'] >= cfg['max_steps'] or terminal['t'] >= cfg['t_max'] or terminal['L'] <= cfg['L_stop']:
        raise ValueError('The original scientific stopping criterion was already reached')
    next_cp, last_dL, last_L = 0., 1e-8, 1.
    low, high, exits, natural_grid = math.inf, -math.inf, {}, []
    for i, h in enumerate(history):
        if h['step'] != i or losses[i] != [h['t'], h['L']]:
            raise ValueError('Loss/step histories must preserve every evaluated update')
        if any(not math.isfinite(h[k]) for k in ('t','L','dt','rel')) or h['dt'] <= 0:
            raise ValueError('Nonfinite or nonpositive saved step')
        if i and h['t'] != history[i-1]['t'] + history[i-1]['dt']:
            raise ValueError('Physical time increments disagree with saved Euler steps')
        t, L = h['t'], h['L']
        low, high = min(low, L), max(high, L)
        for name, ratio in [('ratio_1pct',1.01),('ratio_5pct',1.05)]:
            if name not in exits and (L <= 0 or high > ratio * low):
                exits[name] = dict(ratio=ratio,first_exit_step=i,first_exit_t=t,
                    last_valid_step=i-1,last_valid_t=history[i-1]['t'] if i else None,
                    previous_state_saved=bool(i),crossing_state_saved=True)
        dL = 1. - L
        # Replay the natural checkpoint rule without the artificial external stop.
        if (t >= next_cp or (dL > 1e-8 and dL > cfg['dl_ratio'] * last_dL and L > cfg['L_stop'])
                or abs(L-last_L) > .02):
            natural_grid.append(i)
            last_dL, last_L = max(dL,1e-8), L
            next_cp = max(t*cfg['cp_ratio'], t+cfg['cp_min'])
    for name, exit_info in exits.items():
        saved = raw['ratio_prefixes'][name]
        if any(saved.get(k) != v for k,v in exit_info.items()):
            raise ValueError('Saved loss-prefix boundary disagrees with full history')
    for row in rows:
        h = history[row['step']]
        if any(row[k] != h[k] for k in ('t','L','dt')):
            raise ValueError('Snapshot metadata disagrees with all-update history')
    if any(terminal[k] != history[-1][k] for k in ('step','t','L')):
        raise ValueError('Terminal metadata disagrees with final evaluated state')
    return dict(next_cp=next_cp,last_dL=last_dL,last_L=last_L,
                loss_low=low,loss_high=high,prefix_exits=exits,
                natural_grid_steps=natural_grid,
                forced_terminal_extra=terminal['step'] not in natural_grid)


def load_checkpoint(raw_path, snaps_path, cfg):
    import numpy as np
    raw = json.loads(Path(raw_path).read_text())
    control = recover_control(raw, cfg)
    with np.load(snaps_path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in ('t','P','V','a')}
    count, d, m = len(raw['rows']), cfg['d'], cfg['m']
    expected = {'t':(count,), 'P':(count,d+1,m), 'V':(count,d+1,m), 'a':(count,m)}
    for k, shape in expected.items():
        if arrays[k].shape != shape or arrays[k].dtype != np.dtype('float64') or not np.isfinite(arrays[k]).all():
            raise ValueError('Invalid saved array shape/dtype/values: ' + k)
    if arrays['t'].tolist() != [row['t'] for row in raw['rows']]:
        raise ValueError('Snapshot times must match all saved checkpoint rows exactly')
    snaps = [(float(arrays['t'][i]), arrays['P'][i], arrays['V'][i], arrays['a'][i]) for i in range(count)]
    return dict(raw=raw,snaps=snaps,control=control,
        inputs=dict(raw_sha256=sha(raw_path),snapshots_sha256=sha(snaps_path)),
        initial_array_sha256={k:hashlib.sha256(arrays[k][0].tobytes()).hexdigest() for k in ('P','V','a')},
        final_array_sha256={k:hashlib.sha256(arrays[k][-1].tobytes()).hexdigest() for k in ('P','V','a')})


def continuation_provenance(checkpoint, termination):
    prior = checkpoint['raw']
    return dict(kind='exact_state_continuation_of_interrupted_validation',
        independent_seed=False,original_result_completed=False,
        original_termination=copy.deepcopy(prior['termination']),
        input_sha256=checkpoint['inputs'],initial_array_sha256=checkpoint['initial_array_sha256'],
        resumed_array_sha256=checkpoint['final_array_sha256'],
        anchor_re_evaluation='exact equality required for loss, adaptive relative rate and dt',
        global_step_cap=prior['cfg']['max_steps'],global_time_horizon=prior['cfg']['t_max'],
        global_loss_stop=prior['cfg']['L_stop'],scheduler_control=checkpoint['control'],
        segments=[dict(kind='preserved_interrupted_segment',first_evaluated_step=0,
                       last_evaluated_step=prior['termination']['step'],termination=prior['termination']),
                  dict(kind='continuation_segment',anchor_step=prior['termination']['step'],
                       first_new_evaluated_step=prior['termination']['step']+1,
                       last_evaluated_step=termination['step'],termination=termination)],
        merged_history_rule='Prior evaluated states preserved; append new states only; no duplicate anchor',
        checkpoint_grid_note='Preserve forced terminal snapshot as an extra; recover natural scheduler before interruption')


def diagnostic_plan(raw, saved_diagnostics, computed, original_order):
    """Keep already computed old metrics; prioritize exact missing saved states."""
    reused, missing = [], []
    for prior in saved_diagnostics['rows']:
        index = prior['index']
        if index >= len(raw['rows']) or raw['rows'][index]['step'] != prior['step']:
            raise ValueError('Saved diagnostic index is not the same preserved checkpoint')
        if prior['status'] == 'ok':
            computed[index].update(copy.deepcopy(prior),diagnostic_segment='preserved_interrupted_segment')
            reused.append(index)
        else:
            missing.append(index)
    required = []
    for step in (358,380):
        hits = [i for i,row in enumerate(raw['rows']) if row['step'] == step]
        if len(hits) != 1:
            raise ValueError('Required exact order4 diagnostic checkpoint is missing')
        if hits[0] not in reused:
            required.append(hits[0])
    order = list(dict.fromkeys(required + missing + original_order))
    order = [i for i in order if i not in reused]
    return order, dict(required_first_steps=[358,380],preserved_success_indices=reused,
                      missing_original_indices=missing,full_compute_order=order,
                      reuse_basis='Pinned input diagnostic JSON and unchanged old snapshots/config/kernel hashes')
