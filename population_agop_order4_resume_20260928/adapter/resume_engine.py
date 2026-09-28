"""Resume-only adapter derived from the pinned run loop; kernels remain frozen.

Only initialization/control restoration and anchor deduplication differ. The
original arithmetic/checkpoint body is retained verbatim after bootstrap.
"""
import copy
import math
import sys
import time
import numpy as np
import swsmall_periodic as frozen
from swsmall_periodic import (validate_head_lr, update_directions,
    first_ratio_crossing, alignment_stats, make_link, swpop, vlab)
_STOP_REQUESTED = False
_write_partial = frozen._write_partial

def run(cfg, checkpoint_state):
    head_lr = validate_head_lr(cfg.get('head_lr', 1.0))
    d, r, m = cfg["d"], len(cfg["c"]), cfg["m"]
    f, brk = make_link(cfg["link"])
    T = swpop.Teacher(f, np.asarray(cfg["c"], float), breaks=brk, n_x=cfg["n_x"])
    E = swpop.SwiGLUPop(d, T, alpha=cfg["alpha"], intercept="refit", bias=True,
                        n_pair=cfg["n_pair"], n_diag=cfg["n_diag"], n_z=cfg["n_z"])
    prior = checkpoint_state['raw']
    control = checkpoint_state['control']
    snaps = [(x[0],x[1].copy(),x[2].copy(),x[3].copy()) for x in checkpoint_state['snaps']]
    P, V, a = [x.copy() for x in snaps[-1][1:]]
    initial_P, initial_V, initial_a = snaps[0][1:]
    s = cfg['s']
    bal0 = initial_a ** 2 - (initial_V ** 2).sum(0)
    weighted_bal0 = initial_a ** 2 / head_lr - (initial_V ** 2).sum(0)
    t, step = prior['termination']['t'], prior['termination']['step']
    rows = copy.deepcopy(prior['rows']); Lhist = copy.deepcopy(prior['Lhist'])
    step_history = copy.deepcopy(prior['step_history'])
    next_cp = control['next_cp']; t0 = time.time()
    last_dL, last_L = control['last_dL'], control['last_L']
    prefix_exits = copy.deepcopy(control['prefix_exits'])
    loss_low, loss_high, previous = control['loss_low'], control['loss_high'], None
    anchor_pending = True
    if any(getattr(T,k) != prior[k] for k in ('gamma','EY','V')):
        raise ValueError('Teacher normalization differs from the interrupted segment')
    def checkpoint(P, V, a, ev, step, t, L, dt, role):
        if rows and rows[-1]['step'] == step:
            rows[-1].setdefault('checkpoint_roles', []).append(role)
            return rows[-1]
        M = E.agop(P, V, a, ev)
        al = vlab.top_r_alignment(M, r)
        e_top = np.linalg.eigh(0.5 * (M + M.T))[1][:, -1]
        row = dict(step=step, t=t, L=L, A=al['A'], Amin=al['cos2_min'], Atop=float((e_top[:r] ** 2).sum()),
            gap=al['gap'], agop_tr=float(np.trace(M)), refit=E.refit(ev) / T.V, dt=dt,
            bal_drift=float(np.abs(a ** 2 - (V ** 2).sum(0) - bal0).max() / max(s * s, 1e-300)),
            weighted_bal_drift=float(np.abs(a ** 2 / head_lr - (V ** 2).sum(0) - weighted_bal0).max() / max(s * s, 1e-300)),
            balance_invariant='a^2/head_lr - sum(V^2), including value bias',
            legacy_balance_note='bal_drift records a^2-sum(V^2); it is not an invariant when head_lr differs from1',
            head_lr=head_lr, checkpoint_roles=[role])
        row.update(alignment_stats(P, V, a, d, r))
        rows.append(row); snaps.append((t, P.copy(), V.copy(), a.copy()))
        return row
    while True:
        ev = E.evaluate(P, V, a, need_grad=True)
        L = ev["L"] / T.V
        if not anchor_pending:
            Lhist.append((t, L))
        GP, GV, Ga = update_directions(ev, m, head_lr)
        gn = np.sqrt((GP ** 2).sum(0) + (GV ** 2).sum(0) + Ga ** 2)
        th = np.sqrt((P ** 2).sum(0) + (V ** 2).sum(0) + a ** 2)
        rel = float((gn / np.maximum(th, 1e-300)).max())
        dt = min(cfg["dt_max"], cfg["h"] / max(rel, 1e-300))
        if anchor_pending:
            anchor = prior['step_history'][-1]
            if L != anchor['L'] or dt != anchor['dt'] or rel != anchor['rel']:
                raise ValueError('Anchor reevaluation differs: require exact saved loss/dt/rel before advancing')
            if _STOP_REQUESTED:
                break
            previous = (P, V, a, ev, step, t, L, dt)
            P = P - dt * GP; V = V - dt * GV; a = a - dt * Ga
            t += dt; step += 1
            anchor_pending = False
            continue
        step_history.append(dict(step=step, t=t, L=L, dt=dt, rel=rel))
        loss_low = min(loss_low, L); loss_high = max(loss_high, L)
        crossed_names = []
        for name, ratio in [('ratio_1pct', 1.01), ('ratio_5pct', 1.05)]:
            if first_ratio_crossing(loss_low, loss_high, L, name in prefix_exits, ratio):
                prefix_exits[name] = dict(ratio=ratio, first_exit_step=step, first_exit_t=t,
                    last_valid_step=step-1, last_valid_t=previous[5] if previous is not None else None,
                    previous_state_saved=previous is not None, crossing_state_saved=True)
                crossed_names.append(name)
                if previous is not None:
                    checkpoint(*previous, role=name+'_last_valid')
        if crossed_names:
            checkpoint(P, V, a, ev, step, t, L, dt, role='+'.join(crossed_names)+'_first_crossing')
            _write_partial(cfg, rows, Lhist, step_history, snaps, T, force=True)
        stop = _STOP_REQUESTED or (t >= cfg["t_max"]) or (L <= cfg["L_stop"]) or (step >= cfg["max_steps"]) or not np.isfinite(L)
        dL = 1.0 - L
        if (t >= next_cp or stop or (dL > 1e-8 and dL > cfg["dl_ratio"] * last_dL and L > cfg["L_stop"])
                or abs(L - last_L) > 0.02):
            last_dL = max(dL, 1e-8); last_L = L
            row = checkpoint(P, V, a, ev, step, t, L, dt, role='original_grid_or_stop')
            _write_partial(cfg, rows, Lhist, step_history, snaps, T, force=stop)
            if cfg["verbose"]:
                print("t=%9.1f st=%6d L=%.6f A=%.3f Amin=%.3f gap=%7.3g refit=%.4f pU=%.3f vU=%.3f p50=%.3f pw=%.3f "
                      "nmax=%.3g nmed=%.3g spec=%s dt=%.3g [%.0fs]" % (
                          t, step, L, row['A'], row['Amin'], row['gap'], row["refit"], row["p_massU"], row["v_massU"],
                          row["p_eU50"], row["p_eU_wout"], row["norm_max"], row["norm_med"], row["spec09_counts"], dt,
                          time.time() - t0))
                sys.stdout.flush()
            next_cp = max(t * cfg["cp_ratio"], t + cfg["cp_min"])
        if stop:
            break
        previous = (P, V, a, ev, step, t, L, dt)
        P = P - dt * GP; V = V - dt * GV; a = a - dt * Ga
        t += dt; step += 1
    out = dict(cfg=cfg, rows=rows, Lhist=Lhist, step_history=step_history, gamma=T.gamma, EY=T.EY, V=T.V)
    out['head_lr'] = head_lr
    out['dynamics'] = 'block-specific adaptive Euler population flow; finite-order quadrature; all P,V,a train'
    out['balance_invariant'] = 'a^2/head_lr - sum(V^2), including value bias'
    out['ratio_prefixes'] = {name: prefix_exits.get(name, dict(ratio=ratio, first_exit_step=None,
        first_exit_t=None, last_valid_step=step, last_valid_t=t, previous_state_saved=False,
        crossing_state_saved=False, right_censored=True)) for name, ratio in [('ratio_1pct', 1.01), ('ratio_5pct', 1.05)]}
    for value in out['ratio_prefixes'].values(): value.setdefault('right_censored', False)
    out["termination"] = dict(step=step, t=t, L=L, wall_seconds=time.time()-t0, reason=("external_stop" if _STOP_REQUESTED else "nonfinite_loss" if not np.isfinite(L) else "loss_stop" if L <= cfg["L_stop"] else "max_steps" if step >= cfg["max_steps"] else "t_max"))
    out["_snaps"] = snaps
    return out
