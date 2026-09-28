"""Small-initialization SwiGLU flow with a declared head-rate multiplier.

Population expectations use finite-order deterministic Gaussian quadrature.
All blocks train; the head uses a separately declared positive learning-rate factor:
    p <- p - dt*m*grad_p L, v <- v - dt*m*grad_v L,
    a <- a - dt*head_lr*m*grad_a L.
head_lr=1 recovers the original common-rate arithmetic. The scaled head direction
also enters the relative per-neuron update cap. This is block-specific flow when
head_lr != 1, not common-rate gradient descent.
dt adaptive (relative per-neuron change <= h per step, dt <= dt_max); the physical time
t = sum dt is recorded.  Init: p_j, v_j ~ N(0, s^2 I_{d+1}/(d+1)), a_j ~ N(0, s^2) (balanced in expectation; head variance
independent of m).  Metrics: loss/V, AGOP top-r alignment (A = mean cos^2, A_min = min cos^2, eigengap), A_top = |U^T e|^2
for the top AGOP eigenvector e, weight mass in U,
per-neuron alignment of p and v, norm-weighted alignment, raw-feature refit, norms, balancedness drift.
"""
import argparse, json, math, sys, time
import numpy as np
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_here,):  # bundled local imports only
    if _os.path.exists(_os.path.join(_p, 'vlab.py')) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import swpop, vlab


_STOP_REQUESTED = False
_LAST_PARTIAL_WALL = 0.0

def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True

def _write_partial(cfg, rows, Lhist, step_history, snaps, teacher, force=False):
    """Atomic durable checkpoint sidecars, without altering any numerical state."""
    global _LAST_PARTIAL_WALL
    if not cfg.get('out'):
        return
    last = step_history[-1]
    force = force or not np.isfinite(last['dt']) or last['dt'] <= 0 or last['t'] + last['dt'] == last['t']
    if not force and time.time() - _LAST_PARTIAL_WALL < 15.0:
        return
    import hashlib
    root = _os.path.join(_here, '..', 'partials')
    _os.makedirs(root, exist_ok=True)
    tag = _os.path.basename(cfg['out']).removesuffix('.json')
    target = _os.path.join(root, tag+'.json')
    data = dict(cfg=cfg, rows=rows, Lhist=Lhist, step_history=step_history,
                gamma=teacher.gamma, EY=teacher.EY, V=teacher.V, partial=True,
                recording_engine_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                pid=_os.getpid())
    with open(target+'.tmp', 'w') as fh:
        json.dump(data, fh)
    _os.replace(target+'.tmp', target)
    sn_target = _os.path.join(root, tag+'_snaps.npz')
    np.savez_compressed(sn_target+'.tmp.npz', t=np.array([x[0] for x in snaps]),
                        P=np.array([x[1] for x in snaps]), V=np.array([x[2] for x in snaps]),
                        a=np.array([x[3] for x in snaps]))
    _os.replace(sn_target+'.tmp.npz', sn_target)
    _LAST_PARTIAL_WALL = time.time()

# Breadth14 uses the same fixed scalar teacher definitions for training and diagnostics.
# This replaces only link parsing/normalization metadata, not any update direction.
from canonical_links import make_link

def alignment_stats(P, V, a, d, r):
    pi, nu = P[:d], V[:d]
    out = {}
    for tag, W in (("p", pi), ("v", nu)):
        n2 = (W ** 2).sum(0); u2 = (W[:r] ** 2).sum(0); eU = u2 / np.maximum(n2, 1e-300)
        out[tag + "_massU"] = float(u2.sum() / n2.sum())
        out[tag + "_eU50"] = float(np.median(eU)); out[tag + "_eU_mean"] = float(eU.mean())
        out[tag + "_frac_half"] = float((eU > 0.5).mean())
    # output-scale weights |a| |pi|^2 |nu|  (the size of the neuron's degree-3 term)
    w = np.abs(a) * (pi ** 2).sum(0) * np.sqrt((nu ** 2).sum(0))
    eUp = (pi[:r] ** 2).sum(0) / np.maximum((pi ** 2).sum(0), 1e-300)
    out["p_eU_wout"] = float((w * eUp).sum() / w.sum())
    th2 = (P ** 2).sum(0) + (V ** 2).sum(0) + a ** 2
    out["norm_med"] = float(np.sqrt(np.median(th2))); out["norm_max"] = float(np.sqrt(th2.max()))
    out["winner"] = int(np.argmax(th2)); out["winner_eUp"] = float(eUp[out["winner"]])
    # which teacher axis each neuron's p is closest to
    arg = np.abs(pi[:r]).argmax(0); spec = np.abs(pi[:r]).max(0) / np.sqrt(np.maximum((pi ** 2).sum(0), 1e-300))
    out["spec09_counts"] = [int(((arg == i) & (spec > 0.9)).sum()) for i in range(r)]
    return out

def validate_head_lr(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('head_lr must be a positive finite number')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('head_lr must be a positive finite number')
    return value

def update_directions(ev, m, head_lr=1.0):
    head_lr = validate_head_lr(head_lr)
    GP, GV, Ga = m * ev['gP'], m * ev['gV'], m * ev['ga']
    if head_lr != 1.0:
        Ga = head_lr * Ga
    return GP, GV, Ga

def first_ratio_crossing(low, high, loss, already_crossed, ratio):
    """Loss-only first-exit rule; has no feedback into parameter updates."""
    return not already_crossed and (not np.isfinite(loss) or loss <= 0 or high > ratio * low)

def run(cfg):
    head_lr = validate_head_lr(cfg.get('head_lr', 1.0))
    d, r, m = cfg["d"], len(cfg["c"]), cfg["m"]
    f, brk = make_link(cfg["link"])
    T = swpop.Teacher(f, np.asarray(cfg["c"], float), breaks=brk, n_x=cfg["n_x"])
    E = swpop.SwiGLUPop(d, T, alpha=cfg["alpha"], intercept="refit", bias=True,
                        n_pair=cfg["n_pair"], n_diag=cfg["n_diag"], n_z=cfg["n_z"])
    rng = np.random.default_rng(cfg["seed"])
    s = cfg["s"]
    P = rng.standard_normal((d + 1, m)) * s / math.sqrt(d + 1)
    V = rng.standard_normal((d + 1, m)) * s / math.sqrt(d + 1)
    a = rng.standard_normal(m) * s * cfg["head_ratio"]
    bal0 = a ** 2 - (V ** 2).sum(0)
    weighted_bal0 = a ** 2 / head_lr - (V ** 2).sum(0)
    t = 0.0; step = 0; rows = []; Lhist = []; step_history = []; snaps = []
    next_cp = 0.0; t0 = time.time(); last_dL = 1e-8; last_L = 1.0
    prefix_exits = {}; loss_low = float('inf'); loss_high = -float('inf'); previous = None
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
        Lhist.append((t, L))
        GP, GV, Ga = update_directions(ev, m, head_lr)
        gn = np.sqrt((GP ** 2).sum(0) + (GV ** 2).sum(0) + Ga ** 2)
        th = np.sqrt((P ** 2).sum(0) + (V ** 2).sum(0) + a ** 2)
        rel = float((gn / np.maximum(th, 1e-300)).max())
        dt = min(cfg["dt_max"], cfg["h"] / max(rel, 1e-300))
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

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16); ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--c", type=str, default="1.0"); ap.add_argument("--link", type=str, default="he3")
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--s", type=float, default=0.1)
    ap.add_argument("--head_ratio", type=float, default=1.0)
    ap.add_argument('--head_lr', type=float, default=1.0, help='Positive head-block learning-rate multiplier; P,V rates unchanged')
    ap.add_argument("--h", type=float, default=0.01); ap.add_argument("--dt_max", type=float, default=50.0)
    ap.add_argument("--t_max", type=float, default=1e6); ap.add_argument("--L_stop", type=float, default=0.3)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--cp_ratio", type=float, default=1.15); ap.add_argument("--cp_min", type=float, default=1.0)
    ap.add_argument("--dl_ratio", type=float, default=2.0)
    ap.add_argument("--n_pair", type=int, default=16); ap.add_argument("--n_diag", type=int, default=48)
    ap.add_argument("--n_z", type=int, default=24); ap.add_argument("--n_x", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=str, default="")
    ap.add_argument("--quiet", action="store_true")
    A = ap.parse_args(argv)
    # Explicit prelaunch runtime amendment: replace queued old CLI horizon by
    # re-execing the SAME pinned engine with a recorded effective argv.
    if argv is None and A.out:
        override_path = _os.path.join(_here, '..', 'configs', 'runtime_overrides.json')
        if _os.path.exists(override_path):
            tag = _os.path.basename(A.out).removesuffix('.json')
            override = json.load(open(override_path)).get(tag, {})
            changed = {k: v for k, v in override.items() if getattr(A, k) != v}
            if changed:
                import datetime, hashlib
                new_argv = [sys.executable] + list(sys.argv)
                for key, value in changed.items():
                    new_argv[new_argv.index('--'+key) + 1] = str(value)
                event = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event='engine_exec_override', tag=tag,
                             pid=_os.getpid(), original_command=[sys.executable]+list(sys.argv), effective_command=new_argv,
                             reason='documented runtime budget amendment, no step size, seed, initialization or gradient changes', effective_overrides=changed,
                             engine_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest())
                with open(_os.path.join(_here, '..', 'process_overrides.jsonl'), 'a') as fh:
                    fh.write(json.dumps(event)+'\n')
                print(json.dumps(event), flush=True)
                _os.execv(sys.executable, new_argv)
    cfg = vars(A).copy(); cfg["c"] = [float(x) for x in A.c.split(",")]; cfg["verbose"] = not A.quiet
    import signal
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    out = run(cfg)
    import datetime, hashlib
    reproduction = dict(engine_version='recording-v4-periodic-partials-graceful-stop',
                        engine_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                        effective_command=[sys.executable]+list(sys.argv),
                        raw_launch_command=[sys.executable]+list(sys.argv), override_reason=None,
                        pid=_os.getpid())
    override_events = _os.path.join(_here, '..', 'process_overrides.jsonl')
    if _os.path.exists(override_events):
        for line in open(override_events):
            event = json.loads(line)
            if event.get('pid') == _os.getpid():
                reproduction.update(raw_launch_command=event['original_command'],
                                    override_reason=event['reason'], override_event_utc=event['utc'])
    out['reproduction'] = reproduction
    snaps = out.pop("_snaps")
    if A.out:
        with open(A.out, "w") as fh:
            json.dump(out, fh)
        np.savez_compressed(A.out.replace(".json", "_snaps.npz"), t=np.array([x[0] for x in snaps]),
                            P=np.array([x[1] for x in snaps]), V=np.array([x[2] for x in snaps]),
                            a=np.array([x[3] for x in snaps]))

if __name__ == "__main__":
    main()
