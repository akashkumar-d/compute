"""Write manifest.json for campaign D: the natural-CE-plateau learner across p = 23, 31, 47, 61, 97 (development sweep).

Families (every entry is one training run with a fixed configuration; the worker calls it a "chain" as in campaign C):
  H    one hidden layer, width 512: hidden decay lambda_W dial at every p, 2 seeds;
  HW   one hidden layer at p = 97, widths 1024 and 2048: the same dial on fewer points, 1 seed;
  D3   depth 3, width 256, large-init deep layers: last-hidden-layer rate dial r at p = 23, 47, 61 (2 seeds) and 97 (1 seed);
  D4   depth 4, the same.
The learner is campaign C's (fixed hyperparameters, every layer trained throughout). Runs stop early once held-out accuracy has
stayed >= 0.99 for a while (stop_generalized); otherwise they run to the horizon. Seeds: init 5000 + i, split 6000 + i.

    python3 make_manifest.py            # -> manifest.json
"""
import json, hashlib, math, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'S{i}' for i in range(1, 9)]
CORES = 4
U = 0.03 * 0.015 / 0.07                     # step of a deep layer that shrinks at layer 1's rate (r = 1)
STOP = dict(acc=0.99, factor=1.5, extra=10000)
BASE = dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, lam_v=0.2, lam_w=0.015,
            observers_v1=True, stop_generalized=STOP)

# ---- the grid
PRIMES = [23, 31, 47, 61, 97]
LAM_1HL = {23: [0.005, 0.0075, 0.01, 0.015, 0.02], 31: [0.005, 0.0075, 0.01, 0.015, 0.02], 47: [0.005, 0.0075, 0.01, 0.015, 0.02],
           61: [0.005, 0.0075, 0.01, 0.015, 0.02], 97: [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]}
H_1HL = {23: 40000, 31: 40000, 47: 40000, 61: 50000, 97: 50000}          # horizons (steps); most runs stop earlier
HW = {1024: [0.005, 0.01, 0.015], 2048: [0.0075, 0.015]}                   # p = 97 width scan, seed 1
H_HW = {1024: 50000, 2048: 50000}
DEEP_PRIMES = [23, 47, 61, 97]
R_DIAL = {3: [0.2, 0.3, 0.45, 0.65, 0.9], 4: [0.15, 0.22, 0.32, 0.45, 0.65]}   # last-layer rate relative to layer 1
H_DEEP = {23: 120000, 47: 100000, 61: 100000, 97: 100000}
SEEDS = {'H': [1, 2], 'HW': [1], 'D': {23: [1, 2], 47: [1, 2], 61: [1, 2], 97: [1]}}

# ---- cost model: seconds per step on one thread (measured 2026-09-29 in a 2-core container while two other runs used it,
# so a little pessimistic), including the light observers every 20 steps and the heavy ones every heavy_every steps
SEC = {('H', 23, 512): 0.0065, ('H', 31, 512): 0.0086, ('H', 47, 512): 0.028, ('H', 61, 512): 0.055, ('H', 97, 512): 0.184,
       ('H', 97, 1024): 0.365, ('H', 97, 2048): 0.77,
       (3, 23, 256): 0.025, (3, 47, 256): 0.068, (3, 61, 256): 0.101, (3, 97, 256): 0.306,
       (4, 23, 256): 0.042, (4, 47, 256): 0.087, (4, 61, 256): 0.136, (4, 97, 256): 0.404}
SPEEDUP = {1: 1.0, 2: 1.5, 4: 2.2}           # wall-time speed-up with more threads (measured 1.5x at 2 threads on p = 97)
EARLY = {'H': 0.5, 'HW': 0.5, 'D': 0.7}      # expected fraction of the horizon actually run (stop_generalized). The p = 97
                                             # 1HL pilot (lambda_W 0.015) had test 0.94 at 9k steps, so it would stop near 20k


def threads_for(sec_per_step, steps, early):
    """1 thread, or 4 (a whole session) for a run that would otherwise take over ~4 h of wall time (over 6 h at its full horizon);
    sessions lived about 8 h in campaign C. Only 1 or 4, so that a session's queue never waits on a half-free machine."""
    if sec_per_step * steps * early / 3600 <= 4.0 and sec_per_step * steps / 3600 <= 6.0: return 1
    return 4


def entry(fam, p, depth, width, dial, seed, cfg, sec):
    early = EARLY[fam]
    n = threads_for(sec, cfg['steps'], early)
    wall = sec * cfg['steps'] * early / SPEEDUP[n] / 3600
    return dict(stage=fam if fam in ('H', 'HW') else f'D{depth}', family=fam, cohort=f'{fam}_p{p}' + (f'_w{width}' if fam == 'HW' else ''),
                p=p, depth=depth, width=width, dial=dial, seed=seed, init_seed=5000 + seed, split_seed=6000 + seed, cfg=cfg, threads=n,
                est_wall_hours=round(wall, 2), est_core_hours=round(wall * n, 2))


def build():
    E = []
    for p in PRIMES:
        for lw in LAM_1HL[p]:
            for s in SEEDS['H']:
                cfg = dict(BASE, p=p, depth=1, width=512, lam_w=lw, steps=H_1HL[p], heavy_every=1000 if p == 97 else 500,
                           init_seed=5000 + s, split_seed=6000 + s)
                e = entry('H', p, 1, 512, dict(lam_w=lw), s, cfg, SEC[('H', p, 512)]); e['id'] = f'H_p{p}_lw{lw:.4f}_s{s}'; E.append(e)
    for w, lws in HW.items():
        for lw in lws:
            for s in SEEDS['HW']:
                cfg = dict(BASE, p=97, depth=1, width=w, lam_w=lw, steps=H_HW[w], heavy_every=1000, init_seed=5000 + s, split_seed=6000 + s)
                e = entry('HW', 97, 1, w, dict(lam_w=lw), s, cfg, SEC[('H', 97, w)]); e['id'] = f'HW_p97_w{w}_lw{lw:.4f}_s{s}'; E.append(e)
    for D in (3, 4):
        for p in DEEP_PRIMES:
            for r in R_DIAL[D]:
                for s in SEEDS['D'][p]:
                    cfg = dict(BASE, p=p, depth=D, width=256, deep_init_scale=8.0, lam_deep=0.07, steps=H_DEEP[p],
                               heavy_every=1000 if p == 97 else 500, init_seed=5000 + s, split_seed=6000 + s,
                               eta_layers=[0.03] + [U] * (D - 2) + [r * U], lam_layers=[0.015] + [0.07] * (D - 1))
                    e = entry('D', p, D, 256, dict(r=r), s, cfg, SEC[(D, p, 256)]); e['id'] = f'D{D}_p{p}_r{r:.2f}_s{s}'; E.append(e)
    return E


def assign(E):
    """Longest-first greedy on each session's load (core-hours / cores); inside a session, long runs first, then by priority."""
    load = {w: 0.0 for w in WORKERS}
    for e in sorted(E, key=lambda e: (-e['est_wall_hours'] * e['threads'], e['id'])):
        w = min(WORKERS, key=lambda w: (load[w], w))
        e['worker'] = w; load[w] += e['est_core_hours'] / CORES
    for w in WORKERS:                        # whole-session (4-thread) runs first, so they never wait for cores to drain
        q = sorted([e for e in E if e['worker'] == w], key=lambda e: (-e['threads'], -e['est_wall_hours'], e['id']))
        for i, e in enumerate(q): e['order'] = i
    return load


def main():
    E = build(); load = assign(E)
    ids = [e['id'] for e in E]; assert len(ids) == len(set(ids)), 'duplicate ids'
    man = dict(campaign='D', version=1, created=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               author='claude/plateau-review (session_011vNRtJNdeKePn3Gex5aRg7)', kind='development sweep (not preregistered)',
               workers=WORKERS, cores_per_worker=CORES, stop_generalized=STOP, seed_rule='init 5000 + i, split 6000 + i',
               grid=dict(primes=PRIMES, lam_1hl={str(k): v for k, v in LAM_1HL.items()}, horizon_1hl={str(k): v for k, v in H_1HL.items()},
                         width_scan_p97={str(k): v for k, v in HW.items()}, deep_primes=DEEP_PRIMES, r_dial={str(k): v for k, v in R_DIAL.items()},
                         horizon_deep={str(k): v for k, v in H_DEEP.items()}),
               est_session_hours={w: round(v, 2) for w, v in load.items()}, chains=E)
    fn = os.path.join(HERE, 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    tot = sum(e['est_core_hours'] for e in E)
    print(f"{len(E)} runs, {tot:.0f} core-hours at the estimate; per session: " + ', '.join(f'{w} {v:.1f} h' for w, v in load.items()))
    by = {}
    for e in E: by.setdefault(e['stage'], [0, 0.0]); by[e['stage']][0] += 1; by[e['stage']][1] += e['est_core_hours']
    print('by family: ' + ', '.join(f'{k} {v[0]} runs / {v[1]:.0f} core-h' for k, v in by.items()))
    print('threads: ' + ', '.join(f'{n}: {sum(1 for e in E if e["threads"] == n)}' for n in (1, 2, 4)))
    print('sha256', hashlib.sha256(open(fn, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
