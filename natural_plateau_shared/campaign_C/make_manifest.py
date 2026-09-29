"""Write manifest.json for campaign C: every chain (pilot -> recipe -> main run), its configuration and the worker that runs it.

    python3 make_manifest.py            # writes manifest.json next to this file and prints the per-worker load

A chain is one seed of one cohort. Its pilot runs at the cohort's fixed last-layer rate r0: a short pilot stops at the first logged
step with held-out accuracy >= 0.1 (it only measures t10); a full pilot runs to the horizon and is also the fixed-rate arm of the
paired comparison (stage C2). The main run's last-layer rate comes from recipe.solve_rate(t10, r0, x_target). Nothing here depends
on any campaign result; the file is written once, hashed and registered before any run starts.
"""
import json, math, os, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ETA_UNIT = 0.03 * 0.015 / 0.07           # deep-layer step (decay 0.07) that shrinks at layer 1's rate; same constant as recipe.py

BASE = dict(frac=0.8, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, heavy_every=500, lam_v=0.2, lam_w=0.015,
            deep_init_scale=8.0, lam_deep=0.07, width=256, p=31, observers_v1=True)

# cohort: depth, width, p, hidden optimizer, pilot rate r0, target x*, horizon, BLAS threads, per-step cost (s, one chain alone on
# a core, measured in this container 2026-09-29 00:25 UTC with the v1 observers on), guessed pilot t10 (for load balancing only)
COHORTS = {
    'D3':     dict(depth=3, width=256, p=31, mode='polar', r0=0.50, x=3.85, T=80000,  threads=1, sps=0.0313, t10=18500),
    'D4':     dict(depth=4, width=256, p=31, mode='polar', r0=0.35, x=3.85, T=100000, threads=1, sps=0.0464, t10=22000),
    'D5':     dict(depth=5, width=256, p=31, mode='polar', r0=0.35, x=3.85, T=80000,  threads=1, sps=0.0611, t10=29000),
    'D4p23':  dict(depth=4, width=256, p=23, mode='polar', r0=0.35, x=3.85, T=120000, threads=1, sps=0.0347, t10=30000),
    'D4w512': dict(depth=4, width=512, p=31, mode='polar', r0=0.40, x=4.50, T=100000, threads=2, sps=0.139,  t10=27000),
    'D3ngd':  dict(depth=3, width=256, p=31, mode='ngd',   r0=0.50, x=3.85, T=80000,  threads=1, sps=0.0175, t10=20000),
}
# stage: cohorts, seed indices, pilot kind, purpose
STAGES = {
    'C1': dict(cohorts=['D3', 'D4', 'D5'], seeds=range(21, 41), pilot='short',
               why='recipe confirmation at depths 3-5 (with C2: 40 fresh seeds per depth, the primary endpoint)'),
    'C2': dict(cohorts=['D3', 'D4', 'D5'], seeds=range(1, 21), pilot='full',
               why='recipe vs one fixed rate, paired on the same seed (the full pilot is the fixed-rate arm)'),
    'C3': dict(cohorts=['D4p23'], seeds=range(41, 61), pilot='short', why='transfer to another modulus (p = 23, depth 4)'),
    'C4': dict(cohorts=['D4w512'], seeds=range(61, 65), pilot='short', why='width 512 at depth 4 (target x* = 4.5)'),
    'C5': dict(cohorts=['D3ngd'], seeds=range(1, 21), pilot='short',
               why='optimizer ablation: Frobenius-normalized gradient for the hidden layers (paired with C2 depth 3, same seeds)'),
}
WORKERS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']     # S1-S4 on account 1, S5-S8 on account 2
CORES = 4
PINNED = {'C4': ['S7', 'S7', 'S8', 'S8']}                      # the long 2-thread chains start first on S7 and S8
CONTENTION = 1.15                                             # slowdown when 4 chains share a 4-core machine (estimate)


def seed_pair(i): return 1000 + i, 2000 + i


def cfg_for(co, i, rate, steps, pilot_short=False):
    c = COHORTS[co]; D = c['depth']; a, b = seed_pair(i)
    cfg = dict(BASE, width=c['width'], p=c['p'], depth=D, steps=steps, init_seed=a, split_seed=b,
               eta_layers=[0.03] + [ETA_UNIT] * (D - 2) + [rate * ETA_UNIT], lam_layers=[0.015] + [0.07] * (D - 1))
    if c['mode'] != 'polar': cfg['hidden_mode'] = c['mode']
    if pilot_short: cfg['stop_at_test'] = 0.1
    return cfg


def chains():
    out = []
    for st, S in STAGES.items():
        for co in S['cohorts']:
            c = COHORTS[co]
            for i in S['seeds']:
                short = S['pilot'] == 'short'
                pilot_steps = c['t10'] if short else c['T']
                wall_h = CONTENTION * c['sps'] * (pilot_steps + c['T']) / 3600
                out.append(dict(
                    id=f'{st}_{co}_s{i:03d}', stage=st, cohort=co, seed=i, init_seed=seed_pair(i)[0], split_seed=seed_pair(i)[1],
                    threads=c['threads'], est_wall_hours=round(wall_h, 3), est_core_hours=round(wall_h * c['threads'], 3),
                    pilot=dict(kind=S['pilot'], rate=c['r0'], cfg=cfg_for(co, i, c['r0'], c['T'], pilot_short=short)),
                    recipe=dict(x_target=c['x'], r0=c['r0'], slope=0.9, K1=0.03 * 0.015, round=0.005, clip=[0.10, 1.20],
                                eta_unit=ETA_UNIT, if_no_t10='use t10 = pilot horizon (censored) and flag the chain'),
                    main=dict(steps=c['T'], cfg_template=cfg_for(co, i, float('nan'), c['T']))))
    return out


def assign(ch):
    load = {w: 0.0 for w in WORKERS}; queue = {w: [] for w in WORKERS}
    rest = []
    for st, ws in PINNED.items():
        pinned = [c for c in ch if c['stage'] == st]
        for c, w in zip(pinned, ws):
            c['worker'] = w; queue[w].append(c); load[w] += c['est_core_hours'] / CORES
    rest = [c for c in ch if 'worker' not in c]
    for c in sorted(rest, key=lambda c: (-c['est_wall_hours'], c['id'])):   # longest processing time first
        w = min(WORKERS, key=lambda w: (load[w], w))
        c['worker'] = w; queue[w].append(c); load[w] += c['est_core_hours'] / CORES
    order = []
    for w in WORKERS:
        for k, c in enumerate(queue[w]): c['order'] = k
        order += queue[w]
    return order, load


def main():
    ch = chains(); ch, load = assign(ch)
    for c in ch:   # json cannot hold NaN: the template's last rate is filled by the recipe
        c['main']['cfg_template']['eta_layers'][-1] = None
    man = dict(campaign='C', version=1, created='2026-09-29', author='claude/plateau-review (session_011vNRtJNdeKePn3Gex5aRg7)',
               workers=WORKERS, cores_per_worker=CORES, stages={k: dict(cohorts=v['cohorts'], seeds=[min(v['seeds']), max(v['seeds'])], pilot=v['pilot'], why=v['why'])
                                                              for k, v in STAGES.items()},
               cohorts=COHORTS, seed_rule='init_seed = 1000 + i, split_seed = 2000 + i (development used 2/1 and 101-105/201-205)',
               chains=ch)
    fn = os.path.join(HERE, 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1, sort_keys=False)
    h = hashlib.sha256(open(fn, 'rb').read()).hexdigest()
    tot = sum(c['est_core_hours'] for c in ch)
    print(f'{len(ch)} chains, about {tot:.0f} core-hours; manifest sha256 {h}')
    for w in WORKERS:
        q = [c for c in ch if c['worker'] == w]
        print(f"  {w}: {len(q):2d} chains, {sum(c['est_core_hours'] for c in q):5.1f} core-h -> ~{load[w]:.1f} h on {CORES} cores; "
              + ', '.join(f"{st} {sum(1 for c in q if c['stage'] == st)}" for st in STAGES if any(c['stage'] == st for c in q)))


if __name__ == '__main__':
    main()
