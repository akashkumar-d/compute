"""Write manifest.json for campaign F: new seeds at the settings of the campaign-D and depth-2 figures, so that each has 10 (PLAN.md).

Families (every entry is one training run with a fixed configuration; the worker calls it a "chain" as in campaigns C and D):
  H, HW, D3, D4   campaign D's best setting at each p: the seed-1 configuration of campaign_D/manifest.json, with new seeds
                  (init 5000 + i, split 6000 + i), up to i = 10;
  D2              depth 2, development configurations (the seed-0 run files), with new seeds (init 101 + i, split 201 + i), up to
                  i = 9. No early stop, as in the development runs.
Only init_seed and split_seed change. The session assignment uses the measured durations of the existing seeds.

    python3 make_manifest.py            # -> manifest.json (reads ../campaign_D/manifest.json)
"""
import json, hashlib, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'S{i}' for i in range(1, 9)]
CORES = 4

# ---- campaign D settings: (family, p, width, dial) -> (seeds that exist, new seeds); the configuration is seed 1's in campaign D
D_SETTINGS = [
    ('H', 23, 512, dict(lam_w=0.01), [1, 2], range(3, 11)),
    ('H', 31, 512, dict(lam_w=0.015), [1, 2], range(3, 11)),
    ('H', 47, 512, dict(lam_w=0.02), [1, 2], range(3, 11)),
    ('H', 61, 512, dict(lam_w=0.02), [1, 2], range(3, 11)),
    ('H', 97, 512, dict(lam_w=0.02), [1, 2], range(3, 11)),
    ('HW', 97, 2048, dict(lam_w=0.015), [1], range(2, 11)),
    ('D3', 23, 256, dict(r=0.45), [1, 2], range(3, 11)),
    ('D3', 47, 256, dict(r=0.45), [1, 2], range(3, 11)),
    ('D3', 61, 256, dict(r=0.45), [1, 2], range(3, 11)),
    ('D3', 97, 256, dict(r=0.45), [1], range(2, 11)),
    ('D4', 23, 256, dict(r=0.32), [1, 2], range(3, 11)),
    ('D4', 47, 256, dict(r=0.45), [1, 2], range(3, 11)),
    ('D4', 61, 256, dict(r=0.45), [1, 2], range(3, 11)),
    ('D4', 97, 256, dict(r=0.45), [1], range(2, 11)),
]

# ---- depth 2: the development configurations, copied from the seed-0 run files (natural_plateau_v1/runs/*.tar.gz); CHECKS.md
DEV_BASE = dict(deep_init_scale=8.0, depth=2, eta_v=0.03, eta_w=0.03, frac=0.8, heavy_every=500, init_scale=24.0, lam_deep=0.07,
                lam_v=0.2, lam_w=0.015, loss='ce', obs_every=20, width=256)
DEV_SETTINGS = [   # (p, r, eta_deep, steps, id(seed), seeds that exist, new seeds, source of the seed-0 file)
    (23, 0.45, 0.0028928571428571423, 80000, lambda s: f'T16_D2_p23_lw0.015_r0.45_s{s}_T80k', [0, 1], range(2, 10),
     'runs_tasks_lightning_T13_T16_20260927.tar.gz:T16_depth2_p23_long/T16_D2_p23_lw0.015_r0.45_s0_T80k.json.gz'),
    (31, 0.5, 0.003214285714285714, 40000, lambda s: f'seedD2_r0.5_s{s}', [0, 1, 2, 3, 4], range(5, 10),
     'runs_deep_seeds_20260926T2335.tar.gz:seedD2_r0.5_s0.json'),
    (47, 0.5, 0.003214285714285714, 40000, lambda s: f'T3_D2_p47_r0.5_s{s}', [0], range(1, 10),
     'runs_tasks_B_20260927.tar.gz:T3_depth2_other_moduli/T3_D2_p47_r0.5_s0.json.gz'),
]

# ---- measured wall hours per run at the manifest's threads (mean over the existing seeds; campaign D chain.json / run seconds,
# development run seconds). Runs that do not stop early take longer.
MEASURED = {('H', 23): 0.03, ('H', 31): 0.065, ('H', 47): 0.275, ('H', 61): 0.735, ('H', 97): 1.06, ('HW', 97): 2.6,
            ('D3', 23): 0.705, ('D3', 47): 0.675, ('D3', 61): 1.035, ('D3', 97): 1.88,
            ('D4', 23): 1.125, ('D4', 47): 1.19, ('D4', 61): 1.675, ('D4', 97): 2.46,
            ('D2', 23): 1.075, ('D2', 31): 0.195, ('D2', 47): 0.53}


def d_id(fam, p, width, dial, s):
    if fam == 'H': return f'H_p{p}_lw{dial["lam_w"]:.4f}_s{s}'
    if fam == 'HW': return f'HW_p{p}_w{width}_lw{dial["lam_w"]:.4f}_s{s}'
    return f'{fam}_p{p}_r{dial["r"]:.2f}_s{s}'


def build():
    dman = json.load(open(os.path.join(HERE, '..', 'campaign_D', 'manifest.json')))
    dby = {c['id']: c for c in dman['chains']}
    E, S = [], []
    for fam, p, width, dial, have, new in D_SETTINGS:
        src = dby[d_id(fam, p, width, dial, 1)]
        assert src['family'] in (fam, 'D') and src['p'] == p and src['width'] == width and src['dial'] == dial, src['id']
        key = f'{fam}_p{p}' + (f'_w{width}' if fam == 'HW' else '') + (f'_lw{dial["lam_w"]:g}' if 'lam_w' in dial else f'_r{dial["r"]:g}')
        ids = []
        for s in new:
            cfg = dict(src['cfg'], init_seed=5000 + s, split_seed=6000 + s)
            e = dict(id=d_id(fam, p, width, dial, s), stage=src['stage'], family=src['family'], cohort=src['cohort'], setting=key, p=p,
                     depth=src['depth'], width=width, dial=dial, seed=s, init_seed=5000 + s, split_seed=6000 + s, cfg=cfg,
                     threads=src['threads'], est_wall_hours=MEASURED[(fam, p)], est_core_hours=round(MEASURED[(fam, p)] * src['threads'], 3),
                     config_from=src['id'])
            assert e['id'] not in dby, e['id']
            E.append(e); ids.append(e['id'])
        S.append(dict(setting=key, family=fam, p=p, depth=src['depth'], width=width, dial=dial, source='campaign D',
                      config_from=src['id'], seed_rule='init 5000 + i, split 6000 + i',
                      prior=[d_id(fam, p, width, dial, s) for s in have], new=ids))
    for p, r, eta_deep, steps, rid, have, new, src in DEV_SETTINGS:
        key = f'D2_p{p}_lw0.015_r{r:g}'; ids = []
        for s in new:
            cfg = dict(DEV_BASE, p=p, eta_deep=eta_deep, steps=steps, init_seed=101 + s, split_seed=201 + s)
            e = dict(id=rid(s), stage='D2', family='D2', cohort=f'D2_p{p}', setting=key, p=p, depth=2, width=256, dial=dict(lam_w=0.015, r=r),
                     seed=s, init_seed=101 + s, split_seed=201 + s, cfg=cfg, threads=1, est_wall_hours=MEASURED[('D2', p)],
                     est_core_hours=MEASURED[('D2', p)], config_from=rid(0))
            E.append(e); ids.append(e['id'])
        S.append(dict(setting=key, family='D2', p=p, depth=2, width=256, dial=dict(lam_w=0.015, r=r), source='development runs',
                      config_from=src, seed_rule='init 101 + i, split 201 + i', prior=[rid(s) for s in have], new=ids))
    return E, S


def assign(E):
    """Longest-first greedy on each session's load (core-hours / cores), as in campaign D; inside a session, the 4-thread runs first
    (so they never wait for cores to drain), then long runs first."""
    load = {w: 0.0 for w in WORKERS}
    for e in sorted(E, key=lambda e: (-e['est_wall_hours'] * e['threads'], e['id'])):
        w = min(WORKERS, key=lambda w: (load[w], w))
        e['worker'] = w; load[w] += e['est_core_hours'] / CORES
    for w in WORKERS:
        q = sorted([e for e in E if e['worker'] == w], key=lambda e: (-e['threads'], -e['est_wall_hours'], e['id']))
        for i, e in enumerate(q): e['order'] = i
    return load


def main():
    E, S = build(); load = assign(E)
    ids = [e['id'] for e in E]; assert len(ids) == len(set(ids)), 'duplicate ids'
    man = dict(campaign='F', version=1, created=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               author='claude/plateau-review (session_011vNRtJNdeKePn3Gex5aRg7)',
               kind='new seeds at locked settings (PLAN.md): campaign D best settings and depth 2, to 10 seeds each',
               workers=WORKERS, cores_per_worker=CORES, settings=S, est_session_hours={w: round(v, 2) for w, v in load.items()}, chains=E)
    fn = os.path.join(HERE, 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    tot = sum(e['est_core_hours'] for e in E)
    print(f"{len(E)} runs in {len(S)} settings, {tot:.0f} core-hours at the measured durations; per session: "
          + ', '.join(f'{w} {v:.1f} h' for w, v in load.items()))
    by = {}
    for e in E: by.setdefault(e['family'], [0, 0.0]); by[e['family']][0] += 1; by[e['family']][1] += e['est_core_hours']
    print('by family: ' + ', '.join(f'{k} {v[0]} runs / {v[1]:.0f} core-h' for k, v in by.items()))
    print('threads: ' + ', '.join(f'{n}: {sum(1 for e in E if e["threads"] == n)}' for n in (1, 4)))
    print('sha256', hashlib.sha256(open(fn, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
