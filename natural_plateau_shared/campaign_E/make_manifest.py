"""Write manifest.json for campaign E: the transformer pilot at p = 31 (PLAN.md).

Cells: 1- and 2-layer transformers x three optimizers (paper, hybrid, adamw) = 6 cells.
  development chains  4 settings per cell on the tuning seed (init 101, split 201): 24 runs
  fresh-seed chains   10 per cell (init 1001 + i, split 2001 + i); each waits for its cell's 4 development runs, then trains the
                      setting chosen by PLAN.md's rule, or records a skip if that setting did not pass: 60 chains
Every run uses one core. Seconds per step were measured on a 2-core cloud container (float32).

    python3 make_manifest.py            # -> manifest.json and HASHES.json
"""
import datetime, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = [f'S{i}' for i in range(1, 9)]
CORES = 4
P = 31
STOP = dict(acc=0.99, factor=1.5, extra=10000)
BASE = dict(p=P, frac=0.8, d_model=128, n_heads=4, d_mlp=512, dtype='float32', steps=40000, obs_every=20, heavy_every=500,
            stop_generalized=STOP, eta_w=0.03, eta_v=0.03, lam_v=0.2, adamw=dict(lr=1e-3, wd=1.0, b1=0.9, b2=0.98, eps=1e-8))
ARMS = ('paper', 'hybrid', 'adamw')
SEC = {(1, 'paper'): 0.044, (1, 'hybrid'): 0.044, (1, 'adamw'): 0.031, (2, 'paper'): 0.137, (2, 'hybrid'): 0.120, (2, 'adamw'): 0.101}
OVERHEAD = 1.1          # observers
EARLY = 0.7             # expected fraction of the horizon actually run (stop_generalized)
DEV_SEED = (101, 201)
FRESH = 10


def settings(arm):
    out = []
    for s in (1.0, 4.0):
        if arm == 'adamw':
            for wd in (0.3, 1.0): out.append((f's{s:g}_wd{wd:g}', dict(init=s, wd=wd), dict(init=dict(emb=s, mat=s, head=1.0), adamw=dict(BASE['adamw'], wd=wd))))
        else:
            for lw in (0.005, 0.015): out.append((f's{s:g}_lw{lw:g}', dict(init=s, lam_w=lw), dict(init=dict(emb=s, mat=s, head=1.0), lam_w=lw)))
    return out


def build():
    dev, fresh = [], []
    for L in (1, 2):
        for arm in ARMS:
            cell = f'L{L}_{arm}'; hours = SEC[(L, arm)] * OVERHEAD * BASE['steps'] * EARLY / 3600; ids = []
            for tag, dial, extra in settings(arm):
                cfg = dict(BASE, n_layers=L, opt=arm, init_seed=DEV_SEED[0], split_seed=DEV_SEED[1], **extra)
                cid = f'dev_L{L}_{arm}_{tag}'; ids.append(cid)
                dev.append(dict(id=cid, stage='dev', cell=cell, opt=arm, n_layers=L, p=P, dial=dial, seed=0, init_seed=DEV_SEED[0],
                                split_seed=DEV_SEED[1], cfg=cfg, threads=1, est_core_hours=round(hours, 2)))
            for i in range(FRESH):
                fresh.append(dict(id=f'fresh_L{L}_{arm}_s{i + 1:02d}', stage='fresh', cell=cell, opt=arm, n_layers=L, p=P, dial=None,
                                  seed=i + 1, init_seed=1001 + i, split_seed=2001 + i, after=ids, threads=1, est_core_hours=round(hours, 2),
                                  cfg=dict(steps=BASE['steps'], note='resolved at run time: the development setting chosen by PLAN.md, on these seeds')))
    return dev, fresh


def assign(dev, fresh):
    """Development chains first (they unblock the fresh ones), spread over the sessions by load; then the fresh chains, 1-layer
    cells before 2-layer cells, each cell's seeds spread over the sessions."""
    load = {w: 0.0 for w in WORKERS}; order = {w: 0 for w in WORKERS}
    def put(c):
        w = min(WORKERS, key=lambda w: (load[w], w)); c['worker'] = w; c['order'] = order[w]; order[w] += 1
        load[w] += c['est_core_hours'] / CORES
    for c in sorted(dev, key=lambda c: (-c['est_core_hours'], c['id'])): put(c)
    for c in sorted(fresh, key=lambda c: (c['n_layers'], c['cell'], c['seed'])): put(c)
    return load


def main():
    dev, fresh = build(); E = dev + fresh; load = assign(dev, fresh)
    ids = [e['id'] for e in E]; assert len(ids) == len(set(ids)), 'duplicate ids'
    man = dict(campaign='E', version=1, created=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               author='claude/plateau-review', kind='transformer pilot (development + fresh seeds; rule fixed in PLAN.md before any result)',
               workers=WORKERS, cores_per_worker=CORES, stop_generalized=STOP, p=P,
               seed_rule=f'development: init {DEV_SEED[0]}, split {DEV_SEED[1]}; fresh: init 1001 + i, split 2001 + i (i = 0..{FRESH - 1})',
               est_session_hours={w: round(v, 2) for w, v in load.items()}, chains=E)
    fn = os.path.join(HERE, 'manifest.json')
    with open(fn, 'w') as f: json.dump(man, f, indent=1)
    files = ['worker.py', 'tfm_runner.py', 'tfm_analyze.py', 'endpoints.py', 'analyze.py', 'make_manifest.py', 'manifest.json',
             'tests/test_tfm.py', 'PLAN.md']
    H = dict(created=man['created'], files={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in files})
    with open(os.path.join(HERE, 'HASHES.json'), 'w') as f: json.dump(H, f, indent=1)
    tot = sum(e['est_core_hours'] for e in E)
    print(f'{len(dev)} development + {len(fresh)} fresh chains, {tot:.0f} core-hours at most (fresh chains of a failing cell are skipped); '
          'per session: ' + ', '.join(f'{w} {v:.1f} h' for w, v in load.items()))


if __name__ == '__main__':
    main()
