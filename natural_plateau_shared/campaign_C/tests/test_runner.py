"""Checks for natplat_campaign.py: (1) without the new keys it reproduces code/natplat_server.py bit for bit; (2) observers_v1 leave the
learner untouched; (3) stop_at_test stops at the first logged step with held-out accuracy >= the threshold; (4) hidden_mode='ngd' runs and
differs from polar; (5) the v1 observers give the textbook answers on synthetic inputs."""
import json, os, sys, tempfile, gzip, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
NEW = _mod(os.path.join(ROOT, 'natplat_campaign.py'), 'natplat_campaign')
OLD_PATH = os.environ.get('OLD_RUNNER', os.path.join(ROOT, '..', 'code', 'natplat_server.py'))
BASE = dict(frac=0.8, width=48, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, heavy_every=100, p=11, lam_v=0.2,
            init_seed=5, split_seed=6, lam_w=0.015, deep_init_scale=8.0, lam_deep=0.07, depth=3, steps=600,
            eta_layers=[0.03, 0.006428571428571428, 0.003214285714285714], lam_layers=[0.015, 0.07, 0.07])

def _run(mod, cfg):
    d = tempfile.mkdtemp(); out = os.path.join(d, 'x.json.gz'); return mod.run(cfg, out, ckpt_every=0)

def test_bitexact_legacy():
    if not os.path.exists(OLD_PATH): print('skip: old runner not found'); return
    OLD = _mod(OLD_PATH, 'natplat_server_old')
    a = _run(OLD, dict(BASE)); b = _run(NEW, dict(BASE))
    assert a['loss'] == b['loss'], 'loss differs'
    assert json.dumps(a['obs']) == json.dumps(b['obs']), 'observations differ'

def test_v1_observers_do_not_touch_learner():
    a = _run(NEW, dict(BASE)); b = _run(NEW, dict(BASE, observers_v1=True))
    assert a['loss'] == b['loss']
    hv = [o for o in b['obs'] if 'v1_last_AH' in o]; assert len(hv) == 7, len(hv)
    for k in ('v1_last_agop_purity', 'v1_first_agop_purity', 'v1_ridge_mse_te', 'v1_signed_bound_q50', 'v1_last_Adiff'):
        assert k in hv[-1], k
    # legacy A_H and v1 A_H agree
    for o in hv: assert abs(o['v1_last_AH'] - o['AH_full_h3']) < 1e-9 and abs(o['v1_first_AH'] - o['AH_full_h1']) < 1e-9

def test_stop_at_test():
    cfg = dict(BASE, steps=4000, stop_at_test=0.1)
    r = _run(NEW, cfg); full = _run(NEW, dict(BASE, steps=4000))
    te = [(o['step'], o['test_acc']) for o in full['obs']]
    first = next((s for s, v in te if v >= 0.1), None)
    if first is None: print('note: test never reached 0.1 in 4000 steps; stop check skipped'); return
    assert r.get('stopped_at') == first, (r.get('stopped_at'), first)
    assert len(r['loss']) == first + 1 and r['loss'] == full['loss'][:first + 1]
    assert 'normW' in r['obs'][-1]

def test_ngd_mode():
    a = _run(NEW, dict(BASE, steps=200)); b = _run(NEW, dict(BASE, steps=200, hidden_mode='ngd'))
    assert a['loss'][:1] == b['loss'][:1] and a['loss'][-1] != b['loss'][-1]

def test_v1_synthetic():
    p = 11; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p; yd = (a - b) % p
    k = np.arange(1, 4)
    Hadd = np.concatenate([np.cos(2 * np.pi * np.outer(y, k) / p), np.sin(2 * np.pi * np.outer(y, k) / p)], 1) + 3.0
    Hdif = np.concatenate([np.cos(2 * np.pi * np.outer(yd, k) / p), np.sin(2 * np.pi * np.outer(yd, k) / p)], 1)
    Ep, Et, Edc = NEW._v1_energy(Hadd, y, p); assert abs(Ep / Et - 1) < 1e-12 and abs(Edc - 6 * 9.0) < 1e-9
    Ep, Et, _ = NEW._v1_energy(Hdif, y, p); assert Ep / Et < 1e-12
    Ed, Et, _ = NEW._v1_energy(Hdif, yd, p); assert abs(Ed / Et - 1) < 1e-12
    Ep, Et, _ = NEW._v1_energy(np.ones((p * p, 4)), y, p); assert Et == 0
    B = NEW.fourier_basis(y, p); Bc = B - B.mean(0)
    O, rz, rf = NEW._v1_overlap(Bc[:, :4], Bc); assert abs(O / rz - 1) < 1e-9 and rz == 4 and rf == p - 1
    Zd = Hdif - Hdif.mean(0); O, rz, rf = NEW._v1_overlap(Zd, Bc); assert O / rz < 1e-9
    # signed margins: perfect rule logits give gamma = s, beta = 0
    cfg = dict(BASE, p=p, depth=1, width=16, steps=0); cfg.pop('eta_layers'); cfg.pop('lam_layers')
    R = NEW.Run(cfg); s_ = 2.5
    R.V = np.zeros_like(R.V)
    class Fake: pass
    F = s_ * np.eye(p)[y]
    pres, hs, _ = R.forward(np.arange(p * p))
    o = NEW.observe_v1(R, pres, hs, F)
    assert abs(o['v1_signed_gamma_min'] - s_) < 1e-9 and abs(o['v1_signed_beta_q95']) < 1e-9 and o['v1_signed_certified'] == 1.0

if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'): f(); print('PASS', name)
