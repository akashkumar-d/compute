"""Checks for campaign D's natplat_campaign.py: (1) without the new keys it reproduces code/natplat_server.py bit for bit, and without
stop_generalized it reproduces campaign C's runner bit for bit; (2) observers_v1 leave the learner untouched, also for one hidden
layer; (3) stop_at_test stops at the first logged step with held-out accuracy >= the threshold; (4) stop_generalized stops where
its rule says, with the same trajectory up to there, also after a resume; (5) hidden_mode='ngd' runs and differs from polar;
(6) the v1 observers give the textbook answers on synthetic inputs."""
import json, os, sys, tempfile, gzip, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
NEW = _mod(os.path.join(ROOT, 'natplat_campaign.py'), 'natplat_campaign')
OLD_PATH = os.environ.get('OLD_RUNNER', os.path.join(ROOT, '..', 'code', 'natplat_server.py'))
C_PATH = os.environ.get('C_RUNNER', os.path.join(ROOT, '..', 'campaign_C', 'natplat_campaign.py'))
BASE = dict(frac=0.8, width=48, loss='ce', eta_w=0.03, eta_v=0.03, init_scale=24.0, obs_every=20, heavy_every=100, p=11, lam_v=0.2,
            init_seed=5, split_seed=6, lam_w=0.015, deep_init_scale=8.0, lam_deep=0.07, depth=3, steps=600,
            eta_layers=[0.03, 0.006428571428571428, 0.003214285714285714], lam_layers=[0.015, 0.07, 0.07])

def _run(mod, cfg, ckpt_every=0, out=None):
    out = out or os.path.join(tempfile.mkdtemp(), 'x.json.gz'); return mod.run(cfg, out, ckpt_every=ckpt_every)

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

def test_bitexact_campaign_C():
    if not os.path.exists(C_PATH): print('skip: campaign C runner not found'); return
    OLDC = _mod(C_PATH, 'natplat_campaign_C')
    for cfg in (dict(BASE, observers_v1=True, steps=300), dict(BASE, hidden_mode='ngd', steps=200), dict(BASE, steps=4000, stop_at_test=0.1)):
        a = _run(OLDC, dict(cfg)); b = _run(NEW, dict(cfg))
        assert a['loss'] == b['loss'] and json.dumps(a['obs']) == json.dumps(b['obs']) and a.get('stopped_at') == b.get('stopped_at')
        assert 'stop_reason' not in b

def test_one_hidden_layer_v1():
    cfg = dict(BASE, depth=1, width=64, steps=300); cfg.pop('eta_layers'); cfg.pop('lam_layers')
    a = _run(NEW, dict(cfg)); b = _run(NEW, dict(cfg, observers_v1=True))
    assert a['loss'] == b['loss']
    hv = [o for o in b['obs'] if 'v1_last_AH' in o]; assert hv and abs(hv[-1]['v1_last_AH'] - hv[-1]['AH_full_h1']) < 1e-9

def _expected_stop(obs, acc, factor, extra, T):
    s = None
    for o in obs:
        t = o['step']
        if o['test_acc'] >= acc:
            if s is None: s = t
            if t < T and t >= max(factor * s, s + extra): return t
        else:
            s = None
    return None

def test_stop_generalized():
    full = _run(NEW, dict(BASE, steps=4000))
    te = [o['test_acc'] for o in full['obs']]
    acc = sorted(te)[int(0.6 * len(te))]            # a level the toy run reaches and keeps for a while
    sg = dict(acc=acc, factor=1.2, extra=200)
    want = _expected_stop(full['obs'], acc, 1.2, 200, 4000)
    r = _run(NEW, dict(BASE, steps=4000, stop_generalized=sg))
    if want is None:
        assert 'stopped_at' not in r and r['loss'] == full['loss']; print('note: rule never fired on the toy run'); return
    assert r['stopped_at'] == want and r['stop_reason'] == 'generalized', (r.get('stopped_at'), want)
    assert r['loss'] == full['loss'][:want + 1] and 'normW' in r['obs'][-1]
    # the same after an interrupted run resumes from its checkpoint
    d = tempfile.mkdtemp(); out = os.path.join(d, 'y.json.gz'); cfg = dict(BASE, steps=4000, stop_generalized=sg)
    ck_at = max(200, (want // 2) // 200 * 200)
    part = dict(cfg, steps=4000); R = NEW.Run(part); import numpy as _np
    loss = _np.empty(4001); obs = []
    for t in range(0, ck_at):
        L, gW, gV = R.loss_and_grads(); loss[t] = L
        if t % part['obs_every'] == 0:
            o = R.observe(heavy=(t % part['heavy_every'] == 0)); o['step'] = t; o['loss'] = L; obs.append(o)
        R.step(gW, gV, t)
    NEW._save_ckpt(out + '.ckpt.npz', R, ck_at, loss, obs)
    r2 = NEW.run(cfg, out, ckpt_every=0)
    assert r2['stopped_at'] == want and r2['loss'] == r['loss'], (r2.get('stopped_at'), want)

if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'): f(); print('PASS', name)
