"""Tests of the campaign-E transformer runner (run by `worker.py selftest`, or `python3 tests/test_tfm.py`)."""
import importlib.util, json, os, shutil, tempfile
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('tfm_runner', os.path.join(HERE, 'tfm_runner.py'))
TR = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(TR)


def _cfg(opt, L=2, dtype='float64', steps=100):
    return dict(p=7, frac=0.8, split_seed=1, init_seed=2, d_model=16, n_heads=2, d_mlp=32, n_layers=L, opt=opt, dtype=dtype, steps=steps,
                obs_every=10, heavy_every=50, eta_w=0.03, lam_w=0.015, eta_v=0.03, lam_v=0.2, init=dict(emb=1.0, mat=1.0, head=1.0),
                adamw=dict(lr=1e-3, wd=1.0, b1=0.9, b2=0.98, eps=1e-8))


def test_gradients():
    """Analytic gradients of every parameter against central finite differences (float64, 2 layers)."""
    assert TR.gradcheck(), 'gradients differ from finite differences'


def test_last_position_shortcut():
    """The last block computes only the '=' position; this equals computing every position and reading '='."""
    R = TR.TRun(_cfg('paper', L=1))
    F1, _, _ = R.forward(R.tr)
    x = R.P['WE'][R.X[R.tr]] + R.P['Wpos'][None]
    x2, _ = R._block(x, 0, last_only=False)
    assert float(np.abs(F1 - (x2[:, -1, :] @ R.P['WU'])).max()) < 1e-12


def test_resume_identical():
    """A run resumed from a checkpoint equals the uninterrupted run, for every optimizer (Adam states included)."""
    d = tempfile.mkdtemp()
    try:
        for opt in ('paper', 'hybrid', 'adamw'):
            cfg = _cfg(opt)
            a = TR.run(cfg, os.path.join(d, f'a_{opt}.json.gz'), ckpt_every=0)
            out = os.path.join(d, f'b_{opt}.json.gz')
            R = TR.TRun(cfg); loss = np.empty(cfg['steps'] + 1); obs = []
            for t in range(0, 41):
                if t == 40: TR._save_ckpt(out + '.ckpt.npz', R, t, loss, obs, None); break
                L, G = R.loss_and_grads(); loss[t] = L
                if t % cfg['obs_every'] == 0:
                    o = R.observe(heavy=(t % cfg['heavy_every'] == 0)); o['step'] = t; o['loss'] = L; obs.append(o)
                R.step(G)
            b = TR.run(cfg, out, ckpt_every=0)
            assert b['resumed_from'] == 40
            assert a['loss'] == b['loss'] and json.dumps(a['obs']) == json.dumps(b['obs']), f'{opt}: resumed run differs'
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for n in sorted(k for k in dir() if k.startswith('test_')):
        globals()[n](); print('ok', n)
