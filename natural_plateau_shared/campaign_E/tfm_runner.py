#!/usr/bin/env python3
"""Campaign-E runner: transformers for the natural-CE-plateau pilot (claude/plateau-review, 2026-09-30). numpy only, full batch; cfg['dtype']
float64 (default) or float32 (polar steps are always computed in float64).

Task and outputs follow the campaign runner (natplat_campaign.py): modular addition a + b mod p, a random split of the p^2 pairs,
cross-entropy on the training pairs, and a gzipped JSON with cfg, the training loss at every step and observations
(accuracies and margins every obs_every steps; heavy observers every heavy_every steps).

Model: decoder-only transformer on the sequence [a, b, =] (vocabulary p + 1), learned token and position embeddings, n_layers
blocks of causal multi-head attention and a ReLU MLP, residual stream, no LayerNorm and no biases; the logits are read at "=".
The last block computes only the "=" position, since nothing else reaches the logits.

Optimizers (cfg['opt']):
  paper   every weight matrix (embeddings, attention, MLP) takes the supported polar step of its gradient, scaled to unit
          Frobenius norm, with decoupled decay:  M <- M + eta * (polar(-G) / sqrt(rank) - lam * M);
          the unembedding takes the normalized gradient step:  U <- U + eta_u * (-G / |G| - lam_u * U)   (as the paper's head)
  hybrid  the polar step on the attention and MLP matrices; AdamW on the embeddings and the unembedding
  adamw   AdamW on every parameter (decoupled weight decay, as torch.optim.AdamW)

Heavy observers (read-only): AH_full_h<l> = the addition fraction of the MLP activations of block l at "=" (class-mean energy of
the centered activations over all p^2 inputs, as in the campaign runner), AH_full_r<l> the same for the residual stream after
block l, refit_acc = held-out accuracy of a train-only ridge readout of the final residual stream at "=", refit_acc_mlp the same
from the last MLP activations, normV = |unembedding|_F, and the norm of every matrix.

    python3 tfm_runner.py one CFG.json OUT.json.gz              (resumes from OUT.json.gz.ckpt.npz if present)
    python3 tfm_runner.py run JOBS.json OUTDIR --workers N        (JOBS: {name: cfg})
    python3 tfm_runner.py gradcheck
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import gzip, json, math, sys, time
import numpy as np


def polar_dir(G, rel_cut=1e-7):
    """Supported polar factor of G (as natplat_campaign.polar_dir): U V^T over singular values above rel_cut * s_max."""
    tall = G.shape[0] >= G.shape[1]
    M = G.T @ G if tall else G @ G.T
    ev, Q = np.linalg.eigh(M)
    smax = math.sqrt(max(ev[-1], 0.0))
    if smax == 0: return np.zeros_like(G), 0
    keep = ev > (rel_cut * smax) ** 2
    Qk = Q[:, keep]; sk = np.sqrt(ev[keep])
    P = (G @ Qk) / sk @ Qk.T if tall else Qk @ (Qk.T @ G / sk[:, None])
    return P, int(keep.sum())


MATS = ('Wq', 'Wk', 'Wv', 'Wo', 'Win', 'Wout')


class TRun:
    def __init__(s, cfg):
        s.cfg = cfg; p = cfg['p']; s.p = p
        a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
        perm = np.random.default_rng(cfg['split_seed']).permutation(p * p); n = int(cfg['frac'] * p * p)
        s.tr = np.sort(perm[:n]); s.te = np.sort(perm[n:]); s.y = y
        s.X = np.stack([a, b, np.full(p * p, p)], 1)                          # tokens: a, b, "=" (id p)
        s.Y = np.eye(p)[y]
        d, L, dm = cfg['d_model'], cfg['n_layers'], cfg['d_mlp']
        rng = np.random.default_rng(cfg['init_seed']); ini = cfg.get('init', {})
        ce, cm, ch = ini.get('emb', 1.0), ini.get('mat', 1.0), ini.get('head', 1.0)
        P = {'WE': ce * rng.normal(size=(p + 1, d)) / math.sqrt(d), 'Wpos': ce * rng.normal(size=(3, d)) / math.sqrt(d)}
        for l in range(L):
            for nm, shp in (('Wq', (d, d)), ('Wk', (d, d)), ('Wv', (d, d)), ('Wo', (d, d)), ('Win', (d, dm)), ('Wout', (dm, d))):
                P[f'{nm}{l}'] = cm * rng.normal(size=shp) / math.sqrt(shp[0])
        P['WU'] = ch * rng.normal(size=(d, p)) / math.sqrt(d)
        s.dt = np.dtype(cfg.get('dtype', 'float64')); P = {k: v.astype(s.dt) for k, v in P.items()}; s.Y = s.Y.astype(s.dt)
        s.P = P; s.L = L; s.H = cfg['n_heads']; s.d = d
        s.adam = {}; s.t_adam = 0

    # ------------------------------------------------------------------ forward / backward
    def _block(s, x, l, last_only):
        P, H, d = s.P, s.H, s.d; n, T, _ = x.shape; dh = d // H
        xq = x[:, -1:, :] if last_only else x
        Tq = xq.shape[1]
        q = (xq @ P[f'Wq{l}']).reshape(n, Tq, H, dh).transpose(0, 2, 1, 3)
        k = (x @ P[f'Wk{l}']).reshape(n, T, H, dh).transpose(0, 2, 1, 3)
        v = (x @ P[f'Wv{l}']).reshape(n, T, H, dh).transpose(0, 2, 1, 3)
        sc = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)                      # (n, H, Tq, T)
        if not last_only:
            sc = sc + np.triu(np.full((T, T), -np.inf, dtype=sc.dtype), 1)
        sc = sc - sc.max(-1, keepdims=True); A = np.exp(sc); A /= A.sum(-1, keepdims=True)
        z = (A @ v).transpose(0, 2, 1, 3).reshape(n, Tq, d)
        x1 = xq + z @ P[f'Wo{l}']
        pre = x1 @ P[f'Win{l}']; h = np.maximum(pre, 0.0)
        x2 = x1 + h @ P[f'Wout{l}']
        return x2, dict(x=x, xq=xq, q=q, k=k, v=v, A=A, z=z, x1=x1, pre=pre, h=h, last_only=last_only)

    def forward(s, rows):
        P = s.P; X = s.X[rows]
        x = P['WE'][X] + P['Wpos'][None]
        caches = []
        for l in range(s.L):
            x, c = s._block(x, l, last_only=(l == s.L - 1)); caches.append(c)
        return x[:, -1, :] @ P['WU'], x[:, -1, :], caches

    def _block_back(s, dx2, c, l, G):
        P, H, d = s.P, s.H, s.d; x = c['x']; n, T, _ = x.shape; dh = d // H; Tq = c['xq'].shape[1]
        G[f'Wout{l}'] = c['h'].reshape(-1, c['h'].shape[-1]).T @ dx2.reshape(-1, d)
        dpre = (dx2 @ P[f'Wout{l}'].T) * (c['pre'] > 0)
        G[f'Win{l}'] = c['x1'].reshape(-1, d).T @ dpre.reshape(-1, dpre.shape[-1])
        dx1 = dx2 + dpre @ P[f'Win{l}'].T
        G[f'Wo{l}'] = c['z'].reshape(-1, d).T @ dx1.reshape(-1, d)
        dz = (dx1 @ P[f'Wo{l}'].T).reshape(n, Tq, H, dh).transpose(0, 2, 1, 3)
        A, v, q, k = c['A'], c['v'], c['q'], c['k']
        dA = dz @ v.transpose(0, 1, 3, 2); dv = A.transpose(0, 1, 3, 2) @ dz
        ds = A * (dA - (dA * A).sum(-1, keepdims=True)) / math.sqrt(dh)
        dq = ds @ k; dk = ds.transpose(0, 1, 3, 2) @ q
        dq = dq.transpose(0, 2, 1, 3).reshape(n, Tq, d); dk = dk.transpose(0, 2, 1, 3).reshape(n, T, d)
        dv = dv.transpose(0, 2, 1, 3).reshape(n, T, d)
        G[f'Wq{l}'] = c['xq'].reshape(-1, d).T @ dq.reshape(-1, d)
        G[f'Wk{l}'] = x.reshape(-1, d).T @ dk.reshape(-1, d)
        G[f'Wv{l}'] = x.reshape(-1, d).T @ dv.reshape(-1, d)
        dx = dk @ P[f'Wk{l}'].T + dv @ P[f'Wv{l}'].T
        if c['last_only']: dx[:, -1:, :] += dx1 + dq @ P[f'Wq{l}'].T
        else: dx += dx1 + dq @ P[f'Wq{l}'].T
        return dx

    def loss_and_grads(s, rows=None):
        rows = s.tr if rows is None else rows; n = len(rows); y = s.y[rows]
        F, xf, caches = s.forward(rows)
        Z = F - F.max(1, keepdims=True); Pr = np.exp(Z); Pr /= Pr.sum(1, keepdims=True)
        rel = F - F[np.arange(n), y][:, None]; rel[np.arange(n), y] = -np.inf
        mx = rel.max(1); lr = mx + np.log(np.exp(rel - mx[:, None]).sum(1)); L = float(np.logaddexp(0.0, lr).mean())
        dF = (Pr - s.Y[rows]) / n
        G = {'WU': xf.T @ dF}
        dx = np.zeros((n, 1, s.d), dtype=s.dt); dx[:, 0, :] = dF @ s.P['WU'].T
        for l in range(s.L - 1, -1, -1):
            dx = s._block_back(dx, caches[l], l, G)
        GE = np.zeros_like(s.P['WE']); np.add.at(GE, s.X[rows], dx)
        G['WE'] = GE; G['Wpos'] = dx.sum(0)
        return L, G

    # ------------------------------------------------------------------ updates
    def _polar_step(s, k, g, eta, lam):
        Q, r = polar_dir(-g.astype(np.float64)); Q = Q.astype(s.dt)
        s.P[k] = s.P[k] + eta * ((Q / math.sqrt(r) if r else 0 * g) - lam * s.P[k])

    def _adamw_step(s, k, g, lr, wd, b1, b2, eps):
        m, v = s.adam.get(k, (np.zeros_like(g), np.zeros_like(g)))
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g; s.adam[k] = (m, v)
        t = s.t_adam
        s.P[k] = s.P[k] * (1 - lr * wd) - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

    def step(s, G):
        c = s.cfg; o = c['opt']; s.t_adam += 1
        ad = c.get('adamw', {}); lr, wd = ad.get('lr', 1e-3), ad.get('wd', 1.0)
        b1, b2, eps = ad.get('b1', 0.9), ad.get('b2', 0.98), ad.get('eps', 1e-8)
        for k, g in G.items():
            is_mat = k.rstrip('0123456789') in MATS
            if o == 'adamw' or (o == 'hybrid' and not is_mat):
                s._adamw_step(k, g, lr, wd, b1, b2, eps)
            elif k == 'WU':                                                    # paper: normalized head step
                gn = float(np.linalg.norm(g))
                s.P[k] = s.P[k] + c['eta_v'] * ((-g / gn if gn > 0 else 0 * g) - c['lam_v'] * s.P[k])
            else:
                emb = k in ('WE', 'Wpos')
                s._polar_step(k, g, c.get('eta_emb', c['eta_w']) if emb else c['eta_w'], c.get('lam_emb', c['lam_w']) if emb else c['lam_w'])

    # ------------------------------------------------------------------ observers
    def _ah(s, Hm):
        y, p = s.y, s.p
        C = Hm - Hm.mean(0); den = float((C * C).sum())
        M = np.stack([C[y == k].mean(0) for k in range(p)])
        return float((M[y] ** 2).sum() / den) if den > 0 else 0.0

    def _ridge_acc(s, Hm):
        tr, te, y = s.tr, s.te, s.y
        Htr = Hm[tr]; mu = Htr.mean(0); Hc = Htr - mu; A = Hc.T @ Hc; lam = 1e-3 * np.trace(A) / A.shape[0] + 1e-12
        B = np.linalg.solve(A + lam * np.eye(A.shape[0]), Hc.T @ (s.Y[tr] - s.Y[tr].mean(0)))
        return float((((Hm[te] - mu) @ B).argmax(1) == y[te]).mean())

    def observe(s, heavy):
        p = s.p; allr = np.arange(p * p); y, tr, te = s.y, s.tr, s.te
        F, xf, caches = s.forward(allr)
        pred = F.argmax(1); other = F.copy(); other[allr, y] = -np.inf; marg = F[allr, y] - other.max(1)
        o = dict(train_acc=float((pred[tr] == y[tr]).mean()), test_acc=float((pred[te] == y[te]).mean()),
                 train_margin_min=float(marg[tr].min()), test_margin_min=float(marg[te].min()))
        Ft = F[te]; Z = Ft - Ft.max(1, keepdims=True); Pt = np.exp(Z); Pt /= Pt.sum(1, keepdims=True)
        o['test_loss'] = float(-np.log(np.maximum(Pt[np.arange(len(te)), y[te]], 1e-300)).mean())
        if heavy:
            for l, c in enumerate(caches):
                o[f'AH_full_h{l + 1}'] = s._ah(c['h'][:, -1, :])
                o[f'AH_full_r{l + 1}'] = s._ah(xf if l == s.L - 1 else caches[l + 1]['x'][:, -1, :])
            o['AH_full_logit'] = s._ah(F - F.mean(1, keepdims=True))
            o['refit_acc'] = s._ridge_acc(xf); o['refit_acc_mlp'] = s._ridge_acc(caches[-1]['h'][:, -1, :])
            o['normV'] = float(np.linalg.norm(s.P['WU']))
            for k, v in s.P.items(): o['norm_' + k] = float(np.linalg.norm(v))
            o['logit_rms'] = float(np.sqrt((F ** 2).mean()))
        return o


def _generalized_since(obs, acc):
    s_ = None
    for o in obs:
        if o['test_acc'] >= acc:
            if s_ is None: s_ = o['step']
        else: s_ = None
    return s_


def _save_ckpt(path, R, t_next, loss, obs, gen_since):
    tmp = path + '.tmp.npz'
    arrs = {f'P_{k}': v for k, v in R.P.items()}
    for k, (m, v) in R.adam.items(): arrs[f'm_{k}'] = m; arrs[f'v_{k}'] = v
    np.savez(tmp, t_next=t_next, t_adam=R.t_adam, loss=loss[:t_next], obs=json.dumps(obs),
             gen_since=-1 if gen_since is None else gen_since, **arrs)
    os.replace(tmp, path)


def run(cfg, out, ckpt_every=2000):
    """One run; `out` ends with .json.gz. Resumes from out + '.ckpt.npz' if present (bit-identical to an uninterrupted run)."""
    R = TRun(cfg); T = cfg['steps']; t0 = time.time(); loss = np.empty(T + 1); obs = []; t_start = 0
    sg = cfg.get('stop_generalized'); gen_since = None; stop = False; reason = None
    ck = out + '.ckpt.npz'
    if os.path.exists(ck):
        with np.load(ck, allow_pickle=False) as z:
            t_start = int(z['t_next']); loss[:t_start] = z['loss']; obs = json.loads(str(z['obs'])); R.t_adam = int(z['t_adam'])
            gs = int(z['gen_since']); gen_since = None if gs < 0 else gs
            for k in list(R.P): R.P[k] = z[f'P_{k}']
            for k in R.P:
                if f'm_{k}' in z.files: R.adam[k] = (z[f'm_{k}'], z[f'v_{k}'])
    for t in range(t_start, T + 1):
        if ckpt_every and t > t_start and t % ckpt_every == 0: _save_ckpt(ck, R, t, loss, obs, gen_since)
        L, G = R.loss_and_grads(); loss[t] = L
        if not math.isfinite(L): reason = 'nonfinite'; stop = True
        if t % cfg['obs_every'] == 0 or t == T or stop:
            o = R.observe(heavy=(t % cfg['heavy_every'] == 0 or t == T or stop)); o['step'] = t; o['loss'] = L
            if sg is not None and t < T and not stop:
                if o['test_acc'] >= sg['acc']:
                    if gen_since is None: gen_since = t
                    if t >= max(sg['factor'] * gen_since, gen_since + sg['extra']):
                        if 'normV' not in o: o = R.observe(heavy=True); o['step'] = t; o['loss'] = L
                        stop = True; reason = 'generalized'
                else: gen_since = None
            obs.append(o)
        if t == T or stop: break
        R.step(G)
    res = dict(cfg=cfg, loss=loss[:t + 1].tolist(), obs=obs, seconds=time.time() - t0, resumed_from=t_start)
    if stop: res['stopped_at'] = t; res['stop_reason'] = reason
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(res, f)
    os.replace(tmp, out)
    if os.path.exists(ck): os.remove(ck)
    return res


def _job(args):
    name, cfg, outdir = args
    out = os.path.join(outdir, name + '.json.gz')
    if os.path.exists(out): return name, 'exists', 0.0
    t = time.time()
    try: run(cfg, out); msg = 'ok'
    except Exception as e: msg = 'ERROR ' + repr(e)
    with open(os.path.join(outdir, 'queue.log'), 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} {msg} {time.time() - t:.0f}s\n")
    return name, msg, time.time() - t


def gradcheck():
    cfg = dict(p=5, frac=0.8, split_seed=1, init_seed=2, d_model=8, n_heads=2, d_mlp=12, n_layers=2, opt='paper',
               init=dict(emb=1.0, mat=1.0, head=1.0))
    R = TRun(cfg); rng = np.random.default_rng(0)
    L0, G = R.loss_and_grads(); worst = 0.0
    for k in R.P:
        for _ in range(4):
            idx = tuple(rng.integers(0, n) for n in R.P[k].shape); h = 1e-6
            R.P[k][idx] += h; Lp, _ = R.loss_and_grads(); R.P[k][idx] -= 2 * h; Lm, _ = R.loss_and_grads(); R.P[k][idx] += h
            num = (Lp - Lm) / (2 * h); ana = G[k][idx]
            err = abs(num - ana) / max(1e-8, abs(num) + abs(ana)); worst = max(worst, err)
            if err > 1e-5: print('MISMATCH', k, idx, num, ana, err)
    print(f'gradcheck: worst relative error {worst:.2e} over {4 * len(R.P)} coordinates ({"ok" if worst < 1e-5 else "FAIL"})')
    return worst < 1e-5


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'gradcheck': sys.exit(0 if gradcheck() else 1)
    elif cmd == 'one': run(json.load(open(sys.argv[2])), sys.argv[3])
    elif cmd == 'run':
        from multiprocessing import Pool
        jobs = json.load(open(sys.argv[2])); outdir = sys.argv[3]; os.makedirs(outdir, exist_ok=True)
        w = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 2
        with Pool(w) as pool:
            for name, msg, sec in pool.imap_unordered(_job, [(n, c, outdir) for n, c in jobs.items()]):
                print(f'{name}: {msg} ({sec:.0f}s)', flush=True)
