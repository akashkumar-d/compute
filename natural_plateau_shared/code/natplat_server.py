"""Self-contained runner for the natural-plateau experiments (modular addition, bias-free ReLU MLP, CE).

Same numerics as code/nat.py (verified bit-for-bit on a short run), plus optional per-layer
`eta_layers` / `lam_layers` lists (absent = old behaviour, identical numbers), plus:
  * checkpoint / resume every `ckpt_every` steps (weights + loss + observations), so long runs survive restarts;
  * gzip-compressed outputs (<name>.json.gz) and checkpoint removal on completion (storage-conscious);
  * a process pool with one BLAS thread per worker;
  * a compact text summary so results can be read without downloading files.

Usage (shell or a notebook cell with `!`):
    python natplat_server.py run   jobs.json  OUTDIR  --workers 16
    python natplat_server.py status OUTDIR
    python natplat_server.py summary OUTDIR  [--match SUBSTRING]
jobs.json is a list of {"name": ..., "cfg": {...}} exactly as in code/batch*.json.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, json, math, sys, time, gzip, glob, io

# ----------------------------------------------------------------------------- learner (identical to nat.py)
def fourier_basis(y, p):
    k = np.arange(1, (p - 1) // 2 + 1)
    ang = 2 * np.pi * np.outer(y, k) / p
    return np.concatenate([np.cos(ang), np.sin(ang)], 1)

def orth(M):
    M = M - M.mean(0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > 1e-10 * max(s[0], 1e-300)
    return U[:, keep]

def cca2(Z, B):
    Qz, Qb = orth(Z), orth(B)
    if Qz.shape[1] == 0: return 0.0
    return float((np.linalg.svd(Qz.T @ Qb, compute_uv=False) ** 2).sum() / Qz.shape[1])

def polar_dir(G, rel_cut=1e-7):
    tall = G.shape[0] >= G.shape[1]
    M = G.T @ G if tall else G @ G.T
    ev, Q = np.linalg.eigh(M)
    smax = math.sqrt(max(ev[-1], 0.0))
    if smax == 0: return np.zeros_like(G), 0
    keep = ev > (rel_cut * smax) ** 2
    Qk = Q[:, keep]; sk = np.sqrt(ev[keep])
    P = (G @ Qk) / sk @ Qk.T if tall else Qk @ (Qk.T @ G / sk[:, None])
    return P, int(keep.sum())

class Run:
    def __init__(s, cfg):
        s.cfg = cfg; p = cfg['p']; s.p = p
        a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); y = (a + b) % p
        perm = np.random.default_rng(cfg['split_seed']).permutation(p * p); n = int(cfg['frac'] * p * p)
        s.tr = np.sort(perm[:n]); s.te = np.sort(perm[n:]); s.a, s.b, s.y = a, b, y
        s.Aoh = np.eye(p)[a]; s.Boh = np.eye(p)[b]; s.Y = np.eye(p)[y]
        rng = np.random.default_rng(cfg['init_seed']); D = cfg['depth']
        ws = cfg.get('widths', [cfg['width']] * D)
        c = cfg['init_scale']
        s.Ws = [c * rng.normal(size=(ws[0], 2 * p)) / math.sqrt(p)]
        k = cfg.get('deep_init_scale', 1.0)
        for l in range(1, D): s.Ws.append(k * rng.normal(size=(ws[l], ws[l - 1])) * math.sqrt(2.0 / ws[l - 1]))
        s.V = rng.normal(size=(p, ws[-1])) / (cfg.get('head_init_div', c * k ** (D - 1)) * math.sqrt(ws[-1]))
        s.Bf = fourier_basis(y, p)
    def forward(s, rows):
        W0 = s.Ws[0]; p = s.p
        pre = W0[:, s.a[rows]].T + W0[:, p + s.b[rows]].T
        pres = [pre]; hs = [np.maximum(pre, 0.0)]
        for W in s.Ws[1:]:
            pre = hs[-1] @ W.T; pres.append(pre); hs.append(np.maximum(pre, 0.0))
        F = hs[-1] @ s.V.T
        return pres, hs, F
    def loss_and_grads(s):
        tr = s.tr; n = len(tr); cfg = s.cfg
        pres, hs, F = s.forward(tr)
        if cfg['loss'] == 'ce':
            Z = F - F.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            rel = F - F[np.arange(n), s.y[tr]][:, None]; rel[np.arange(n), s.y[tr]] = -np.inf
            mx = rel.max(1); lr = mx + np.log(np.exp(rel - mx[:, None]).sum(1)); L = float(np.logaddexp(0.0, lr).mean())
            dF = (P - s.Y[tr]) / n
        else:
            R = F - s.Y[tr]; L = float((R * R).sum() / n); dF = 2 * R / n
        gV = dF.T @ hs[-1]
        dh = dF @ s.V; gWs = [None] * len(s.Ws)
        for l in range(len(s.Ws) - 1, -1, -1):
            dpre = dh * (pres[l] > 0)
            if l == 0:
                gWs[0] = np.concatenate([dpre.T @ s.Aoh[tr], dpre.T @ s.Boh[tr]], 1)
            else:
                gWs[l] = dpre.T @ hs[l - 1]; dh = dpre @ s.Ws[l]
        return L, gWs, gV
    def step(s, gWs, gV, t=0):
        cfg = s.cfg
        fh = cfg.get('freeze_hidden_after'); fv = cfg.get('freeze_head_after')
        for l, (W, G) in enumerate(zip(s.Ws, gWs)):
            if fh is not None and t >= fh: continue
            if 'eta_layers' in cfg: eta = cfg['eta_layers'][l]                      # optional per-layer steps
            else: eta = cfg['eta_w'] if l == 0 else cfg.get('eta_deep', cfg['eta_w'])
            if 'lam_layers' in cfg: lam = cfg['lam_layers'][l]                      # optional per-layer decays
            else: lam = cfg['lam_w'] if l == 0 else cfg.get('lam_deep', cfg['lam_w'])
            Q, r = polar_dir(-G)
            s.Ws[l] = W + eta * ((Q / math.sqrt(r) if r else 0 * W) - lam * W)
        gn = np.linalg.norm(gV)
        if fv is not None and t >= fv: return
        if cfg.get('head_mode', 'ngd') == 'ngd':
            s.V = s.V + cfg['eta_v'] * ((-gV / gn if gn > 0 else 0 * gV) - cfg['lam_v'] * s.V)
        elif cfg['head_mode'] == 'polar':
            Q, r = polar_dir(-gV); s.V = s.V + cfg['eta_v'] * (Q / math.sqrt(max(r, 1)) - cfg['lam_v'] * s.V)
    def observe(s, heavy):
        p = s.p; allr = np.arange(p * p)
        pres, hs, F = s.forward(allr)
        y, tr, te = s.y, s.tr, s.te
        pred = F.argmax(1)
        other = F.copy(); other[allr, y] = -np.inf; marg = F[allr, y] - other.max(1)
        o = dict(train_acc=float((pred[tr] == y[tr]).mean()), test_acc=float((pred[te] == y[te]).mean()),
                 train_margin_min=float(marg[tr].min()), test_margin_min=float(marg[te].min()))
        Ft = F[te]
        if s.cfg['loss'] == 'ce':
            Z = Ft - Ft.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True); o['test_loss'] = float(-np.log(np.maximum(P[np.arange(len(te)), y[te]], 1e-300)).mean())
        else:
            o['test_loss'] = float(((Ft - s.Y[te]) ** 2).sum() / len(te))
        if heavy:
            for name, H in [('h%d' % (l + 1), h) for l, h in enumerate(hs)] + [('logit', F - F.mean(1, keepdims=True))]:
                C = H - H.mean(0); den = float((C * C).sum())
                M = np.stack([C[y == c].mean(0) for c in range(p)])
                o['AH_full_' + name] = float((M[y] ** 2).sum() / den) if den > 0 else 0.0
                mu = H[tr].mean(0); Mtr = np.stack([H[tr][y[tr] == c].mean(0) if np.any(y[tr] == c) else mu for c in range(p)])
                sse_c = float(((H[te] - Mtr[y[te]]) ** 2).sum()); sse_0 = float(((H[te] - mu) ** 2).sum())
                o['AH_ho_' + name] = 1 - sse_c / sse_0 if sse_0 > 0 else 0.0
            Hl = hs[-1]; Gm = s.V.T @ s.V; ev, U = np.linalg.eigh(Gm); U = U[:, ::-1]
            rng = np.random.default_rng(12345)
            for r in (4, p - 1):
                Ur = U[:, :r]
                o[f'agop{r}_full'] = cca2(Hl @ Ur, s.Bf)
                o[f'agop{r}_ho'] = cca2(Hl[te] @ Ur, s.Bf[te])
                ctl_f, ctl_h = [], []
                for k in range(8):
                    R_, _ = np.linalg.qr(rng.normal(size=(Hl.shape[1], r)))
                    ctl_f.append(cca2(Hl @ R_, s.Bf)); ctl_h.append(cca2(Hl[te] @ R_, s.Bf[te]))
                o[f'agop{r}_full_rand'] = float(np.mean(ctl_f)); o[f'agop{r}_ho_rand'] = float(np.mean(ctl_h))
            if len(s.Ws) >= 2:
                trr = s.tr; n = len(trr); masks = [(pr[trr] > 0) for pr in pres]
                Gm1 = np.zeros((s.Ws[0].shape[0], s.Ws[0].shape[0]))
                for c in range(p):
                    g = np.broadcast_to(s.V[c], (n, s.V.shape[1])) * masks[-1]
                    for l in range(len(s.Ws) - 1, 0, -1):
                        g = g @ s.Ws[l]
                        if l - 1 >= 1: g = g * masks[l - 1]
                    Gm1 += g.T @ g
                Gm1 /= n; ev1, U1 = np.linalg.eigh(Gm1); U1 = U1[:, ::-1]; H1 = hs[0]
                for r in (4, p - 1):
                    o[f'agopH1_{r}_ho'] = cca2(H1[te] @ U1[:, :r], s.Bf[te]); o[f'agopH1_{r}_full'] = cca2(H1 @ U1[:, :r], s.Bf)
                    ctl = []
                    for k in range(8):
                        R_, _ = np.linalg.qr(rng.normal(size=(H1.shape[1], r))); ctl.append(cca2(H1[te] @ R_, s.Bf[te]))
                    o[f'agopH1_{r}_ho_rand'] = float(np.mean(ctl))
            Htr = Hl[tr]; mu = Htr.mean(0); Hc = Htr - mu; A = Hc.T @ Hc; lam = 1e-3 * np.trace(A) / A.shape[0] + 1e-12
            Bm = np.linalg.solve(A + lam * np.eye(A.shape[0]), Hc.T @ (s.Y[tr] - s.Y[tr].mean(0)))
            o['refit_acc'] = float((((Hl[te] - mu) @ Bm).argmax(1) == y[te]).mean())
            E = sum(np.abs(np.fft.rfft(Wx, axis=1)[:, 1:]) ** 2 for Wx in (s.Ws[0][:, :p], s.Ws[0][:, p:]))
            tot = float(E.sum()); o['W1_four_top1'] = float(E.max(1).sum() / tot); o['W1_four_top4'] = float(np.sort(E.sum(0))[::-1][:4].sum() / tot)
            o['normW'] = [float(np.linalg.norm(W)) for W in s.Ws]; o['normV'] = float(np.linalg.norm(s.V))
            o['logit_rms'] = float(np.sqrt((F ** 2).mean()))
        return o

# ----------------------------------------------------------------------------- run with checkpoint / resume
def _save_ckpt(path, R, t_next, loss, obs):
    tmp = path + '.tmp.npz'
    np.savez(tmp, t_next=t_next, loss=loss[:t_next], V=R.V, nW=len(R.Ws), obs=json.dumps(obs), **{f'W{l}': W for l, W in enumerate(R.Ws)})
    os.replace(tmp, path)

def run(cfg, out, ckpt_every=5000):
    """Run one job; `out` should end with .json.gz. Resumes from out+'.ckpt.npz' if present."""
    R = Run(cfg); T = cfg['steps']; t0 = time.time(); ck = out + '.ckpt.npz'
    loss = np.empty(T + 1); obs = []; t_start = 0
    if os.path.exists(ck):
        with np.load(ck, allow_pickle=False) as z:  # close the file, so replacing it later leaves no .nfs copy on NFS
            t_start = int(z['t_next'])
            loss[:t_start] = z['loss']; obs = json.loads(str(z['obs'])); R.V = z['V']; R.Ws = [z[f'W{l}'] for l in range(int(z['nW']))]
    for t in range(t_start, T + 1):
        if ckpt_every and t > t_start and t % ckpt_every == 0: _save_ckpt(ck, R, t, loss, obs)
        L, gWs, gV = R.loss_and_grads(); loss[t] = L
        if t % cfg['obs_every'] == 0 or t == T:
            o = R.observe(heavy=(t % cfg['heavy_every'] == 0 or t == T)); o['step'] = t; o['loss'] = L; obs.append(o)
        if t == T: break
        R.step(gWs, gV, t)
    res = dict(cfg=cfg, loss=loss.tolist(), obs=obs, seconds=time.time() - t0, resumed_from=t_start)
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt') as f: json.dump(res, f)
    os.replace(tmp, out)
    if os.path.exists(ck): os.remove(ck)
    return res

def _job(args):
    name, cfg, outdir, ckpt_every = args
    out = os.path.join(outdir, name + '.json.gz')
    if os.path.exists(out): return name, 'exists', 0.0
    t = time.time()
    try:
        run(cfg, out, ckpt_every); msg = 'ok'
    except Exception as e:
        msg = 'ERROR ' + repr(e)
    with open(os.path.join(outdir, 'queue.log'), 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} {msg} {time.time() - t:.0f}s\n")
    return name, msg, time.time() - t

def run_batch(jobs, outdir, workers=4, ckpt_every=5000):
    import multiprocessing as mp
    os.makedirs(outdir, exist_ok=True)
    args = [(j['name'], j['cfg'], outdir, ckpt_every) for j in jobs]
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as pool:
        for name, msg, dt in pool.imap_unordered(_job, args):
            print(f"{name}: {msg} ({dt / 60:.1f} min)", flush=True)

# ----------------------------------------------------------------------------- compact summary (no downloads needed)
def _load(fn):
    with (gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn)) as f: return json.load(f)

def _series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o]
    return np.array([x[0] for x in s]), np.array([x[1] for x in s], dtype=float)

def _at(obs, key, t):
    s, v = _series(obs, key)
    return float(v[int(np.argmin(np.abs(s - t)))]) if len(s) else float('nan')

def _longest_flat(c, start, tol, stride=20):
    from collections import deque
    idx = np.arange(start, len(c), stride); x = c[idx]; best = (0, None, None); j = 0; mx, mn = deque(), deque()
    for i in range(len(x)):
        while mx and x[mx[-1]] <= x[i]: mx.pop()
        mx.append(i)
        while mn and x[mn[-1]] >= x[i]: mn.pop()
        mn.append(i)
        while (x[mx[0]] - x[mn[0]]) / (0.5 * (x[mx[0]] + x[mn[0]])) > tol:
            j += 1
            if mx[0] < j: mx.popleft()
            if mn[0] < j: mn.popleft()
        if i - j > best[0]: best = (i - j, int(idx[j]), int(idx[i]))
    return best[1], best[2]

def summarize(fn, tol=0.03):
    d = _load(fn); c = d['cfg']; L = np.array(d['loss']); obs = d['obs']; p = c['p']; D = c['depth']; lp = math.log(p)
    cm = (L[:-1] + L[1:]) / 2; T = len(L) - 1
    s_, nv = _series(obs, 'normV'); kn = np.where(nv >= 0.99 / c['lam_v'])[0] if c['lam_v'] > 0 else []
    knee = int(s_[kn[0]]) if len(kn) else 0
    a, b = _longest_flat(cm, knee, tol)
    s_, te = _series(obs, 'test_acc'); i10 = np.where(te >= 0.1)[0]; t10 = int(s_[i10[0]]) if len(i10) else None
    t90 = next((int(s_[i]) for i in range(len(te)) if np.all(te[i:] >= 0.9)), None)
    f = lambda k: f"{_at(obs, k, a):.2f}→{_at(obs, k, b):.2f}"
    row = dict(run=os.path.basename(fn).replace('.json.gz', '').replace('.json', ''), flat=f"{a}-{b} ({b - a})", CE=f"{cm[a:b + 1].mean():.3f} ({cm[a:b + 1].mean() / lp:.2f})",
               test=f('test_acc'), refit=f('refit_acc'), agop_h1_top4=f('agopH1_4_ho') if D >= 2 else '-', agop_last_top30=f(f'agop{p - 1}_ho'),
               AH=' / '.join(f(f'AH_full_h{l + 1}') for l in range(D)), W1four=f('W1_four_top1'),
               grok=f"{t10}→{t90}", final=f"{te[-1]:.2f}")
    if t10 is not None:
        seg = cm[t10:(t90 or T) + 1]; row['grok_CE_range'] = f"{100 * (seg.max() - seg.min()) / seg.mean():.1f}%"; row['grok_CE_drift'] = f"{100 * (seg[-1] - seg[0]) / seg.mean():+.1f}%"
    return row

def summary_table(outdir, match=''):
    fns = sorted(glob.glob(os.path.join(outdir, '*.json.gz')) + glob.glob(os.path.join(outdir, '*.json')))
    rows = [summarize(fn) for fn in fns if match in os.path.basename(fn)]
    if not rows: return 'no finished runs'
    keys = list(rows[0].keys()) + [k for k in ('grok_CE_range', 'grok_CE_drift') if k not in rows[0]]
    out = ['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
    for r in rows: out.append('| ' + ' | '.join(str(r.get(k, '-')) for k in keys) + ' |')
    return '\n'.join(out)

def status(outdir):
    lines = []
    for ck in sorted(glob.glob(os.path.join(outdir, '*.ckpt.npz'))):
        z = np.load(ck, allow_pickle=False); lines.append(f"{os.path.basename(ck)[:-9]}: checkpoint at step {int(z['t_next'])}")
    done = sorted(glob.glob(os.path.join(outdir, '*.json.gz')))
    lines.append(f"{len(done)} finished: " + ', '.join(os.path.basename(x)[:-8] for x in done))
    return '\n'.join(lines)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        jobs = json.load(open(sys.argv[2])); outdir = sys.argv[3]
        w = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
        run_batch(jobs, outdir, w)
    elif cmd == 'status': print(status(sys.argv[2]))
    elif cmd == 'summary':
        m = sys.argv[sys.argv.index('--match') + 1] if '--match' in sys.argv else ''
        print(summary_table(sys.argv[2], m))
