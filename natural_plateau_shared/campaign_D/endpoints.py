"""Preregistered per-run quantities for campaign C (numpy only). Every definition below is fixed before the campaign starts;
PREREGISTRATION.md states them in words and analyze.py aggregates them.

A run file is the runner's output (json or json.gz): cfg, loss (training CE at every step), obs (every 20 steps: accuracies,
margins, held-out loss; every 500 steps also the heavy observers: A_H per layer, AGOP overlaps, refit accuracy, norms, and in the
campaign the v1 observers).
"""
import gzip, json, math
import numpy as np

K1 = 0.03 * 0.015
BAND = 1.03          # peak band: the window keeps every step whose cycle-mean CE is within 3% of the peak (L >= L*/1.03)
SHARE_MIN = 0.75     # primary: at least 75% of the test 0.1 -> 0.9 rise inside the window
FINAL_MIN = 0.99     # primary: final held-out accuracy
DAH_MIN = 0.02       # primary: A_H of the last hidden layer rises by at least 0.02 across the window (protocol F02)
CURVE = (-17.1, 799.0)   # registered clock (log/A.md 2026-09-27 22:50 UTC): drift over grokking = -17.1% + 799% * exp(-x)


def load(fn):
    op = gzip.open if fn.endswith('.gz') else open
    with op(fn, 'rt') as f: return json.load(f)

def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o and o[key] is not None]
    return np.array([x[0] for x in s], dtype=float), np.array([x[1] for x in s], dtype=float)

def at(s, v, t):
    return float(v[int(np.argmin(np.abs(s - t)))]) if len(s) else float('nan')

def first_cross(s, v, thr, after=-1):
    i = np.where((v >= thr) & (s > after))[0]
    return int(s[i[0]]) if len(i) else None

def first_persistent(s, v, thr):
    bad = np.where(v < thr)[0]
    if not len(bad): return int(s[0]) if len(s) else None
    return int(s[bad[-1] + 1]) if bad[-1] + 1 < len(s) else None

def knee_step(cfg, obs):
    s, nv = series(obs, 'normV')
    i = np.where(nv >= 0.99 / cfg['lam_v'])[0]
    return int(s[i[0]]) if len(i) else None

def cycle_mean(L): return (L[:-1] + L[1:]) / 2

def peak_band(L, knee, band=BAND):
    """t* = argmax of the cycle-mean CE after the knee; W = the maximal run of consecutive steps around t* with CE >= CE(t*)/band."""
    cm = cycle_mean(L)
    ts = int(np.argmax(cm[knee:]) + knee); ok = cm >= cm[ts] / band
    a = ts
    while a > 0 and ok[a - 1]: a -= 1
    b = ts
    while b < len(cm) - 1 and ok[b + 1]: b += 1
    return a, b, ts, float(cm[ts])

def share_of_rise(s_te, te, a, b):
    t = lambda x: min(max(at(s_te, te, x), 0.1), 0.9)
    return (t(b) - t(a)) / 0.8

def last_rate(cfg):
    return cfg['eta_layers'][-1] * cfg['lam_layers'][-1] / K1 if 'eta_layers' in cfg else float('nan')


def run_fields(d):
    """All preregistered per-run fields of one run (a main run, or a C2 full pilot used as the fixed-rate arm)."""
    c = d['cfg']; L = np.asarray(d['loss'], dtype=float); obs = d['obs']; T = len(L) - 1; D = c['depth']; p = c['p']
    s, te = series(obs, 'test_acc'); _, tra = series(obs, 'train_acc'); sm, mtr = series(obs, 'train_margin_min')
    out = dict(depth=D, width=c.get('width'), p=p, horizon=T, rate=last_rate(c), hidden_mode=c.get('hidden_mode', 'polar'),
               final_test=float(te[-1]), final_train=float(tra[-1]))
    t10 = first_cross(s, te, 0.1); t90 = first_persistent(s, te, 0.9); tg = first_cross(s, te, 0.9)
    out.update(t10=t10, t90=t90, t90_first=tg)
    out['x'] = out['rate'] * K1 * t10 if t10 is not None else None
    cm = cycle_mean(L)
    if t10 is not None:                                         # grokking transition t10 -> t90 (or the end), as in the dev tables
        g1 = t90 or T; seg = L[t10:g1 + 1]; cs = cycle_mean(seg) if len(seg) > 1 else seg
        out['drift'] = float(100 * (cs[-1] - cs[0]) / cs.mean()); out['range_min'] = float(100 * (seg.max() - seg.min()) / seg.min())
        if out['x'] is not None: out['drift_pred'] = CURVE[0] + CURVE[1] * math.exp(-out['x']); out['drift_resid'] = out['drift'] - out['drift_pred']
    k = knee_step(c, obs); out['knee'] = k
    if k is None or k >= len(cm):
        out['primary'] = False; out['why_not'] = 'no knee'; return out
    a, b, ts, Ls = peak_band(L, k)
    sh = share_of_rise(s, te, a, b)
    seg = L[a:b + 1]
    ahD = series(obs, f'AH_full_h{D}'); ah1 = series(obs, 'AH_full_h1')
    dAH = at(*ahD, b) - at(*ahD, a); dAH1 = at(*ah1, b) - at(*ah1, a)
    m = (sm >= a) & (sm <= b); fit_strict = bool(m.any() and np.all(mtr[m] > 0))
    out.update(W=[a, b], W_len=b - a, t_peak=ts, CE_peak=Ls, CE_peak_over_logp=Ls / math.log(p), share=float(sh),
               v_min=float((seg.max() - seg.min()) / seg.min()), W_drift=float((seg[-1] - seg[0]) / seg.mean()),
               dAH_last=float(dAH), dAH_first=float(dAH1), fit_strict=fit_strict, test_W_start=at(s, te, a), test_W_end=at(s, te, b))
    checks = dict(share=sh >= SHARE_MIN, final=out['final_test'] >= FINAL_MIN, F02=dAH >= DAH_MIN, fit=fit_strict)
    out['primary_checks'] = checks; out['primary'] = bool(all(checks.values()))
    out['why_not'] = ', '.join(k_ for k_, v_ in checks.items() if not v_) or None
    # ---- protocol-v1 fields on W (as in make_figs_protocol.run_fields, with W in place of the outcome-aware window)
    pos = mtr > 0
    Tfit = int(sm[np.argmax(pos)]) if pos.any() else None
    relapse = sm[(sm > (Tfit or 0)) & ~pos]
    Tfit_persist = int(sm[np.where(~pos)[0].max() + 1]) if (~pos).any() and np.where(~pos)[0].max() + 1 < len(sm) else Tfit
    Dly = max(1000, math.ceil(0.02 * T))
    def bad_prefix(t0):
        if t0 is None: return False
        mm = (s >= t0) & (s <= t0 + Dly)
        return bool(mm.any() and np.all(te[mm] <= 0.2) and s[mm].max() >= t0 + Dly - 20)
    sr, rf = series(obs, 'refit_acc')
    r0 = at(sr, rf, a); Tr_in = first_cross(sr, rf, 0.9, after=a)
    useful = bool(r0 <= 0.70 and Tr_in is not None and Tr_in <= b and at(sr, rf, Tr_in) - r0 >= 0.20)
    grok_in = bool(tg is not None and a <= tg <= b)
    persists = bool(Tfit is not None and not np.any(~pos[(sm > Tfit) & (sm <= b)]))
    core = bool(out['v_min'] <= 0.10 and dAH >= DAH_MIN and fit_strict and out['test_W_start'] <= 0.20 and grok_in)
    joint = bool(core and persists and bad_prefix(Tfit)); joint_pers = bool(core and bad_prefix(Tfit_persist))
    lead = bool(useful and Tr_in is not None and tg is not None and a < Tr_in < tg and at(s, te, Tr_in) <= 0.20)
    out.update(T_fit_first=Tfit, fit_relapses=int(len(relapse)), T_fit_persistent=Tfit_persist, grok_inside_W=grok_in,
               P10_W=bool(out['v_min'] <= 0.10), P03_W=bool(out['v_min'] <= 0.03), joint_PAG10_F02=joint,
               joint_PAG10_F02_persistent_fit=joint_pers, useful_refit=useful, representation_first=bool(joint and lead),
               representation_first_persistent_fit=bool(joint_pers and lead), T_refit90_after_W_start=Tr_in)
    if t10 is not None and tg is not None:                      # transition window [t10, first 0.9]
        tr = L[t10:tg + 1]; v = float((tr.max() - tr.min()) / tr.min())
        out.update(v_min_transition=v, P10_transition=bool(v <= 0.10), P03_transition=bool(v <= 0.03))
    # ---- canonical (class-centered) AGOP vs 8 fixed random frames, v1 observers (campaign runs only)
    for name in ('last', 'first'):
        key = f'v1_{name}_agop_purity'
        sp, pv = series(obs, key)
        if not len(sp): continue
        _, rmed = series(obs, f'v1_{name}_agop_rand_med'); _, rmax = series(obs, f'v1_{name}_agop_rand_max')
        dp = at(sp, pv, b) - at(sp, pv, a); dr = at(sp, rmed, b) - at(sp, rmed, a)
        out[f'agop_{name}_purity_W'] = [at(sp, pv, a), at(sp, pv, b)]; out[f'agop_{name}_rand_med_W'] = [at(sp, rmed, a), at(sp, rmed, b)]
        out[f'agop_{name}_gain_over_random'] = float(dp - dr)
        out[f'agop_{name}_aligned'] = bool(dp - dr > 0 and at(sp, pv, b) > at(sp, rmax, b))   # rises faster than random frames and ends above all 8
        _, ah = series(obs, f'v1_{name}_AH'); _, ad = series(obs, f'v1_{name}_Adiff')
        if len(ah): out[f'v1_{name}_AH_W'] = [at(sp, ah, a), at(sp, ah, b)]; out[f'v1_{name}_Adiff_W'] = [at(sp, ad, a), at(sp, ad, b)]
    sp, cert = series(obs, 'v1_signed_certified')
    if len(sp): out['signed_certified_W'] = [at(sp, cert, a), at(sp, cert, b)]
    return out


# ------------------------------------------------------------------------------------------------------------------ statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float('nan'), float('nan'))
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)

def k_needed(n, lb=0.8):
    return next((k for k in range(n + 1) if wilson(k, n)[0] >= lb), None)

def binom_two_sided(k, n):
    """Exact two-sided sign/McNemar p-value: P(X <= min(k, n-k)) * 2 for X ~ Bin(n, 1/2), capped at 1."""
    if n == 0: return 1.0
    m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


if __name__ == '__main__':
    import sys
    for fn in sys.argv[1:]:
        o = run_fields(load(fn))
        print(fn.split('/')[-1], 'primary', o['primary'], o.get('why_not'), 'share %.2f' % o.get('share', float('nan')), 'x', o.get('x'), 'drift', o.get('drift'))
