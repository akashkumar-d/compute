#!/usr/bin/env python3
"""Campaign E: per-run fields (PLAN.md: campaign C's endpoint with the arm-specific knee) and the selection rule.

    python3 tfm_analyze.py DIR [DIR ...] [--json OUT.json]
Uses endpoints.py (a copy of campaign D's, the same definitions as campaign C) for the band, the share and the helpers.
"""
import glob, gzip, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import endpoints as E


def fields(d):
    c = d['cfg']; L = np.asarray(d['loss'], float); obs = d['obs']; p = c['p']; D = c['n_layers']; T = len(L) - 1
    s, te = E.series(obs, 'test_acc'); _, tra = E.series(obs, 'train_acc'); sm, mtr = E.series(obs, 'train_margin_min')
    out = dict(opt=c['opt'], depth=D, p=p, horizon=T, final_test=float(te[-1]), final_train=float(tra[-1]), stop=d.get('stop_reason'))
    t10 = E.first_cross(s, te, 0.1); t90 = E.first_persistent(s, te, 0.9); tg = E.first_cross(s, te, 0.9)
    pos = mtr > 0; Tfit = int(sm[np.argmax(pos)]) if pos.any() else None
    out.update(t10=t10, t90=t90, t90_first=tg, T_fit_first=Tfit, CE_at_t10_over_logp=float(L[t10] / math.log(p)) if t10 is not None else None)
    if t10 is not None:
        g1 = tg if tg is not None else T; seg = L[t10:g1 + 1]
        out['range_transition'] = float((seg.max() - seg.min()) / seg.min())
        cs = E.cycle_mean(seg) if len(seg) > 1 else seg; out['drift_transition'] = float((cs[-1] - cs[0]) / cs.mean())
    sr, rf = E.series(obs, 'refit_acc'); tr90 = E.first_cross(sr, rf, 0.9)
    out['refit90'] = tr90; out['refit_first'] = bool(tr90 is not None and (tg is None or tr90 < tg))
    if c['opt'] == 'paper': knee = E.knee_step(c, obs); out['knee_kind'] = 'head norm cap'
    else: knee = Tfit; out['knee_kind'] = 'first fit'
    out['knee'] = knee
    cm = E.cycle_mean(L)
    if knee is None or knee >= len(cm):
        out.update(primary=False, why_not='no knee'); return out
    a, b, ts, Ls = E.peak_band(L, knee); sh = E.share_of_rise(s, te, a, b); seg = L[a:b + 1]
    ah = E.series(obs, f'AH_full_h{D}'); dAH = E.at(*ah, b) - E.at(*ah, a)
    m = (sm >= a) & (sm <= b); fit = bool(m.any() and np.all(mtr[m] > 0))
    out.update(W=[a, b], share=float(sh), CE_peak_over_logp=Ls / math.log(p), v_band=float((seg.max() - seg.min()) / seg.min()),
               dAH_last=float(dAH), fit_strict=fit, test_W_start=E.at(s, te, a), test_W_end=E.at(s, te, b))
    checks = dict(share=sh >= E.SHARE_MIN, final=out['final_test'] >= E.FINAL_MIN, F02=dAH >= E.DAH_MIN, fit=fit)
    out['primary'] = bool(all(checks.values())); out['why_not'] = ', '.join(k for k, v in checks.items() if not v) or None
    return out


def select_setting(dev):
    """PLAN.md's rule. dev: {development chain id: fields}. Returns (chosen id, whether it passed).
    Among the passing settings, the smallest raw-CE range over the 10% -> 90% held-out rise; if none passes, the largest band
    share among the runs with final held-out accuracy >= 0.99 (else among all). Ties go to the first id in sorted order."""
    ids = sorted(dev)
    passed = [k for k in ids if dev[k].get('primary')]
    if passed:
        return min(passed, key=lambda k: (dev[k].get('range_transition', float('inf')), ids.index(k))), True
    pool = [k for k in ids if dev[k].get('final_test', 0) >= 0.99] or ids
    return max(pool, key=lambda k: (dev[k].get('share', 0.0), -ids.index(k))), False


def load(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; js = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    if js in args: args.remove(js)
    rows = {}
    for dd in args:
        for fn in sorted(glob.glob(os.path.join(dd, '*.json.gz'))):
            rows[os.path.basename(fn)[:-8]] = fields(load(fn))
    hdr = f"{'run':34s} {'opt':6s} D  fit@   t10→t90          CE@t10/logp range10→90 refit1st band           share  CE/logp dAH    final  pass"
    print(hdr)
    for k, f in rows.items():
        W = f.get('W'); band = f'{W[0]}-{W[1]}' if W else '-'
        t = f"{f['t10']}→{f['t90'] if f['t90'] is not None else '-'}" if f['t10'] is not None else '-'
        rt = f"{100 * f['range_transition']:.1f}%" if f.get('range_transition') is not None else '-'
        ce10 = f"{f['CE_at_t10_over_logp']:.2f}" if f.get('CE_at_t10_over_logp') is not None else '-'
        print(f"{k:34s} {f['opt']:6s} {f['depth']}  {str(f['T_fit_first']):6s} {t:16s} {ce10:11s} {rt:10s} {str(f['refit_first']):8s} "
              f"{band:14s} {100 * f.get('share', 0):5.0f}% {f.get('CE_peak_over_logp', float('nan')):6.2f} {f.get('dAH_last', float('nan')):6.3f} "
              f"{f['final_test']:.3f}  {'PASS' if f.get('primary') else 'fail: ' + str(f.get('why_not'))}")
    if js: json.dump(rows, open(js, 'w'), indent=1)


if __name__ == '__main__':
    main()
