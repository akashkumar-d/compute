#!/usr/bin/env python3
"""Campaign-D analysis: collect the finished runs from every session branch, compute each run's fields with campaign C's
definitions (endpoints.py), and write a development report across p.

    python3 analyze.py collect                 # fetch the claude/* branches of origin; copy each finished run to _collected/<id>/
    python3 analyze.py report [--from DIR]     # per-run fields and summaries -> ANALYSIS.md and analysis.json (DIR: _collected/)
    python3 analyze.py status                  # campaign progress across branches (same as worker.py status --all)

Campaign D is a development sweep, not a preregistered test. "Pass" below is campaign C's primary endpoint applied to one run:
the peak band holds >= 75% of the test 0.1 -> 0.9 rise, final test >= 0.99, last-layer A_H rises >= 0.02 across the band, and every
logged training margin in the band is positive. For one hidden layer the "last" hidden layer is the only one.
"""
import argparse, json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints as E

COLLECTED = os.path.join(HERE, '_collected')
FILES = ['chain.json', 'run.json.gz']


# ------------------------------------------------------------------------------------------------------------------ collect
def collect(fetch=True):
    import worker as Wk
    g = Wk.Git(HERE)
    if fetch:
        br, errs = g.fetch_changed(g.remote_heads())
        if errs: print('fetch errors (branches skipped):', '; '.join(errs)[:400])
    else:
        br = [b.split('refs/remotes/origin/', 1)[1] for b in g.run('for-each-ref', '--format=%(refname)', 'refs/remotes/origin/').stdout.split()
              if b.split('refs/remotes/origin/', 1)[1].startswith(Wk.BRANCH_PREFIXES)]
    copies = {}
    for b in br:
        ref = f'refs/remotes/origin/{b}'
        r = g.run('ls-tree', '-r', '--name-only', ref, '--', f'{g.rel}/results', check=False)
        for path in r.stdout.splitlines():
            parts = path.split('/')
            if len(parts) < 3 or parts[-3] != 'results' or parts[-1] not in ('chain.json', 'FAILED.json'): continue
            meta = json.loads(g.run('show', f'{ref}:{path}').stdout)
            copies.setdefault(parts[-2], []).append(dict(branch=b, ref=ref, dir=os.path.dirname(path), failed=parts[-1] == 'FAILED.json',
                                                         ran_by=meta.get('ran_by') or meta.get('worker'), owner=meta.get('owner'),
                                                         finished=meta.get('finished') or meta.get('at')))
    os.makedirs(COLLECTED, exist_ok=True)
    index = {}
    for cid, cs in sorted(copies.items()):
        pool = [c for c in cs if not c['failed']] or cs
        pool.sort(key=lambda c: (c['ran_by'] != c['owner'], c['finished'] or ''))     # the owner's copy, else the earliest
        best = pool[0]; dst = os.path.join(COLLECTED, cid); os.makedirs(dst, exist_ok=True)
        for n in (['FAILED.json'] if best['failed'] else FILES):
            blob = subprocess.run(['git', 'cat-file', 'blob', f"{best['ref']}:{best['dir']}/{n}"], cwd=g.root, capture_output=True, timeout=300)
            if blob.returncode == 0:
                fn = os.path.join(dst, n)
                with open(fn + '.tmp', 'wb') as f: f.write(blob.stdout)
                os.replace(fn + '.tmp', fn)
        index[cid] = dict(chosen=best['branch'], failed=best['failed'],
                          copies=[dict(branch=c['branch'], ran_by=c['ran_by'], finished=c['finished'], failed=c['failed']) for c in cs])
    json.dump(index, open(os.path.join(COLLECTED, 'index.json'), 'w'), indent=1)
    print(f'collected {len(index)} runs from {len(br)} branches into {COLLECTED}')
    return index


# ------------------------------------------------------------------------------------------------------------------ per run
def load_manifest():
    m = json.load(open(os.path.join(HERE, 'manifest.json'))); m['_by_id'] = {c['id']: c for c in m['chains']}; return m

def rows_from(src):
    man = load_manifest(); rows = {}
    for cid, ch in man['_by_id'].items():
        d = os.path.join(src, cid)
        r = dict(id=cid, family=ch['family'], stage=ch['stage'], p=ch['p'], depth=ch['depth'], width=ch['width'], dial=ch['dial'],
                 seed=ch['seed'], status='missing')
        if os.path.exists(os.path.join(d, 'FAILED.json')) and not os.path.exists(os.path.join(d, 'chain.json')): r['status'] = 'failed'
        if os.path.exists(os.path.join(d, 'chain.json')) and os.path.exists(os.path.join(d, 'run.json.gz')):
            meta = json.load(open(os.path.join(d, 'chain.json')))
            f = E.run_fields(E.load(os.path.join(d, 'run.json.gz')))
            f['stopped_early'] = meta['run'].get('stop_reason') == 'generalized'
            f['ran_by'] = meta.get('ran_by'); f['seconds'] = meta['run'].get('seconds'); f['threads'] = meta.get('threads')
            r.update(status='done', f=f)
        rows[cid] = r
    return man, rows


# ------------------------------------------------------------------------------------------------------------------ helpers
def pct(x): return '–' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{100 * x:.0f}%'
def num(x, fmt='.2f'): return '–' if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)
def kstep(x): return '–' if x is None else f'{x / 1000:.1f}k'

def done(rows, **kw):
    out = [r for r in rows.values() if r['status'] == 'done' and all(r[k] == v for k, v in kw.items())]
    return sorted(out, key=lambda r: (json.dumps(r['dial'], sort_keys=True), r['seed']))

def per_seed(rs, fn):
    return ' / '.join(fn(r['f']) for r in sorted(rs, key=lambda r: r['seed']))

def passes(rs): return sum(1 for r in rs if r['f'].get('primary'))

def clock_fit(rs):
    """Least squares drift = a + b*exp(-x) over the runs with a clock; balance point x0 = ln(b / -a) when a < 0 < b."""
    pts = [(r['f']['x'], r['f']['drift']) for r in rs if r['f'].get('x') is not None and r['f'].get('drift') is not None]
    if len(pts) < 3: return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    A = np.stack([np.ones_like(x), np.exp(-x)], 1); (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ np.array([a, b]); r2 = 1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    x0 = math.log(b / -a) if (a < 0 < b) else None
    return dict(a=float(a), b=float(b), r2=float(r2), x0=x0, n=len(pts), x_range=[float(x.min()), float(x.max())])


# ------------------------------------------------------------------------------------------------------------------ report
def report(src, out_md, out_json):
    man, rows = rows_from(src)
    n_done = sum(r['status'] == 'done' for r in rows.values()); n_fail = sum(r['status'] == 'failed' for r in rows.values())
    L = [f'# Campaign D: the natural CE plateau across p (development sweep)', '',
         f'Built from {n_done}/{len(rows)} finished runs ({n_fail} failed) in `{os.path.relpath(src, HERE)}`. Per-run fields use campaign C\'s '
         'definitions (`endpoints.py`); "pass" is campaign C\'s primary endpoint applied to one run (peak band holds >= 75% of the test '
         '0.1 -> 0.9 rise, final test >= 0.99, last-layer ΔA_H >= 0.02 across the band, training margins positive in the band). '
         'This is development: nothing here was registered in advance.', '']
    summary = {}
    # ---- one hidden layer
    L += ['## One hidden layer (width 512): hidden-decay dial at each p', '',
          '| p | λ_W | pass | share of the rise in the peak band | CE level (× log p) | raw CE range in the band | drift over grokking | '
          't10 → t90 | final test | ΔA_H |', '|---|---|---|---|---|---|---|---|---|---|']
    for p in man['grid']['primes']:
        best = None
        for lw in man['grid']['lam_1hl'][str(p)]:
            rs = [r for r in done(rows, family='H', p=p) if abs(r['dial']['lam_w'] - lw) < 1e-12]
            if not rs: L.append(f'| {p} | {lw} | – | not finished | | | | | | |'); continue
            k = passes(rs)
            L.append(f"| {p} | {lw} | {k}/{len(rs)} | {per_seed(rs, lambda f: pct(f.get('share')))} | {per_seed(rs, lambda f: num(f.get('CE_peak_over_logp')))} | "
                     f"{per_seed(rs, lambda f: pct(f.get('v_min')))} | {per_seed(rs, lambda f: num(f.get('drift'), '+.1f') + '%' if f.get('drift') is not None else '–')} | "
                     f"{per_seed(rs, lambda f: kstep(f.get('t10')) + '→' + kstep(f.get('t90')))} | {per_seed(rs, lambda f: num(f.get('final_test')))} | "
                     f"{per_seed(rs, lambda f: num(f.get('dAH_last')))} |")
            score = (k, np.mean([r['f'].get('share') or 0 for r in rs]))
            if best is None or score > best[0]: best = (score, lw, rs)
        if best: summary.setdefault(p, {})['H'] = dict(lam_w=best[1], passes=best[0][0], n=len(best[2]), mean_share=float(best[0][1]),
                                                        ce_level=float(np.mean([r['f'].get('CE_peak_over_logp') or np.nan for r in best[2]])))
    L.append('')
    # ---- width at p = 97
    hw = done(rows, family='HW')
    if hw or any(r['family'] == 'HW' for r in rows.values()):
        L += ['## One hidden layer at p = 97: width', '', '| width | λ_W | pass | share | CE level (× log p) | raw CE range in the band | t10 → t90 | final test | ΔA_H |',
              '|---|---|---|---|---|---|---|---|---|']
        for r in sorted([r for r in rows.values() if r['family'] in ('H', 'HW') and r['p'] == 97 and r['status'] == 'done'],
                        key=lambda r: (r['width'], r['dial']['lam_w'], r['seed'])):
            f = r['f']
            L.append(f"| {r['width']} | {r['dial']['lam_w']} (seed {r['seed']}) | {'yes' if f.get('primary') else 'no'} | {pct(f.get('share'))} | "
                     f"{num(f.get('CE_peak_over_logp'))} | {pct(f.get('v_min'))} | {kstep(f.get('t10'))}→{kstep(f.get('t90'))} | {num(f.get('final_test'))} | {num(f.get('dAH_last'))} |")
        L.append('')
    # ---- deep
    for D in (3, 4):
        L += [f'## Depth {D} (width 256): last-hidden-layer rate dial at each p', '',
              '| p | r | pass | x = η_L·λ_L·t10 | drift over grokking | share | CE level (× log p) | t10 → t90 | final test | ΔA_H last |',
              '|---|---|---|---|---|---|---|---|---|---|']
        fits = []
        for p in man['grid']['deep_primes']:
            allp = done(rows, family='D', depth=D, p=p)
            for r_ in man['grid']['r_dial'][str(D)]:
                rs = [r for r in allp if abs(r['dial']['r'] - r_) < 1e-12]
                if not rs: L.append(f'| {p} | {r_} | – | not finished | | | | | | |'); continue
                L.append(f"| {p} | {r_} | {passes(rs)}/{len(rs)} | {per_seed(rs, lambda f: num(f.get('x')))} | "
                         f"{per_seed(rs, lambda f: num(f.get('drift'), '+.1f') + '%' if f.get('drift') is not None else '–')} | {per_seed(rs, lambda f: pct(f.get('share')))} | "
                         f"{per_seed(rs, lambda f: num(f.get('CE_peak_over_logp')))} | {per_seed(rs, lambda f: kstep(f.get('t10')) + '→' + kstep(f.get('t90')))} | "
                         f"{per_seed(rs, lambda f: num(f.get('final_test')))} | {per_seed(rs, lambda f: num(f.get('dAH_last')))} |")
            fit = clock_fit(allp)
            if allp:
                k = passes(allp); summary.setdefault(p, {})[f'D{D}'] = dict(passes=k, n=len(allp), fit=fit)
            fits.append((p, fit, len(allp)))
        L += ['', f'Clock at depth {D}: drift = a + b·exp(−x) fitted per p over its runs; balance point x₀ where the drift is zero '
              '(campaign C at p = 31: x₀ = 3.92 at depth 3, 4.06 at depth 4).', '']
        for p, fit, n in fits:
            if fit: L.append(f"- p = {p}: a = {fit['a']:+.1f}%, b = {fit['b']:.0f}%, R² = {fit['r2']:.2f}, x₀ = {num(fit['x0'])} ({fit['n']} runs, x {fit['x_range'][0]:.2f}–{fit['x_range'][1]:.2f})")
            else: L.append(f'- p = {p}: fewer than 3 runs with a clock ({n} finished)')
        L.append('')
    # ---- summary across p
    L += ['## Summary across p', '', '| p | 1HL: best λ_W (pass, mean share, CE × log p) | depth 3: runs passing, x₀ | depth 4: runs passing, x₀ |', '|---|---|---|---|']
    for p in man['grid']['primes']:
        s = summary.get(p, {}); h = s.get('H'); d3 = s.get('D3'); d4 = s.get('D4')
        hs = f"{h['lam_w']} ({h['passes']}/{h['n']}, {pct(h['mean_share'])}, {num(h['ce_level'])})" if h else '–'
        ds = lambda d: f"{d['passes']}/{d['n']}, x₀ {num((d['fit'] or {}).get('x0'))}" if d else ('campaign C' if p == 31 else '–')
        L.append(f'| {p} | {hs} | {ds(d3)} | {ds(d4)} |')
    L.append('')
    # ---- every run
    L += ['## Every run', '', '| run | status | steps (horizon) | t10 → t90 | x | peak band (share) | CE × log p | final test | ΔA_H last | pass |', '|---|---|---|---|---|---|---|---|---|---|']
    for cid in sorted(rows):
        r = rows[cid]
        if r['status'] != 'done': L.append(f"| {cid} | {r['status']} | | | | | | | | |"); continue
        f = r['f']; W = f.get('W')
        L.append(f"| {cid} | done{' (stopped early)' if f.get('stopped_early') else ''} | {f.get('horizon')} ({man['_by_id'][cid]['cfg']['steps']}) | "
                 f"{kstep(f.get('t10'))}→{kstep(f.get('t90'))} | {num(f.get('x'))} | {(str(W[0]) + '–' + str(W[1])) if W else '–'} ({pct(f.get('share'))}) | "
                 f"{num(f.get('CE_peak_over_logp'))} | {num(f.get('final_test'))} | {num(f.get('dAH_last'))} | {'PASS' if f.get('primary') else 'fail: ' + str(f.get('why_not'))} |")
    open(out_md, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(summary={str(k): v for k, v in summary.items()}, rows=rows), open(out_json, 'w'), indent=1, default=float)
    print(f'wrote {out_md} and {out_json}: {n_done}/{len(rows)} runs finished')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('collect'); p.add_argument('--no-fetch', action='store_true')
    p = sub.add_parser('report'); p.add_argument('--from', dest='src', default=COLLECTED)
    p.add_argument('--out', default=os.path.join(HERE, 'ANALYSIS.md')); p.add_argument('--json', default=os.path.join(HERE, 'analysis.json'))
    sub.add_parser('status')
    a = ap.parse_args()
    if a.cmd == 'collect': collect(fetch=not a.no_fetch)
    elif a.cmd == 'report': report(a.src, a.out, a.json)
    elif a.cmd == 'status':
        import worker as Wk
        print(Wk.campaign_status(Wk.load_manifest(Wk.DEFAULT_MANIFEST)))


if __name__ == '__main__':
    main()
