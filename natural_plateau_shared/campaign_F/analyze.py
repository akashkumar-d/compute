#!/usr/bin/env python3
"""Campaign-F analysis: collect the finished runs from every session branch, then report every setting at 10 seeds (PLAN.md).

    python3 analyze.py collect                      # fetch the claude/* branches of origin; copy each finished run to _collected/<id>/
    python3 analyze.py report [--from DIR] [--prior-d DIR] [--prior-dev DIR] [--out ANALYSIS.md] [--json analysis.json]
    python3 analyze.py status                       # campaign progress across branches (same as worker.py status --all)

  --from       this campaign's runs (default _collected/): folders <id>/{chain.json, run.json.gz}, or tars of them
  --prior-d    campaign D's runs, for the seeds that already exist: its results_archive/ tars or collected folders
               (default ../campaign_D/results_archive, else ../campaign_D/_collected)
  --prior-dev  the development run archives that hold the existing depth-2 seeds (natural_plateau_v1/runs/*.tar.gz on the Mac)

Per-run fields are campaign C's (endpoints.run_fields, unchanged). "Fresh" seeds are the runs of this campaign: they ran after
PLAN.md fixed the settings. The older seeds are reported with them but not counted as fresh.
"""
import argparse, glob, gzip, hashlib, json, math, os, subprocess, sys, tarfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints as E

COLLECTED = os.path.join(HERE, '_collected')
FILES = ['chain.json', 'run.json.gz']
KEEP = ('W', 'share', 'primary', 'why_not', 't10', 't90', 't90_first', 'CE_peak_over_logp', 'final_test', 'final_train', 'dAH_last',
        'T_fit_first', 'T_fit_persistent', 'fit_relapses', 'knee', 'horizon', 'v_min', 'test_W_start', 'test_W_end')


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


# ------------------------------------------------------------------------------------------------------------------ run files
class Runs:
    """Run files by id in a folder: <id>/run.json.gz (or main.json.gz), <id>.json(.gz), and the same inside .tar / .tar.gz files."""
    def __init__(self, root):
        self.idx, self.failed = {}, set()
        if not root or not os.path.exists(root): return
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            b = os.path.basename(d)
            if os.path.isdir(d):
                for fn in ('run.json.gz', 'main.json.gz'):
                    if os.path.exists(os.path.join(d, fn)): self.idx.setdefault(b, ('file', os.path.join(d, fn)))
                if os.path.exists(os.path.join(d, 'FAILED.json')) and not os.path.exists(os.path.join(d, 'chain.json')): self.failed.add(b)
            elif b.endswith(('.json', '.json.gz')) and b not in ('index.json', 'analysis.json'):
                self.idx.setdefault(b.replace('.json.gz', '').replace('.json', ''), ('file', d))
        for tf in sorted(glob.glob(os.path.join(root, '*.tar')) + glob.glob(os.path.join(root, '*.tar.gz'))):
            with tarfile.open(tf) as t:
                for m in t.getmembers():
                    if not m.isfile(): continue
                    parts = m.name.split('/')
                    if parts[-1] in ('run.json.gz', 'main.json.gz') and len(parts) >= 2: key = parts[-2]
                    elif parts[-1].endswith(('.json', '.json.gz')) and parts[-1] != 'chain.json': key = parts[-1].replace('.json.gz', '').replace('.json', '')
                    else: continue
                    self.idx.setdefault(key, ('tar', tf, m.name))
    def has(self, rid): return rid in self.idx
    def load(self, rid):
        src = self.idx[rid]
        if src[0] == 'file': raw = open(src[1], 'rb').read(); where = src[1]
        else:
            with tarfile.open(src[1]) as t: raw = t.extractfile(src[2]).read()
            where = f'{os.path.basename(src[1])}:{src[2]}'
        if raw[:2] == b'\x1f\x8b': raw = gzip.decompress(raw)
        return json.loads(raw.decode()), dict(where=where, sha256_json=hashlib.sha256(raw).hexdigest())


def clean(x):
    """JSON-safe: numpy scalars to Python, NaN and infinities to None."""
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if math.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    return x


# ------------------------------------------------------------------------------------------------------------------ report
def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def report(src, prior_d, prior_dev, out_md, out_json):
    man = json.load(open(os.path.join(HERE, 'manifest.json'))); by_id = {c['id']: c for c in man['chains']}
    new_runs, d_runs, dev_runs = Runs(src), Runs(prior_d), Runs(prior_dev)
    rows, summary = {}, {}
    for S in man['settings']:
        prior_src = d_runs if S['source'] == 'campaign D' else dev_runs
        entries = [(rid, prior_src, False) for rid in S['prior']] + [(rid, new_runs, True) for rid in S['new']]
        for rid, R, fresh in entries:
            seed = by_id[rid]['seed'] if rid in by_id else int(rid.split('_s')[-1].split('_')[0])
            r = dict(id=rid, setting=S['setting'], family=S['family'], p=S['p'], depth=S['depth'], width=S['width'], dial=S['dial'], seed=seed,
                     fresh=fresh, source='campaign F' if fresh else ('campaign D' if S['source'] == 'campaign D' else 'development runs'),
                     status='missing')
            if R.has(rid):
                d, prov = R.load(rid); f = E.run_fields(d)
                r.update(status='done', file=prov, steps=len(d['loss']) - 1, stop_reason=d.get('stop_reason'), f=clean({k: f.get(k) for k in KEEP}))
            elif rid in R.failed: r['status'] = 'failed'
            rows[rid] = r
        rs = [rows[x] for x in S['prior'] + S['new']]; done = [r for r in rs if r['status'] == 'done']
        fr = [r for r in done if r['fresh']]; k_f = sum(bool(r['f']['primary']) for r in fr); k_a = sum(bool(r['f']['primary']) for r in done)
        why = {}
        for r in done:
            for w in (r['f'].get('why_not') or '').split(','):
                if w.strip(): why[w.strip()] = why.get(w.strip(), 0) + 1
        summary[S['setting']] = dict(
            family=S['family'], p=S['p'], depth=S['depth'], width=S['width'], dial=S['dial'], seeds=len(rs), done=len(done),
            fresh_planned=len(S['new']), fresh_done=len(fr), fresh_pass=k_f, fresh_wilson95=list(E.wilson(k_f, len(fr))) if fr else None,
            all_pass=k_a, all_done=len(done), failed_checks=why,
            median=dict(share=med([r['f'].get('share') for r in done]), CE_peak_over_logp=med([r['f'].get('CE_peak_over_logp') for r in done]),
                        t10=med([r['f'].get('t10') for r in done]), t90=med([r['f'].get('t90') for r in done]),
                        final_test=med([r['f'].get('final_test') for r in done])),
            t90_missing=sum(1 for r in done if r['f'].get('t90') is None))
    # ---- markdown
    n_new = len(man['chains']); n_new_done = sum(1 for r in rows.values() if r['fresh'] and r['status'] == 'done')
    n_fail = sum(1 for r in rows.values() if r['fresh'] and r['status'] == 'failed')
    L = ['# Campaign F: the campaign-D and depth-2 settings at 10 seeds', '',
         f'{n_new_done}/{n_new} new runs finished ({n_fail} failed). Settings, seeds and endpoint: PLAN.md (fixed before any run). '
         'Pass = campaign C\'s per-run endpoint: the loss-only band holds >= 75% of the held-out 10% -> 90% rise, final held-out >= 0.99, '
         'last-layer ΔA_H >= 0.02 across the band, every logged training margin positive in the band. Fresh = run in this campaign.', '',
         '## Per setting', '',
         '| family | p | width | setting | fresh seeds: pass | Wilson 95% | all seeds: pass | median share | median CE / log p | median t10 → t90 | median final | failed checks (all seeds) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    fmt = lambda x, f='.2f': '–' if x is None else format(x, f)
    pct = lambda x: '–' if x is None else f'{100 * x:.0f}%'
    iv = lambda x: '–' if x is None else str(x)
    for key, s in summary.items():
        dial = ', '.join(f"{'λ_W' if k == 'lam_w' else k} {v:g}" for k, v in s['dial'].items())
        w = f"[{s['fresh_wilson95'][0]:.2f}, {s['fresh_wilson95'][1]:.2f}]" if s['fresh_wilson95'] else '–'
        m = s['median']
        t = f"{fmt(m['t10'], '.0f')} → {fmt(m['t90'], '.0f')}" + (f" ({s['t90_missing']} never persistent)" if s['t90_missing'] else '')
        L.append(f"| {s['family']} | {s['p']} | {s['width']} | {dial} | {s['fresh_pass']}/{s['fresh_done']}"
                 + (f" (of {s['fresh_planned']})" if s['fresh_done'] < s['fresh_planned'] else '') + f" | {w} | {s['all_pass']}/{s['all_done']} | "
                 f"{pct(m['share'])} | {fmt(m['CE_peak_over_logp'])} | {t} | {fmt(m['final_test'], '.3f')} | "
                 + (', '.join(f'{k} {v}' for k, v in sorted(s['failed_checks'].items())) or '–') + ' |')
    L += ['', '## Every run', '',
          '| run | source | seed | status | pass | share | band | CE / log p | t10 → t90 | final | ΔA_H last | first fit | why not |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key in summary:
        for r in sorted([x for x in rows.values() if x['setting'] == key], key=lambda x: x['seed']):
            if r['status'] != 'done':
                L.append(f"| {r['id']} | {r['source']} | {r['seed']} | {r['status']} | | | | | | | | | |"); continue
            f = r['f']; W = f.get('W')
            L.append(f"| {r['id']} | {r['source']} | {r['seed']} | done{' (stopped early)' if r.get('stop_reason') == 'generalized' else ''} | "
                     f"{'PASS' if f.get('primary') else 'no'} | {pct(f.get('share'))} | "
                     f"{f'{W[0]}–{W[1]}' if W else '–'} | {fmt(f.get('CE_peak_over_logp'))} | {iv(f.get('t10'))} → {iv(f.get('t90'))} | "
                     f"{fmt(f.get('final_test'), '.3f')} | {fmt(f.get('dAH_last'))} | {iv(f.get('T_fit_first'))} | {f.get('why_not') or ''} |")
    open(out_md, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(campaign='F', plan='PLAN.md', summary=summary, rows=rows), open(out_json, 'w'), indent=1, sort_keys=True)
    print(f'wrote {out_md} and {out_json}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('collect'); p.add_argument('--no-fetch', action='store_true')
    p = sub.add_parser('report'); p.add_argument('--from', dest='src', default=COLLECTED)
    dflt_d = os.path.join(HERE, '..', 'campaign_D', 'results_archive')
    p.add_argument('--prior-d', default=dflt_d if os.path.exists(dflt_d) else os.path.join(HERE, '..', 'campaign_D', '_collected'))
    p.add_argument('--prior-dev', default=None)
    p.add_argument('--out', default=os.path.join(HERE, 'ANALYSIS.md')); p.add_argument('--json', default=os.path.join(HERE, 'analysis.json'))
    sub.add_parser('status')
    a = ap.parse_args()
    if a.cmd == 'collect': collect(fetch=not a.no_fetch)
    elif a.cmd == 'report': report(a.src, a.prior_d, a.prior_dev, a.out, a.json)
    else:
        import worker as Wk
        print(Wk.campaign_status(Wk.load_manifest(Wk.DEFAULT_MANIFEST)))


if __name__ == '__main__':
    main()
