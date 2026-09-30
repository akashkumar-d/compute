#!/usr/bin/env python3
"""Campaign-E analysis: collect the finished chains from every session branch, compute each run's fields (tfm_analyze.py, PLAN.md)
and report the development stage, the setting chosen for each cell and the fresh-seed passes.

    python3 analyze.py collect                 # fetch the claude/* branches of origin; copy each finished chain to _collected/<id>/
    python3 analyze.py report [--from DIR]     # -> ANALYSIS.md and analysis.json (DIR: _collected/)
    python3 analyze.py status                  # campaign progress across branches (same as worker.py status --all)
"""
import argparse, json, math, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints as E
import tfm_analyze as A

COLLECTED = os.path.join(HERE, '_collected')
FILES = ['chain.json', 'run.json.gz']


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
    os.makedirs(COLLECTED, exist_ok=True); index = {}
    for cid, cs in sorted(copies.items()):
        pool = [c for c in cs if not c['failed']] or cs
        pool.sort(key=lambda c: (c['ran_by'] != c['owner'], c['finished'] or ''))
        best = pool[0]; dst = os.path.join(COLLECTED, cid); os.makedirs(dst, exist_ok=True)
        for n in (['FAILED.json'] if best['failed'] else FILES):
            blob = subprocess.run(['git', 'cat-file', 'blob', f"{best['ref']}:{best['dir']}/{n}"], cwd=g.root, capture_output=True, timeout=300)
            if blob.returncode == 0:
                fn = os.path.join(dst, n)
                with open(fn + '.tmp', 'wb') as f: f.write(blob.stdout)
                os.replace(fn + '.tmp', fn)
        index[cid] = dict(chosen=best['branch'], failed=best['failed'], copies=[dict(branch=c['branch'], ran_by=c['ran_by']) for c in cs])
    json.dump(index, open(os.path.join(COLLECTED, 'index.json'), 'w'), indent=1)
    print(f'collected {len(index)} chains from {len(br)} branches into {COLLECTED}')
    return index


def report(src, out_md, out_json):
    man = json.load(open(os.path.join(HERE, 'manifest.json'))); by_id = {c['id']: c for c in man['chains']}
    rows = {}
    for cid, ch in by_id.items():
        d = os.path.join(src, cid); r = dict(id=cid, stage=ch['stage'], cell=ch['cell'], opt=ch['opt'], n_layers=ch['n_layers'], status='missing')
        if os.path.exists(os.path.join(d, 'chain.json')):
            meta = json.load(open(os.path.join(d, 'chain.json'))); r['meta_selection'] = meta.get('selection')
            if meta.get('skipped'): r.update(status='skipped', reason=meta.get('reason'))
            elif os.path.exists(os.path.join(d, 'run.json.gz')):
                r.update(status='done', f=A.fields(A.load(os.path.join(d, 'run.json.gz'))), dial=meta.get('dial'))
        elif os.path.exists(os.path.join(d, 'FAILED.json')): r['status'] = 'failed'
        rows[cid] = r
    cells = sorted({c['cell'] for c in man['chains']}, key=lambda c: (c[:2], ['paper', 'hybrid', 'adamw'].index(c.split('_', 1)[1])))
    L = ['# Campaign E: transformer pilot at p = 31', '',
         f"{sum(r['status'] in ('done', 'skipped') for r in rows.values())}/{len(rows)} chains finished "
         f"({sum(r['status'] == 'skipped' for r in rows.values())} skipped by the rule, {sum(r['status'] == 'failed' for r in rows.values())} failed). "
         'Endpoint, knee and selection rule: PLAN.md (fixed before any result).', '',
         '## Development stage (tuning seed)', '',
         '| cell | setting | pass | share | CE × log p (band) | raw CE range 10→90% | refit first | t10 → t90 | final test | why not |',
         '|---|---|---|---|---|---|---|---|---|---|']
    fmt = lambda x, f='.2f': '–' if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, f)
    summary = {}
    for cell in cells:
        dev = {cid: r for cid, r in rows.items() if r['cell'] == cell and r['stage'] == 'dev'}
        for cid, r in sorted(dev.items()):
            if r['status'] != 'done': L.append(f"| {cell} | {cid} | {r['status']} | | | | | | | |"); continue
            f = r['f']
            L.append(f"| {cell} | {cid.split('_', 3)[-1]} | {'PASS' if f.get('primary') else 'no'} | {fmt(100 * f.get('share', 0), '.0f')}% | "
                     f"{fmt(f.get('CE_peak_over_logp'))} | {fmt(100 * f['range_transition'], '.1f') + '%' if f.get('range_transition') is not None else '–'} | "
                     f"{'yes' if f.get('refit_first') else 'no'} | {f['t10']} → {f['t90']} | {f['final_test']:.3f} | {f.get('why_not') or ''} |")
        done = {cid: r['f'] for cid, r in dev.items() if r['status'] == 'done'}
        chosen, passed = A.select_setting(done) if done else (None, False)
        fr = [r for r in rows.values() if r['cell'] == cell and r['stage'] == 'fresh']
        k = sum(1 for r in fr if r['status'] == 'done' and r['f'].get('primary')); n = sum(1 for r in fr if r['status'] == 'done')
        summary[cell] = dict(chosen=chosen, dev_passed=passed, fresh_pass=k, fresh_done=n, fresh_skipped=sum(r['status'] == 'skipped' for r in fr),
                             wilson=E.wilson(k, n) if n else None)
    L += ['', '## Chosen setting and fresh seeds', '', '| cell | chosen development setting | passed at development | fresh seeds: pass / run | Wilson 95% |',
          '|---|---|---|---|---|']
    for cell, s in summary.items():
        w = f"[{s['wilson'][0]:.2f}, {s['wilson'][1]:.2f}]" if s['wilson'] else '–'
        run = f"{s['fresh_pass']}/{s['fresh_done']}" if s['fresh_done'] else ('skipped by the rule' if s['fresh_skipped'] else 'not yet')
        L.append(f"| {cell} | {s['chosen']} | {'yes' if s['dev_passed'] else 'no'} | {run} | {w} |")
    L += ['', '## Fresh-seed runs', '', '| run | pass | share | CE × log p | raw CE range 10→90% | refit first | t10 → t90 | final test | why not |',
          '|---|---|---|---|---|---|---|---|---|']
    for cid, r in sorted(rows.items()):
        if r['stage'] != 'fresh' or r['status'] != 'done': continue
        f = r['f']
        L.append(f"| {cid} | {'PASS' if f.get('primary') else 'no'} | {fmt(100 * f.get('share', 0), '.0f')}% | {fmt(f.get('CE_peak_over_logp'))} | "
                 f"{fmt(100 * f['range_transition'], '.1f') + '%' if f.get('range_transition') is not None else '–'} | {'yes' if f.get('refit_first') else 'no'} | "
                 f"{f['t10']} → {f['t90']} | {f['final_test']:.3f} | {f.get('why_not') or ''} |")
    open(out_md, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(summary=summary, rows=rows), open(out_json, 'w'), indent=1, default=str)
    print(f'wrote {out_md} and {out_json}')


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
    else:
        import worker as Wk
        print(Wk.campaign_status(Wk.load_manifest(Wk.DEFAULT_MANIFEST)))


if __name__ == '__main__':
    main()
