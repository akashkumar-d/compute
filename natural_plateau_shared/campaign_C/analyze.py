#!/usr/bin/env python3
"""Campaign-C analysis: collect finished chains from every session branch, compute the preregistered endpoints, write the report.

    python3 analyze.py collect                 # fetch the claude/* branches of origin; copy each finished chain to _collected/<chain>/
    python3 analyze.py report [--from DIR]     # endpoints and aggregates -> ANALYSIS.md and analysis.json (DIR defaults to _collected/)
    python3 analyze.py dev [--runs DIR]        # the primary endpoint on the development cohorts (PREREGISTRATION.md, section 6)
    python3 analyze.py status                  # campaign progress across branches (same as worker.py status --all)

Definitions live in endpoints.py and are stated in PREREGISTRATION.md; this file only aggregates.
"""
import argparse, glob, json, math, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints as E

COLLECTED = os.path.join(HERE, '_collected')
PRIMARY_COHORTS = ['D3', 'D4', 'D5']


# ------------------------------------------------------------------------------------------------------------------ collect
def collect(fetch=True):
    import worker as Wk
    g = Wk.Git(HERE)
    br = g.remote_branches() if fetch else [b.split('refs/remotes/origin/', 1)[1] for b in
                                            g.run('for-each-ref', '--format=%(refname)', 'refs/remotes/origin/').stdout.split() if b.split('refs/remotes/origin/', 1)[1].startswith(Wk.BRANCH_PREFIXES)]
    if fetch: g.fetch(br)
    copies = {}
    for b in br:
        ref = f'refs/remotes/origin/{b}'
        r = g.run('ls-tree', '-r', '--name-only', ref, '--', f'{g.rel}/results', check=False)
        for path in r.stdout.splitlines():
            parts = path.split('/')
            if len(parts) < 3 or parts[-3] != 'results' or parts[-1] not in ('chain.json', 'FAILED.json'): continue
            cid = parts[-2]
            meta = json.loads(g.run('show', f'{ref}:{path}').stdout)
            copies.setdefault(cid, []).append(dict(branch=b, ref=ref, dir=os.path.dirname(path), failed=parts[-1] == 'FAILED.json',
                                                   ran_by=meta.get('ran_by') or meta.get('worker'), owner=meta.get('owner'),
                                                   finished=meta.get('finished') or meta.get('at'), meta=meta))
    os.makedirs(COLLECTED, exist_ok=True)
    index = {}
    for cid, cs in sorted(copies.items()):
        ok = [c for c in cs if not c['failed']]
        pool = ok or cs
        pool.sort(key=lambda c: (c['ran_by'] != c['owner'], c['finished'] or ''))   # the owner's copy, else the earliest
        best = pool[0]; dst = os.path.join(COLLECTED, cid); os.makedirs(dst, exist_ok=True)
        names = ['FAILED.json'] if best['failed'] else ['chain.json', 'pilot.json.gz', 'main.json.gz']
        for n in names:
            fn = os.path.join(dst, n)
            blob = subprocess.run(['git', 'cat-file', 'blob', f"{best['ref']}:{best['dir']}/{n}"], cwd=g.root, capture_output=True, timeout=300)
            if blob.returncode == 0:
                with open(fn + '.tmp', 'wb') as f: f.write(blob.stdout)
                os.replace(fn + '.tmp', fn)
        index[cid] = dict(chosen=best['branch'], failed=best['failed'], copies=[dict(branch=c['branch'], ran_by=c['ran_by'], finished=c['finished'], failed=c['failed']) for c in cs])
    json.dump(index, open(os.path.join(COLLECTED, 'index.json'), 'w'), indent=1)
    print(f'collected {len(index)} chains from {len(br)} branches into {COLLECTED} ({sum(len(v["copies"]) > 1 for v in index.values())} with duplicates)')
    return index


# ------------------------------------------------------------------------------------------------------------------ report
def load_manifest():
    m = json.load(open(os.path.join(HERE, 'manifest.json')))
    m['_by_id'] = {c['id']: c for c in m['chains']}
    return m

def chain_rows(src):
    man = load_manifest(); rows = {}
    for cid, ch in man['_by_id'].items():
        d = os.path.join(src, cid)
        r = dict(id=cid, stage=ch['stage'], cohort=ch['cohort'], seed=ch['seed'], status='missing')
        if os.path.exists(os.path.join(d, 'FAILED.json')) and not os.path.exists(os.path.join(d, 'chain.json')): r['status'] = 'failed'
        if os.path.exists(os.path.join(d, 'chain.json')) and os.path.exists(os.path.join(d, 'main.json.gz')):
            meta = json.load(open(os.path.join(d, 'chain.json')))
            r.update(status='done', meta=dict(pilot_t10=meta['pilot']['t10'], censored=meta['pilot']['censored'], rate=meta['recipe']['rate'],
                                               x_pred=meta['recipe']['x_pred'], ran_by=meta.get('ran_by'), reused_pilot=meta['main'].get('reused_pilot')))
            r['main'] = E.run_fields(E.load(os.path.join(d, 'main.json.gz')))
            if ch['pilot']['kind'] == 'full': r['fixed'] = E.run_fields(E.load(os.path.join(d, 'pilot.json.gz')))
        rows[cid] = r
    return man, rows

def pct(x): return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:.0f}%'
def f2(x, d=2): return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{d}f}'

def verdict_line(k, n_done, n_plan, name):
    lo, hi = E.wilson(k, n_done) if n_done else (float('nan'), float('nan'))
    lo_w, _ = E.wilson(k, n_plan)
    need = E.k_needed(n_plan)
    if n_done == n_plan: v = 'CONFIRMED' if lo >= 0.8 else 'not confirmed'
    else: v = f'{n_plan - n_done} not finished (missing or FAILED); counting them as failures: {"CONFIRMED" if lo_w >= 0.8 else "not confirmed"}'
    return f'| {name} | {k}/{n_done} | {f2(lo)}–{f2(hi)} | {need}/{n_plan} | {v} |'

def report(src, out_md, out_json):
    man, rows = chain_rows(src)
    done = [r for r in rows.values() if r['status'] == 'done']
    L = [f'# Campaign C: analysis', '',
         f'Built {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} from {len(done)}/{len(rows)} finished chains ({src}). '
         f'Definitions: PREREGISTRATION.md and endpoints.py. Manifest sha256 `{E_sha(os.path.join(HERE, "manifest.json"))}`.', '']
    # ---- primary
    L += ['## Primary endpoint (C1 + C2 recipe runs, 40 fresh seeds per depth)', '',
          'A run passes when the peak-band window (cycle-mean CE within 3% of its post-knee peak) holds ≥ 75% of the test 0.1 → 0.9 rise, '
          'the final test accuracy is ≥ 0.99, A_H of the last hidden layer rises by ≥ 0.02 across the window, and every logged training '
          'margin in the window is positive. A depth is confirmed when the 95% Wilson lower bound of the pass rate is ≥ 0.80.', '',
          '| depth | passed / finished | 95% Wilson interval | needed | verdict |', '|---|---|---|---|---|']
    prim = {}
    for co in PRIMARY_COHORTS:
        rs = [r for r in rows.values() if r['cohort'] == co and r['stage'] in ('C1', 'C2')]
        dn = [r for r in rs if r['status'] == 'done']; k = sum(r['main']['primary'] for r in dn)
        prim[co] = dict(k=k, n_done=len(dn), n_plan=len(rs), wilson=E.wilson(k, len(dn)) if dn else None)
        L.append(verdict_line(k, len(dn), len(rs), f'depth {co[1]}'))
    L += ['', 'Failed checks among finished runs (a run can fail several):', '']
    for co in PRIMARY_COHORTS:
        dn = [r for r in rows.values() if r['cohort'] == co and r['stage'] in ('C1', 'C2') and r['status'] == 'done']
        cnt = {k: sum(1 for r in dn if 'primary_checks' in r['main'] and not r['main']['primary_checks'][k]) for k in ('share', 'final', 'F02', 'fit')}
        nk = sum(1 for r in dn if r['main'].get('why_not') == 'no knee')
        L.append(f"- depth {co[1]}: share {cnt['share']}, final {cnt['final']}, F02 {cnt['F02']}, fit {cnt['fit']}" + (f', no knee {nk}' if nk else ''))
    # ---- C2 paired
    L += ['', '## Secondary 1: recipe vs one fixed rate on the same seeds (C2)', '',
          '| depth | fixed rate passes | recipe passes | recipe only / fixed only | exact McNemar p | median abs. drift, fixed → recipe | recipe smaller abs. drift (sign test p) |',
          '|---|---|---|---|---|---|---|']
    for co in PRIMARY_COHORTS:
        pr = [r for r in rows.values() if r['cohort'] == co and r['stage'] == 'C2' and r['status'] == 'done' and 'fixed' in r]
        if not pr: L.append(f'| depth {co[1]} | – | – | – | – | – | – |'); continue
        a = sum(r['fixed']['primary'] for r in pr); b_ = sum(r['main']['primary'] for r in pr)
        b = sum(1 for r in pr if r['main']['primary'] and not r['fixed']['primary']); c = sum(1 for r in pr if r['fixed']['primary'] and not r['main']['primary'])
        dd = [(abs(r['fixed']['drift']), abs(r['main']['drift'])) for r in pr if r['fixed'].get('drift') is not None and r['main'].get('drift') is not None]
        wins = sum(1 for f_, m_ in dd if m_ < f_); ties = sum(1 for f_, m_ in dd if m_ == f_)
        L.append(f"| depth {co[1]} | {a}/{len(pr)} | {b_}/{len(pr)} | {b} / {c} | {f2(E.binom_two_sided(b, b + c), 3)} | "
                 f"{f2(np.median([x[0] for x in dd]), 1)}% → {f2(np.median([x[1] for x in dd]), 1)}% | {wins}/{len(dd) - ties} ({f2(E.binom_two_sided(wins, len(dd) - ties), 3)}) |")
    # ---- C3, C4, C5
    L += ['', '## Secondary 2: other modulus (C3, p = 23, depth 4), width 512 (C4) and the optimizer ablation (C5)', '']
    for st, name in (('C3', 'p = 23, depth 4'), ('C4', 'width 512, depth 4'), ('C5', 'normalized-gradient hidden updates, depth 3')):
        rs = [r for r in rows.values() if r['stage'] == st]; dn = [r for r in rs if r['status'] == 'done']; k = sum(r['main']['primary'] for r in dn)
        wl = ('Wilson %s–%s' % tuple(f2(x) for x in E.wilson(k, len(dn)))) if dn else 'no runs yet'
        L.append(f'- **{st}** ({name}): {k}/{len(dn)} finished runs pass ({wl}); {len(rs) - len(dn)} not finished')
    c5 = {r['seed']: r for r in rows.values() if r['stage'] == 'C5' and r['status'] == 'done'}
    c2 = {r['seed']: r for r in rows.values() if r['stage'] == 'C2' and r['cohort'] == 'D3' and r['status'] == 'done'}
    both = sorted(set(c5) & set(c2))
    if both:
        b = sum(1 for s in both if c2[s]['main']['primary'] and not c5[s]['main']['primary']); c = sum(1 for s in both if c5[s]['main']['primary'] and not c2[s]['main']['primary'])
        L.append(f'- C5 vs the polar recipe runs on the same seeds (C2, depth 3): {len(both)} pairs, polar only {b}, NGD only {c}, exact McNemar p = {f2(E.binom_two_sided(b, b + c), 3)}')
    # ---- clock
    L += ['', '## Secondary 3: the clock', '', '| cohort | target x* | x of the main runs: median (range) | within ±0.25 of x* | CE drift over grokking: median (range) | residual from the registered curve: median |', '|---|---|---|---|---|---|']
    for co, cc in man['cohorts'].items():
        dn = [r for r in rows.values() if r['cohort'] == co and r['status'] == 'done' and r['main'].get('x') is not None]
        if not dn: L.append(f"| {co} | {cc['x']} | – | – | – | – |"); continue
        xs = np.array([r['main']['x'] for r in dn]); dr = np.array([r['main'].get('drift', np.nan) for r in dn]); rs_ = np.array([r['main'].get('drift_resid', np.nan) for r in dn])
        L.append(f"| {co} | {cc['x']} | {np.median(xs):.2f} ({xs.min():.2f}–{xs.max():.2f}) | {np.mean(np.abs(xs - cc['x']) <= 0.25) * 100:.0f}% | "
                 f"{np.nanmedian(dr):+.1f}% ({np.nanmin(dr):+.1f} to {np.nanmax(dr):+.1f}) | {np.nanmedian(rs_):+.1f} points |")
    # ---- protocol secondaries
    L += ['', '## Secondary 4: protocol-v1 fields on the peak-band window, and the canonical AGOP', '',
          '| cohort | runs | joint (strict) | joint (persistent fit) | P10 / P03 over t10→first 0.9 | representation first | AGOP last layer above random | AGOP first layer above random |', '|---|---|---|---|---|---|---|---|']
    for co in man['cohorts']:
        dn = [r['main'] for r in rows.values() if r['cohort'] == co and r['status'] == 'done' and 'W' in r['main']]
        if not dn: continue
        n = len(dn); s_ = lambda k: sum(1 for o in dn if o.get(k)) if any(k in o for o in dn) else '–'
        g_ = lambda k: (f" (median gain over random {np.median([o[k] for o in dn if k in o]):+.2f})" if any(k in o for o in dn) else '')
        L.append(f"| {co} | {n} | {s_('joint_PAG10_F02')} | {s_('joint_PAG10_F02_persistent_fit')} | {s_('P10_transition')} / {s_('P03_transition')} | "
                 f"{s_('representation_first')} | {s_('agop_last_aligned')}{g_('agop_last_gain_over_random')} | {s_('agop_first_aligned')}{g_('agop_first_gain_over_random')} |")
    # ---- provenance
    H = (json.load(open(os.path.join(HERE, 'HASHES.json'))) if os.path.exists(os.path.join(HERE, 'HASHES.json')) else {}).get('files', {})
    mism = []
    for cid, r in sorted(rows.items()):
        if r['status'] != 'done': continue
        code = json.load(open(os.path.join(src, cid, 'chain.json'))).get('code', {})
        for key, fn in (('worker', 'worker.py'), ('runner', 'natplat_campaign.py'), ('recipe', 'recipe.py'), ('manifest', 'manifest.json')):
            if H.get(fn) and code.get(key) and code[key] != H[fn]: mism.append(f'{cid} ({fn})')
    L += ['', '## Provenance', '', f"Registered hashes: {'HASHES.json' if H else 'HASHES.json missing'}. Chains whose recorded code differs from the registered files: "
          + (', '.join(mism) if mism else 'none') + '.']
    # ---- per chain
    L += ['', '## Every chain', '', 'For C2 the last column also gives the fixed-rate arm (the full pilot at r0).', '',
          '| chain | pilot t10 | rate | x | t10 → t90 | window (share) | final test | ΔA_H last | fit | primary |', '|---|---|---|---|---|---|---|---|---|---|']
    for cid in sorted(rows):
        r = rows[cid]
        if r['status'] != 'done': L.append(f'| {cid} | {r["status"]} | | | | | | | | |'); continue
        o = r['main']; m = r['meta']
        L.append(f"| {cid} | {m['pilot_t10'] if not m['censored'] else 'none (censored)'} | {m['rate']} | {f2(o.get('x'))} | {o['t10']} → {o['t90']} | "
                 f"{o['W'][0] if 'W' in o else '–'}–{o['W'][1] if 'W' in o else '–'} ({pct(o.get('share'))}) | {f2(o['final_test'], 3)} | {f2(o.get('dAH_last'), 3)} | "
                 f"{'yes' if o.get('fit_strict') else 'no'} | {'PASS' if o['primary'] else 'fail: ' + (o.get('why_not') or '')}"
                 + (f"; fixed {'PASS' if r['fixed']['primary'] else 'fail'} ({pct(r['fixed'].get('share'))})" if 'fixed' in r else '') + ' |')
    open(out_md, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(primary=prim, rows=rows), open(out_json, 'w'), indent=1, default=float)
    print(f'wrote {out_md} and {out_json}')
    for co, v in prim.items(): print(f"  depth {co[1]}: {v['k']}/{v['n_done']} pass (planned {v['n_plan']})")

def E_sha(fn):
    import hashlib
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ dev check
DEV = {   # development cohorts (runs before the campaign; README v1.2), as used by make_figs_protocol.py
    'D3 (rule 0.5x, 4 seeds)': ['T8_D3_lastslow_r1.0_0.5_tune_T100k', 'T13_D3_lastslow_r1.0_0.5_s0_T100k', 'T13_D3_lastslow_r1.0_0.5_s1_T100k', 'T13_D3_lastslow_r1.0_0.5_s2_T100k'],
    'D4 (per-seed rates, 4 seeds)': ['T11_D4_lastslow_r1.0_1.0_0.35_tune_T120k', 'T14_D4_r1.0_1.0_0.45_s0_T120k', 'T18_D4_r1.0_1.0_0.4_s1_T120k', 'T18_D4_r1.0_1.0_0.4_s2_T120k'],
    'D4 (one rate 0.35x, 4 seeds)': ['T11_D4_lastslow_r1.0_1.0_0.35_tune_T120k', 'T14_D4_r1.0_1.0_0.35_s0_T120k', 'T17_D4_r1.0_1.0_0.35_s1_T120k', 'T17_D4_r1.0_1.0_0.35_s2_T120k'],
    'D5 (recipe, T23)': ['T23_D5_r1.0_1.0_1.0_0.285_s1_T70k', 'T23_D5_r1.0_1.0_1.0_0.300_s2_T70k', 'T23_D5_r1.0_1.0_1.0_0.315_s3_T70k', 'T23_D5_r1.0_1.0_1.0_0.320_s4_T70k'],
    'D5 (one rate 0.35x, T22)': ['T22_D5_r1.0_1.0_1.0_0.35_s1_T70k', 'T22_D5_r1.0_1.0_1.0_0.35_s2_T70k', 'T22_D5_r1.0_1.0_1.0_0.35_s3_T70k', 'T22_D5_r1.0_1.0_1.0_0.35_s4_T70k'],
    'D4 width 512 (T24, 120k)': ['T24_D4_w512_r1.0_1.0_0.4_tune_T120k', 'T24_D4_w512_r1.0_1.0_0.4_s0_T120k'],
}

def dev(runs_dir):
    out = []
    for name, runs in DEV.items():
        res = []
        for r in runs:
            fn = next((f for f in (os.path.join(runs_dir, r + '.json'), os.path.join(runs_dir, r + '.json.gz')) if os.path.exists(f)), None)
            if fn is None: res.append((r, None)); continue
            res.append((r, E.run_fields(E.load(fn))))
        k = sum(1 for _, o in res if o and o['primary']); n = sum(1 for _, o in res if o)
        out.append(f'- **{name}**: {k}/{n} pass. ' + '; '.join(
            f"{r.split('_')[-2] if r.split('_')[-2] != 'tune' else 'tune'} {pct(o['share'])}{'' if o['primary'] else ' (fails: ' + o['why_not'] + ')'}" for r, o in res if o))
    print('\n'.join(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('collect'); p.add_argument('--no-fetch', action='store_true')
    p = sub.add_parser('report'); p.add_argument('--from', dest='src', default=COLLECTED)
    p.add_argument('--out', default=os.path.join(HERE, 'ANALYSIS.md')); p.add_argument('--json', default=os.path.join(HERE, 'analysis.json'))
    p = sub.add_parser('dev'); p.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
    sub.add_parser('status')
    a = ap.parse_args()
    if a.cmd == 'collect': collect(fetch=not a.no_fetch)
    elif a.cmd == 'report': report(a.src, a.out, a.json)
    elif a.cmd == 'dev': dev(a.runs)
    elif a.cmd == 'status':
        import worker as Wk
        print(Wk.campaign_status(Wk.load_manifest(Wk.DEFAULT_MANIFEST)))

if __name__ == '__main__':
    main()
