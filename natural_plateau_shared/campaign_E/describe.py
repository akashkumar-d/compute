#!/usr/bin/env python3
"""Campaign E: descriptive statistics of the development runs (post hoc, not part of PLAN.md; written after the results).

    python3 describe.py [SRC] [--json OUT.json]
SRC is _collected/ (the default) or results_archive/ (the tars are read in place). Prints a markdown table, one row per development
run. The numbers quoted in FINDINGS.md come from this script. Needs numpy; uses endpoints.py and tfm_analyze.py from this folder.

Columns:
  fit         first logged update at which every training pair is classified correctly
  t10, t90    first update with held-out accuracy >= 0.1 / >= 0.9 (first crossing, not the persistent t90 of the endpoint)
  CE rise     two-update mean training CE / log p from t10 to t90 (to the end if 0.9 is never reached): median [10%, 90% quantiles]
  swing       median |log10 L(t+1) - log10 L(t)| after the fit (decades per update)
  spikes      after the fit: excursions of the training CE above 1e-3 * log p (separated by > 200 updates); count, median spacing
  peak        largest two-update mean CE / log p after the fit
  unfit       share of the logged steps after the fit with some training pair misclassified
  |W_U|       unembedding Frobenius norm: at initialization / at the end (the paper's knee is |W_U| >= 0.99 / lambda_V = 4.95)
  logit rms   RMS of the logits at the end of the run
  A_H         Fourier fraction of the last block's MLP activations at "=": at t90 / at the end
  refit90     first heavy step at which the train-only ridge refit reaches 90% held-out accuracy
"""
import glob, gzip, io, json, math, os, sys, tarfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import endpoints as E
import tfm_analyze as A


def runs(src):
    """Yield (id, run dict) for every development run in a folder of run folders or of tars."""
    for d in sorted(glob.glob(os.path.join(src, 'dev_*', 'run.json.gz'))):
        yield os.path.basename(os.path.dirname(d)), A.load(d)
    for tf in sorted(glob.glob(os.path.join(src, '*.tar'))):
        with tarfile.open(tf) as t:
            for m in sorted(t.getmembers(), key=lambda m: m.name):
                parts = m.name.split('/')
                if parts[-1] == 'run.json.gz' and parts[0].startswith('dev_'):
                    yield parts[0], json.loads(gzip.decompress(t.extractfile(m).read()).decode())


def describe(d):
    f = A.fields(d); c = d['cfg']; lp = math.log(c['p']); L = np.asarray(d['loss'], float); cm = E.cycle_mean(L)
    obs = d['obs']; Tf = f['T_fit_first']; out = dict(fit=Tf, t10=f['t10'], t90_first=f['t90_first'], final_test=f['final_test'])
    end = f['t90_first'] if f['t90_first'] is not None else len(cm) - 1
    seg = cm[f['t10']:end + 1] / lp
    out.update(ce_rise_median=float(np.median(seg)), ce_rise_q10=float(np.quantile(seg, 0.1)), ce_rise_q90=float(np.quantile(seg, 0.9)))
    post = L[Tf:] / lp
    out['swing_decades'] = float(np.median(np.abs(np.diff(np.log10(np.maximum(post, 1e-300))))))
    peaks = []
    for i in np.where(post > 1e-3)[0]:
        if not peaks or i - peaks[-1] > 200: peaks.append(int(i))
    out['spikes'] = len(peaks); out['spike_spacing'] = float(np.median(np.diff(peaks))) if len(peaks) > 1 else None
    out['peak_after_fit'] = float(cm[Tf:].max() / lp)
    sm, mtr = E.series(obs, 'train_margin_min'); m = sm >= Tf; out['unfit_share'] = float((mtr[m] <= 0).mean())
    _, nv = E.series(obs, 'normV'); _, lr = E.series(obs, 'logit_rms')
    out.update(normWU_init=float(nv[0]), normWU_min=float(nv.min()), normWU_end=float(nv[-1]), logit_rms_end=float(lr[-1]))
    sa, ah = E.series(obs, f"AH_full_h{c['n_layers']}"); out.update(AH_at_t90=E.at(sa, ah, end), AH_end=float(ah[-1]))
    out.update(refit90=f['refit90'], refit_first=f['refit_first'], why_not=f['why_not'], knee=f['knee'], band=f.get('W'))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    js = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    if js in args: args.remove(js)
    src = args[0] if args else os.path.join(HERE, '_collected')
    rows = {rid: describe(d) for rid, d in runs(src)}
    g = lambda x, fmt='.1e': '–' if x is None else format(x, fmt)
    print('| run | fit | t10 → t90 | CE rise: median [q10, q90] | swing | spikes (spacing) | peak | unfit | |W_U| init / end | logit rms | '
          'A_H t90 / end | refit90 | final |')
    print('|' + '---|' * 13)
    for rid, o in rows.items():
        print(f"| {rid[4:]} | {o['fit']} | {o['t10']} → {o['t90_first'] if o['t90_first'] is not None else 'not reached'} | "
              f"{g(o['ce_rise_median'])} [{g(o['ce_rise_q10'], '.0e')}, {g(o['ce_rise_q90'], '.0e')}] | {o['swing_decades']:.2f} | "
              f"{o['spikes']} ({g(o['spike_spacing'], '.0f')}) | {o['peak_after_fit']:.2g} | {100 * o['unfit_share']:.1f}% | "
              f"{o['normWU_init']:.2f} / {o['normWU_end']:.2f} | {o['logit_rms_end']:.0f} | {o['AH_at_t90']:.2f} / {o['AH_end']:.2f} | "
              f"{g(o['refit90'], 'd') if o['refit90'] is not None else '–'} | {o['final_test']:.3f} |")
    if js: json.dump(rows, open(js, 'w'), indent=1)


if __name__ == '__main__':
    main()
