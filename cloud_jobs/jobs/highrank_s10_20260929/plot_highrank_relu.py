#!/usr/bin/env python3
"""Publication figures for the ten-fresh-seed high-rank ReLU cohort (saved data only).

Inputs
  --data DIR     collected cohort: DIR/data/<arm>/{loss_every_step.npy, rows.jsonl, ...}
                 and canonical per-arm records (DIR/canonical/*.json[.gz])
  --confirm DIR  the four completed confirmation arms (h2, r16, seeds 9351-9354) with RECORDS.json
Outputs
  highrank_relu_rank_sweep.{pdf,png}   h2 teacher, r = 2, 4, 8, 16 (10 seeds each)
  highrank_relu_teachers_r8.{pdf,png}  r = 8: h2, |z|, exp(-z^2/2) (10 seeds each)
  PLOT_DATA_relu.json                  per-cell counts, plateau ends and plotted summaries

Statistics: medians over all ten seeds and pointwise 10th-90th percentile bands, only on
the support shared by all ten; later individual tails stay as thin lines. Loss is recorded
at every update; diagnostics are linearly interpolated between saved checkpoints for display
only. Qualification counts and plateau endpoints come from the canonical per-arm records.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

C = {'loss': '#25395B', 'min': '#007C83', 'mean': '#BB7518', 'refit': '#814B9C',
     'plateau': '#DDEDEC', 'boundary': '#598C89', 'grid': '#E2E7EC', 'muted': '#505A60',
     'chance': '#8A949B'}
QUALIFIED = ('qualified_candidate_observed',)
SCALE = 10.0  # display: log(1 + n / SCALE)


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif', 'font.size': 7.5, 'axes.labelsize': 7.4,
                         'axes.titlesize': 7.6, 'xtick.labelsize': 6.4, 'ytick.labelsize': 6.6,
                         'legend.fontsize': 6.4, 'axes.linewidth': .55, 'lines.linewidth': 1.15,
                         'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'path.simplify': False})


# ------------------------------------------------------------------ data
def load_canonical(paths):
    recs = {}
    for p in paths:
        p = Path(p)
        obj = json.load(gzip.open(p, 'rt')) if p.suffix == '.gz' else json.loads(p.read_text())
        if 'record' in obj:
            recs[obj['record']['id']] = obj['record']
        for a in obj.get('arms', []):
            recs[a['id']] = a
    return recs


def load_run(folder, canon):
    folder = Path(folder)
    arm = folder.name
    loss = np.load(folder / 'loss_every_step.npy').astype(float)
    rows_path = folder / 'rows.jsonl'
    opener = open if rows_path.exists() else (lambda p, m: gzip.open(str(p) + '.gz', m))
    rows = {}
    with opener(rows_path, 'rt') as fh:
        for line in fh:
            if line.strip():
                x = json.loads(line)
                rows[int(x['step'])] = x
    steps = np.array(sorted(s for s in rows if s < len(loss)))
    get = lambda k: np.array([np.nan if rows[s].get(k) is None else rows[s][k] for s in steps], float)
    r = len(rows[steps[0]]['principal_cosines_sq_ascending'])
    pc = np.array([rows[s]['principal_cosines_sq_ascending'] for s in steps], float)
    rec = canon[arm]
    var = float(rec.get('target_variance') or rows[steps[0]].get('loss_baseline') or 1.0)
    pre = rec['prefixes']
    first = lambda t: (pre[t].get('first_qualified_complete_sequence') or {}).get('step')
    return dict(id=arm, seed=int(arm.rsplit('seed', 1)[1]), r=r, loss=loss / var, var=var,
                ck=steps, A_min=get('A_min'), A_mean=get('A_mean'), refit=get('refit') / var,
                pcos=pc, p1=pre['1.01']['end_step'], p5=pre['1.05']['end_step'],
                q1=pre['1.01']['numerical_qualification_assessment'],
                q5=pre['1.05']['numerical_qualification_assessment'], seq1=first('1.01'), seq5=first('1.05'),
                valid=np.array([bool(rows[s].get('agop_valid', True)) for s in steps]))


# ------------------------------------------------------------------ summaries
def display_grid(runs, horizon):
    g = np.unique(np.concatenate([np.round(np.expm1(np.linspace(0, np.log1p(horizon / SCALE), 900)) * SCALE),
                                  *[r['ck'] for r in runs]]))
    return g[g <= horizon]


def interp(run, key, grid):
    x = run['ck']
    y = run[key]
    ok = np.isfinite(y) if y.ndim == 1 else np.all(np.isfinite(y), axis=1)
    x, y = x[ok], y[ok]
    out = np.full((len(grid),) + y.shape[1:], np.nan)
    inside = (grid >= x[0]) & (grid <= x[-1])
    if y.ndim == 1:
        out[inside] = np.interp(grid[inside], x, y)
    else:
        for j in range(y.shape[1]):
            out[inside, j] = np.interp(grid[inside], x, y[:, j])
    return out


def band(stack):
    return (np.nanmedian(stack, axis=0), np.nanpercentile(stack, 10, axis=0), np.nanpercentile(stack, 90, axis=0))


# ------------------------------------------------------------------ drawing
def fwd(x):
    return np.log1p(np.maximum(np.asarray(x, float), -SCALE * .999) / SCALE)


def inv(y):
    return SCALE * np.expm1(y)


def xaxis(ax, horizon, show_labels):
    ax.set_xscale('function', functions=(fwd, inv))
    ax.set_xlim(-.4, horizon)
    ticks = [t for t in (0, 100, 1000, 3000) if t <= horizon]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v >= 1000 else f'{v:g}'))
    ax.xaxis.set_minor_locator(FixedLocator([t for t in (10, 30, 300) if t <= horizon]))
    ax.tick_params(axis='x', which='minor', length=1.5, width=.4)
    if not show_labels:
        ax.tick_params(labelbottom=False)


def yaxis(ax):
    ax.set_ylim(-.035, 1.055)
    ax.set_yticks([0, .5, 1])
    ax.grid(axis='y', color=C['grid'], lw=.45)
    ax.set_axisbelow(True)


def plateau(ax, end):
    ax.axvspan(0, end, color=C['plateau'], alpha=.75, lw=0, zorder=0)
    ax.axvline(end, color=C['boundary'], lw=.7, ls=(0, (2, 2)), zorder=5)


def draw_series(ax, grid, med, lo, hi, color, ls='-', lw=1.2):
    ax.fill_between(grid, lo, hi, color=color, alpha=.16, lw=0, zorder=2)
    ax.plot(grid, med, color=color, lw=lw, ls=ls, zorder=4)


def render(cells, title, subtitle, out_stem, horizon=3000):
    style()
    n = len(cells)
    width = 5.5
    fig = plt.figure(figsize=(width, 5.35))
    left, right = .115, .905 if n > 3 else .89
    top = .818 if any(c.get('formula') for c in cells) else .838
    gs = fig.add_gridspec(4, n, left=left, right=right, top=top, bottom=.175, hspace=.2, wspace=.16,
                          height_ratios=[1, 1, .95, 1])
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(4)])
    fig.text((left + right) / 2, .985, title, ha='center', va='top', fontsize=8, color=C['loss'])
    fig.text((left + right) / 2, .96, subtitle, ha='center', va='top', fontsize=6.2, color=C['muted'],
             linespacing=1.35)
    summary = {}
    heat_artist = None
    for j, cell in enumerate(cells):
        runs = cell['runs']
        k = len(runs)
        r = runs[0]['r']
        common_loss = min(len(x['loss']) for x in runs) - 1
        common_diag = min(int(x['ck'][-1]) for x in runs)
        p5 = min(x['p5'] for x in runs)
        p1 = min(x['p1'] for x in runs)
        q1 = sum(x['q1'] in QUALIFIED for x in runs)
        q5 = sum(x['q5'] in QUALIFIED for x in runs)
        s1 = sum(x['seq1'] is not None for x in runs)
        s5 = sum(x['seq5'] is not None for x in runs)
        pos = axes[0, j].get_position()
        cx = (pos.x0 + pos.x1) / 2
        y = .898
        fig.text(cx, y, cell['title'], ha='center', va='center', fontsize=7.6, weight='bold')
        if cell.get('formula'):
            y -= .026
            fig.text(cx, y, cell['formula'], ha='center', va='center', fontsize=7.0)
        qual = f'{q1}/{k} qualify (1% and 5%)' if q1 == q5 else f'{q1}/{k} (1%), {q5}/{k} (5%) qualify'
        fig.text(cx, y - .024, f'plateau 0–{p5} updates', ha='center', va='center', fontsize=5.6, color='#356763')
        fig.text(cx, y - .042, qual, ha='center', va='center', fontsize=5.6, color='#356763')
        # --- row 0: loss
        ax = axes[0, j]
        xaxis(ax, horizon, False); yaxis(ax); plateau(ax, p5)
        for x in runs:
            ax.plot(np.arange(len(x['loss'])), x['loss'], color=C['loss'], lw=.4, alpha=.22, zorder=3)
        steps = np.arange(common_loss + 1)
        med, lo, hi = band(np.stack([x['loss'][:common_loss + 1] for x in runs]))
        draw_series(ax, steps, med, lo, hi, C['loss'])
        ax.plot([0], [med[0]], marker='o', ms=2.6, mfc='white', mec=C['loss'], mew=.65, zorder=6)
        # --- row 1: A_min, A_mean, chance
        grid = display_grid(runs, common_diag)
        ax = axes[1, j]
        xaxis(ax, horizon, False); yaxis(ax); plateau(ax, p5)
        ax.axhline(r / 64, color=C['chance'], lw=.6, ls=(0, (1, 1.6)), zorder=1)
        am, al, ah = band(np.stack([interp(x, 'A_mean', grid) for x in runs]))
        draw_series(ax, grid, am, al, ah, C['mean'], ls=(0, (4, 1.6)), lw=1.05)
        mm, ml, mh = band(np.stack([interp(x, 'A_min', grid) for x in runs]))
        draw_series(ax, grid, mm, ml, mh, C['min'])
        for x in runs:  # tails beyond the shared support
            tail = x['ck'] > common_diag
            if tail.any():
                ax.plot(np.r_[common_diag, x['ck'][tail]], np.r_[np.interp(common_diag, x['ck'], x['A_min']),
                        x['A_min'][tail]], color=C['min'], lw=.4, alpha=.25, zorder=3)
        # --- row 2: per-direction heatmap (median of k-th smallest cos^2)
        ax = axes[2, j]
        xaxis(ax, horizon, False)
        pcs = np.stack([interp(x, 'pcos', grid) for x in runs])  # seeds x grid x r
        pmed = np.nanmedian(pcs, axis=0)
        gx = fwd(grid)
        mids = inv((gx[1:] + gx[:-1]) / 2)
        edges = np.r_[grid[0] - (mids[0] - grid[0]), mids, grid[-1] + (grid[-1] - mids[-1])]
        heat_artist = ax.pcolormesh(edges, np.arange(r + 1) + .5, pmed.T, cmap='Blues', vmin=0, vmax=1,
                                    shading='flat', rasterized=True, zorder=1)
        ax.axvline(p5, color='white', lw=.8, ls=(0, (2, 2)), zorder=5)
        ax.set_ylim(.5, r + .5)
        ax.set_yticks([1, r] if r > 2 else [1, 2])
        ax.tick_params(axis='y', length=0, labelsize=5.8, pad=1.5)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        # --- row 3: refit
        ax = axes[3, j]
        xaxis(ax, horizon, True); yaxis(ax); plateau(ax, p5)
        fm, fl, fh = band(np.stack([interp(x, 'refit', grid) for x in runs]))
        draw_series(ax, grid, fm, fl, fh, C['refit'])
        ax.plot([grid[0]], [fm[0]], marker='o', ms=2.6, mfc='white', mec=C['refit'], mew=.65, zorder=6)
        for x in runs:
            tail = x['ck'] > common_diag
            if tail.any():
                ax.plot(np.r_[common_diag, x['ck'][tail]], np.r_[np.interp(common_diag, x['ck'], x['refit']),
                        x['refit'][tail]], color=C['refit'], lw=.4, alpha=.25, zorder=3)
        if j:
            for i in (0, 1, 3):  # heatmap rows keep their own 1..r labels
                axes[i, j].tick_params(labelleft=False)
        i0 = int(np.searchsorted(grid, p5))
        summary[cell['key']] = dict(
            seeds=[x['seed'] for x in runs], rank=r, qualified_1pct=q1, qualified_5pct=q5,
            qualified_sequence_1pct=s1, qualified_sequence_5pct=s5, common_plateau_end_5pct=p5,
            common_plateau_end_1pct=p1, individual_plateau_end_5pct=[x['p5'] for x in runs],
            loss_common_support_updates=common_loss, diagnostics_common_support_updates=common_diag,
            median_at_common_plateau_end=dict(loss=float(med[min(p5, common_loss)]), A_min=float(mm[min(i0, len(mm) - 1)]),
                                              A_mean=float(am[min(i0, len(am) - 1)]), refit=float(fm[min(i0, len(fm) - 1)])),
            median_initial=dict(A_min=float(mm[0]), A_mean=float(am[0]), refit=float(fm[0])),
            first_qualified_sequence_step_5pct=[x['seq5'] for x in runs],
            final_loss_over_var=[float(x['loss'][-1]) for x in runs])
    labels = ['Training loss\nMSE / Var(Y)', 'AGOP\nalignment', 'Each teacher\ndirection $k$\n(sorted)',
              'Refit error\nMSE / Var(Y)']
    for i, lab in enumerate(labels):
        axes[i, 0].set_ylabel(lab, labelpad=4 if i != 2 else 6)
    cax = fig.add_axes([right + .012, axes[2, -1].get_position().y0, .011, axes[2, -1].get_position().height])
    cb = fig.colorbar(heat_artist, cax=cax, ticks=[0, .5, 1])
    cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.ax.set_title(r'$\cos^2\theta_k$', fontsize=6, pad=3)
    fig.text((left + right) / 2, .128, 'GD updates (early time expanded; same scale in every column)',
             ha='center', fontsize=7)
    handles = [Line2D([], [], color=C['loss'], lw=1.2, label='Loss'),
               Line2D([], [], color=C['min'], lw=1.2, label=r'$A_{\min}$: weakest direction'),
               Line2D([], [], color=C['mean'], lw=1.05, ls=(0, (4, 1.6)), label=r'$A_{\rm mean}$'),
               Line2D([], [], color=C['chance'], lw=.8, ls=(0, (1, 1.6)), label=r'$r/d$ (random subspace)'),
               Line2D([], [], color=C['refit'], lw=1.2, label='Refit error'),
               Patch(facecolor=C['plateau'], label='All-seed loss plateau')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=((left + right) / 2, .083), ncol=6, frameon=False,
               handlelength=1.5, columnspacing=.9, handletextpad=.4, fontsize=6.1)
    fig.text((left + right) / 2, .038, 'Bold: median of 10 seeds · band: 10th–90th percentile · thin: individual '
             'seeds beyond shared support · heatmap: median of the $k$-th smallest squared principal cosine '
             'between the teacher subspace and the top-$r$ AGOP eigenspace', ha='center', fontsize=5.3,
             color=C['muted'], wrap=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', facecolor='white',
                    metadata={'Creator': 'plot_highrank_relu.py'} if ext == 'pdf' else None)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--confirm', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    canon = load_canonical(sorted(glob.glob(str(a.data / 'canonical' / '*.json*'))) + [a.confirm / 'RECORDS.json'])
    runs = {}
    for folder in sorted((a.data / 'data').iterdir()):
        runs[folder.name] = load_run(folder, canon)
    for folder in sorted(p for p in a.confirm.iterdir() if p.is_dir()):
        runs[folder.name] = load_run(folder, canon)

    def cell(prefix, seeds=range(9351, 9361)):
        chosen = []
        for s in seeds:
            hits = [v for k, v in runs.items() if k.startswith(prefix) and k.endswith(f'seed{s}')]
            if prefix.startswith('v11_hr_relu_h2_r16') and not hits:
                hits = [v for k, v in runs.items() if k.startswith('v10_confirm_relu_h2_r16') and k.endswith(f'seed{s}')]
            if len(hits) != 1:
                raise SystemExit(f'{prefix} seed {s}: found {len(hits)} runs')
            chosen.append(hits[0])
        return chosen

    a.out.mkdir(parents=True, exist_ok=True)
    rank_cells = [dict(key=f'h2_r{r}', title=f'({chr(97 + i)}) Rank $r={r}$', formula=None,
                       runs=cell(f'v11_hr_relu_h2_r{r}_')) for i, r in enumerate((2, 4, 8, 16))]
    s1 = render(rank_cells, 'ReLU student  |  quadratic teacher at ranks 2–16  |  10 initializations per rank',
                r'teacher $y=r^{-1/2}\Sigma_{i=1}^{r}h_2(u_i^\top x)$, $h_2(z)=(z^2-1)/\sqrt{2}$;  '
                r'$d=64$, $m=256$, population GD with step $h=0.5$, profiled intercept' + '\n'
                'fresh seeds 9351–9360 fixed before any outcome (rank 16 includes the four confirmation seeds 9351–9354)',
                str(a.out / 'highrank_relu_rank_sweep'))
    teacher_cells = [dict(key='h2_r8', title='(a) Quadratic', formula=r'$h_2(z)=(z^2-1)/\sqrt{2}$',
                          runs=cell('v11_hr_relu_h2_r8_')),
                     dict(key='abs_r8', title='(b) Absolute value', formula=r'$|z|$',
                          runs=cell('v11_hr_relu_abs_r8_')),
                     dict(key='rbf_r8', title='(c) Gaussian bump', formula=r'$e^{-z^2/2}$',
                          runs=cell('v11_hr_relu_gaussian_rbf_r8_'))]
    s2 = render(teacher_cells, 'ReLU student  |  rank-8 additive teachers  |  10 initializations per teacher',
                r'teacher $y\propto\Sigma_{i=1}^{8}q(u_i^\top x)$ with $\mathbb{E}[y^2]=1$ (mean retained, fitted by the '
                r'profiled intercept);  $d=64$, $m=256$, step $h=0.5$' + '\n' + 'fresh seeds 9351–9360 fixed before any outcome',
                str(a.out / 'highrank_relu_teachers_r8'))
    (a.out / 'PLOT_DATA_relu.json').write_text(json.dumps(dict(rank_sweep=s1, teachers_r8=s2), indent=1) + '\n')
    print(json.dumps({k: (v['qualified_1pct'], v['qualified_5pct'], v['common_plateau_end_5pct'])
                      for k, v in {**s1, **s2}.items()}))


if __name__ == '__main__':
    main()
