"""Saved-data reporting for the fixed four-seed ReLU h2 rank-16 confirmation.

Scientific criteria come only from the unchanged canonical summary. Metric
extraction, gap handling, interpolation, units, colors and curve drawing reuse
the canonical plotter. The explicitly named four-seed aggregate is separate
from its two-seed aggregate. No model or training module is imported.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V10 = ROOT.parent
CANONICAL = V10 / 'rank_sweep/analysis/plot_coverage.py'
SUMMARIZER = V10 / 'rank_sweep/analysis/summarize_coverage.py'
RANK_WRAPPER = V10 / 'rank_sweep/analysis_rank/plot_rank.py'
SEEDS = (9351, 9352, 9353, 9354)
PLOT_SHA = '8362da196bb7ec04cd928fa8f88acb8b903de3b7bae6a73c0bb7a59f877db3fb'
SUMMARY_SHA = '57ca1a05e364a9798e3a04a63968c9994ac26ec8cf96f8f127174d3c02cd2166'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(CANONICAL) != PLOT_SHA or sha(SUMMARIZER) != SUMMARY_SHA:
    raise RuntimeError('Canonical analysis changed; review before reporting')
spec = importlib.util.spec_from_file_location('confirmation_rank_presentation', RANK_WRAPPER)
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)
p = rank.p


def select_arms(summary, manifest):
    """Require every fixed outcome; recover absent planning metadata only."""
    if (manifest.get('total_arms') != 4 or manifest.get('confirmation_denominator') != 4
            or manifest.get('development_seeds_not_pooled') != [641, 642]):
        raise ValueError('The fixed four-arm confirmation manifest is required')
    jobs = manifest['configs']
    if len(jobs) != 4 or sorted(j['seed'] for j in jobs) != list(SEEDS):
        raise ValueError('All four prespecified fresh seeds are required')
    plans = {j['id']: j for j in jobs}
    if len(plans) != 4:
        raise ValueError('Duplicate planned IDs')
    expected = json.loads((ROOT/'bundle/reference/v10_rank_relu_h2_r16_d64_m256_seed641.json').read_text())
    for job in jobs:
        config = job['config']
        clean_config = {k: v for k, v in config.items() if k not in ('id', 'seed')}
        if clean_config != {k: v for k, v in expected.items() if k not in ('id', 'seed')}:
            raise ValueError('Confirmation recipe differs from the frozen development reference')
        if (job['engine'], job['teacher'], job['rank'], job['width']) != ('relu', 'h2', 16, 256):
            raise ValueError('Unexpected confirmation cell')
        if config['id'] != job['id'] or config['seed'] != job['seed']:
            raise ValueError('Manifest/config identity disagreement')
    if (summary.get('schema') != 'breadth14_saved_coverage_v1'
            or summary.get('snapshot_usable') is not True
            or summary.get('inputs_unchanged') is not True):
        raise ValueError('Stable canonical summary required')
    source = summary['arms']
    if len(source) != 4 or len({a['id'] for a in source}) != 4 or {a['id'] for a in source} != set(plans):
        raise ValueError('All four original planned outcomes must remain; no development pooling')
    arms, restored = [], []
    for item in source:
        arm = copy.deepcopy(item)
        job = plans[arm['id']]
        if arm.get('config') is not None and arm['config'] != job['config']:
            raise ValueError('Summary/manifest configuration mismatch')
        metadata = rank.metadata(job)
        fields = []
        for key, value in metadata.items():
            if arm.get(key) is None:
                arm[key] = value
                if value is not None:
                    fields.append(key)
            elif arm[key] != value:
                raise ValueError(f'Summary/manifest mismatch: {arm["id"]}:{key}')
        if fields:
            restored.append(dict(id=arm['id'], planning_fields=fields))
        if p.finite(arm.get('target_variance')) and arm['target_variance'] != 1:
            raise ValueError('Quadratic confirmation requires Var(Y)=1')
        arms.append(arm)
    return sorted(arms, key=lambda a: a['seed']), restored


def common_support(arms):
    if not arms or any(not a.get('history') for a in arms):
        return None
    lo = max(a['history'][0]['time'] for a in arms)
    hi = min(a['history'][-1]['time'] for a in arms)
    return [lo, hi] if hi >= lo else None


def aggregate_four(arms):
    """Four-seed median/range; every displayed aggregate requires 4/4 support.

    The canonical interpolation routine preserves missing-point gaps. Finite
    unresolved values remain visible with flags, exactly as in rank plots.
    The independent median check uses the middle two sorted values, not the
    two-seed mean check in the unchanged canonical aggregate.
    """
    if len(arms) != 4 or tuple(a['seed'] for a in arms) != SEEDS:
        raise ValueError('Four-seed aggregate requires all prescribed seeds in order')
    bounds = common_support(arms)
    if bounds is None:
        return None
    low, high = bounds
    extras = [q['time'] for a in arms for q in a.get('checkpoints', []) if low <= q['time'] <= high]
    extras += [p.prefix(a, r)['end_time'] for a in arms for r in (1.01, 1.05)
               if p.finite(p.prefix(a, r).get('end_time')) and low <= p.prefix(a, r)['end_time'] <= high]
    grid = p.np.unique(p.np.r_[p.np.linspace(low, high, 601), extras])
    result = dict(support=bounds, x=grid, metrics={})
    for name in ('loss', 'A_min', 'A_mean', 'refit'):
        series = [p.metric(a, name) for a in arms]
        interpolated = [p.interpolate(x, y, flags, grid) for x, y, flags, _ in series]
        values = p.np.array([v[0] for v in interpolated])
        valid = p.np.array([v[1] for v in interpolated])
        flags = p.np.array([v[2] for v in interpolated])
        all_four = valid.all(axis=0)
        q = p.np.full((3, len(grid)), p.np.nan)
        q[:, all_four] = p.np.quantile(values[:, all_four], [0, .5, 1], axis=0)
        independent = p.np.array([p.independent_interpolation(x, y, grid) for x, y, _, _ in series])
        p.np.testing.assert_allclose(values, independent, rtol=0, atol=2e-12, equal_nan=True)
        complete = p.np.isfinite(independent).all(axis=0)
        ordered = p.np.sort(independent[:, complete], axis=0)
        expected = p.np.full_like(q, p.np.nan)
        expected[:, complete] = p.np.array([ordered[0], (ordered[1] + ordered[2])/2, ordered[3]])
        p.np.testing.assert_allclose(q, expected, rtol=0, atol=2e-12, equal_nan=True)
        result['metrics'][name] = dict(values=values, valid=valid, flags=flags,
                                      range_median=q, support_count=valid.sum(axis=0),
                                      duplicate_time_points=sum(row[3] for row in series))
    return result


def zoom_end(arms, ratio):
    bounds = common_support(arms)
    ends = [p.prefix(a, ratio).get('end_time') for a in arms]
    if bounds is None or any(not p.finite(t) for t in ends):
        return None
    return min(bounds[1], *ends)


def counts(arms):
    return {str(r): {label: sum(p.prefix(a, r).get(field) is not None for a in arms)
                    for label, field in [('original', 'first_material_candidate'),
                                         ('qualified', 'first_numerically_qualified_candidate'),
                                         ('qualified_later_release', 'first_qualified_complete_sequence')]}
            for r in (1.01, 1.05)}


def render_columns(columns, title, subtitle, median=False):
    fig = p.plt.figure(figsize=(18.5, 8.6))
    outer = fig.add_gridspec(1, len(columns), left=.07, right=.98, bottom=.26, top=.76, wspace=.32)
    fig.text(.525, .962, title, ha='center', fontsize=14, color='black')
    fig.text(.525, .916, subtitle, ha='center', fontsize=11, color='black')
    fig.text(.525, .878, 'ReLU student · quadratic h₂ · r=16, d=64, m=256, scale=0.01 · profiled intercept · E[Y²]=Var(Y)=1', ha='center', fontsize=10, color='black')
    fig.text(.525, .840, 'Fresh seeds 9351–9354 only; confirmation denominator 4. Development seeds 641/642 are separate.', ha='center', fontsize=9.5, color='black')
    original_zoom = p.zoom_end
    try:
        for i, (label, arms, agg, ratio) in enumerate(columns):
            p.zoom_end = lambda cell, r=ratio: zoom_end(cell, r) if r is not None else None
            start = len(fig.axes)
            axes = p.cell_axes(fig, outer[0, i], 'relu', 'h2', arms, agg, ratio is not None)
            header = fig.axes[start]
            header.get_subplotspec().get_gridspec().set_height_ratios([.95, 1, 1, 1])
            header.texts[0].set_text(label)
            header.texts[0].set_fontsize(10)
            c = counts(arms); denominator = len(arms)
            fmt = lambda key: ', '.join(f"{c[str(r)][key]}/{denominator}" for r in (1.01, 1.05))
            gaps = rank.diagnostic_coverage_counts(arms)
            censored = [sum(not p.prefix(a, r) or bool(p.prefix(a, r).get('right_censored')) for a in arms) for r in (1.01, 1.05)]
            header.texts[1].set_text(f"Original / qualified 1%: {c['1.01']['original']}/{denominator}, {c['1.01']['qualified']}/{denominator}\n"
                                     f"Original / qualified 5%: {c['1.05']['original']}/{denominator}, {c['1.05']['qualified']}/{denominator}\n"
                                     f"Loss-prefix censored 1%/5%: {censored[0]}/{denominator}, {censored[1]}/{denominator}\n"
                                     f"Diagnostic gaps/warnings 1%/5%: {gaps['1.01']}/{denominator}, {gaps['1.05']}/{denominator}")
            header.texts[1].set_y(.75)
            header.texts[1].set_fontsize(7)
            for ax in axes:
                for text in ax.texts:
                    text.set_text(text.get_text().replace('No paired initial window', 'No saved initial window'))
                if not median:
                    for line in ax.lines:
                        if line.get_color() in [p.COLORS[k] for k in ('loss','A_min','A_mean','refit')]:
                            line.set_alpha(1)
                            line.set_linewidth(1.4)
    finally:
        p.zoom_end = original_zoom
    handles = [p.Line2D([0],[0],color=p.COLORS['loss'],label='Raw MSE'),
               p.Line2D([0],[0],color=p.COLORS['A_min'],label='AGOP minimum'),
               p.Line2D([0],[0],color=p.COLORS['A_mean'],ls='--',label='AGOP mean'),
               p.Line2D([0],[0],color=p.COLORS['refit'],label='Refit raw MSE'),
               p.Patch(facecolor=p.COLORS['prefix5'],label='Initial 5% prefix'),
               p.Patch(facecolor=p.COLORS['prefix1'],alpha=.3,label='Initial 1% prefix'),
               p.Line2D([0],[0],color=p.COLORS['warning'],marker='|',lw=0,label='Unresolved'),
               p.Line2D([0],[0],color=p.COLORS['warning'],marker='x',lw=0,label='Missing diagnostic')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.525,.143),ncol=4,frameon=False,fontsize=9)
    foot = ('Bold: median only where all four seeds have finite support. Band: four-seed range, not a confidence interval. Faint: individual runs.' if median
            else 'Each column is one prespecified seed, with its own saved loss-only prefix. No seed is selected or replaced; no medians are used on this sheet.')
    fig.text(.07,.113,foot,fontsize=9,color='black')
    fig.text(.07,.080,'Counts use saved same-checkpoint criteria, never curve interpolation. Refit is bounded: ‖a‖₂ ≤ 32, ‖a‖₁ ≤ 256. Accepted force time is shown.',fontsize=9,color='black')
    fig.text(.07,.047,'Numerical qualification preserves original guards; missing values remain gaps and unresolved values retain flags. Four runs do not establish high probability.',fontsize=9,color='black')
    return fig


def make_report(summary, arms, output):
    totals = counts(arms)
    candidates = []
    for arm in arms:
        for ratio in (1.01, 1.05):
            pref = p.prefix(arm, ratio)
            candidates.append(dict(id=arm['id'], seed=arm['seed'], ratio=ratio,
                                   first_material_candidate=pref.get('first_material_candidate'),
                                   first_numerically_qualified_candidate=pref.get('first_numerically_qualified_candidate'),
                                   first_qualified_complete_sequence=pref.get('first_qualified_complete_sequence'),
                                   prefix=copy.deepcopy(pref)))
    (output/'CANDIDATES.json').write_text(json.dumps(p.clean(dict(confirmation_denominator=4, records=candidates)),indent=2)+'\n')
    lines = ['# Four fresh-seed confirmation: ReLU h2, rank 16', '',
             'All four prespecified seeds 9351–9354 are retained. Development seeds 641/642 are not pooled. The selected recipe and criteria were frozen before these outcomes; this is confirmation of that recipe only.', '',
             '| Initial window | Original candidates | Numerically qualified | Qualified with later loss release |',
             '|---|---:|---:|---:|']
    for ratio in (1.01,1.05):
        c=totals[str(ratio)]
        lines.append(f"| {round(100*(ratio-1))}% | {c['original']}/4 | {c['qualified']}/4 | {c['qualified_later_release']}/4 |")
    lines += ['', '## Every planned outcome', '',
              '| Seed | Process | Saved stop | Final raw MSE | Diagnostics complete | 1% / 5% qualified assessment |',
              '|---|---|---|---:|---|---|']
    for a in arms:
        outcome = ' / '.join(p.prefix(a,r).get('numerical_qualification_assessment',p.prefix(a,r).get('assessment','unavailable')) for r in (1.01,1.05))
        lines.append(f"| {a['seed']} | {a.get('process_state','unavailable')} | {a.get('stop_reason','unavailable')} | {p.fmt(a.get('final_loss_raw'))} | {a.get('diagnostics_complete',False)} | {outcome} |")
    lines += ['', '## First numerically qualified saved checkpoint', '',
              'The two windows are reported separately even when their first qualified checkpoint is the same.', '',
              '| Seed | Window | Step | Force time | ΔA_min | Refit lower gain / Var(Y) | Later raw-loss drop |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for record in candidates:
        q=record['first_numerically_qualified_candidate'] or {}
        seq=record['first_qualified_complete_sequence'] or {}
        # Later release shown only when the complete sequence is this same point.
        later=seq.get('later_loss_drop_raw') if seq.get('step')==q.get('step') else None
        lines.append('| '+' | '.join([str(record['seed']),f"{round(100*(record['ratio']-1))}%",str(q.get('step','unavailable')),
                                    p.fmt(q.get('time')),p.fmt(q.get('A_min_gain')),p.fmt(q.get('refit_gain_lower_over_variance')),p.fmt(later)])+' |')
    lines += ['', '## Figures and interpretation', '',
              '- [Four-seed median and seed range: full / 1% / 5%](four_seed_median.png). All displayed medians require 4/4 finite support. Bands are seed ranges, not confidence intervals.',
              '- Individual seeds: [full](seeds_full.png), [1%](seeds_initial_1pct.png), [5%](seeds_initial_5pct.png). Each seed uses its own loss-only zoom; the median uses the smallest common loss-only window.',
              '- Matching SVGs, a four-page PDF, raw plotted per-seed arrays, four-seed aggregate arrays and complete arm records are saved alongside these figures.', '',
              'The exact all-update initial max/min loss windows are 1.01 and 1.05. Each candidate uses one saved checkpoint with ΔA_min ≥ 0.5 and bounded-refit initial-lower minus current-upper gain ≥ 0.1 Var(Y), plus the original and additional numerical screens. Later release requires a subsequent raw-loss decrease ≥ 0.1 Var(Y) from a qualified candidate. Cell-level canonical `complete_sequences` tracks original candidates; this report explicitly uses `first_qualified_complete_sequence` for qualified later release.', '',
              'Raw MSE equals variance-normalized MSE here because the fixed h2 target has Var(Y)=1. The raw teacher and profiled output intercept are unchanged. Bounded refits retain l2 ≤ 32 and l1 ≤ 256. The force-time horizon is 1500 and the raw-loss stop is 0.01; horizon completion does not imply reaching the loss target.', '',
              'Curves use saved data only. Missing diagnostics remain gaps, unresolved finite values retain warnings, and no interpolation determines a candidate. Four-seed aggregation is a separately named presentation adapter using unchanged canonical metric/interpolation functions; the existing two-seed aggregate is untouched.', '',
              'This is numerical confirmation of one selected recipe with four fresh seeds. It does not establish a high-probability, architecture-wide or formal population claim. No new model evaluations, quadrature tests, Jacobian audits or training were performed. All saved numerical warnings, completion and censoring fields remain in the records.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return totals


def write_report(summary_path, manifest_path, output):
    summary_path,manifest_path,output=(Path(q).resolve() for q in (summary_path,manifest_path,output))
    if not output.is_relative_to(HERE) or output==HERE or output.exists():
        raise ValueError('Choose a fresh child output directory under confirmation/analysis')
    tracked={str(q):sha(q) for q in [summary_path,manifest_path,Path(__file__),CANONICAL,SUMMARIZER,RANK_WRAPPER,ROOT/'bundle/PROTOCOL.md',ROOT/'bundle/DESIGN.json',ROOT/'bundle/FRESH_SEED_AUDIT.json',ROOT/'bundle/reference/v10_rank_relu_h2_r16_d64_m256_seed641.json']}
    summary=json.loads(summary_path.read_text());manifest=json.loads(manifest_path.read_text())
    if summary.get('script_sha256')!=SUMMARY_SHA or summary.get('input_sha256',{}).get(str(manifest_path))!=sha(manifest_path):
        raise ValueError('Summary must come from the unchanged canonical source and exact manifest')
    arms,restored=select_arms(summary,manifest)
    unavailable=[]
    for name,digest in summary['input_sha256'].items():
        if Path(name).is_file():
            if sha(name)!=digest:raise RuntimeError(f'Changed summary input: {name}')
            tracked[name]=digest
        else: unavailable.append(name)
    units=[p.normalized_units_check(a) for a in arms]
    agg=aggregate_four(arms)
    p.configure();p.plt.rcParams.update({'text.color':'black','xtick.color':'black','ytick.color':'black'})
    output.mkdir(parents=True)
    with p.PdfPages(output/'confirmation_sheets.pdf') as pdf:
        for ratio,suffix in [(None,'full'),(1.01,'initial_1pct'),(1.05,'initial_5pct')]:
            mode='Full observed trajectories' if ratio is None else f'Initial {round(100*(ratio-1))}% loss-prefix zoom'
            fig=render_columns([(f'Seed {a["seed"]}',[a],None,ratio) for a in arms],
                               'Fresh-seed confirmation: ReLU, quadratic teacher, rank 16',mode)
            for ext in ('png','svg'):fig.savefig(output/f'seeds_{suffix}.{ext}',dpi=140,facecolor='white')
            pdf.savefig(fig,facecolor='white');p.plt.close(fig)
        fig=render_columns([(label,arms,agg,r) for r,label in [(None,'Full observed trajectories'),(1.01,'Initial 1% shared prefix'),(1.05,'Initial 5% shared prefix')]],
                           'Four fresh seeds: median and seed range',
                           'Common finite support requires all 4 seeds; ranges are not confidence intervals',median=True)
        for ext in ('png','svg'):fig.savefig(output/f'four_seed_median.{ext}',dpi=140,facecolor='white')
        pdf.savefig(fig,facecolor='white');p.plt.close(fig)
    for a in arms:
        arrays={}
        for name in ('loss','A_min','A_mean','refit'):
            x,y,flags,duplicates=p.metric(a,name)
            arrays.update({f'{name}_time':x,f'{name}_value':y,f'{name}_screen':flags})
        p.np.savez_compressed(output/f'seed_{a["seed"]}_saved_curves.npz',**arrays)
    if agg is not None:
        p.np.savez_compressed(output/'aggregate_four_fresh_seeds.npz',x=agg['x'],**{f'{metric}_{key}':v for metric,info in agg['metrics'].items() for key,v in info.items()})
    totals=make_report(summary,arms,output)
    records=dict(confirmation_denominator=4,seeds=list(SEEDS),development_seeds_excluded=[641,642],arms=arms,planning_metadata_restored=restored)
    (output/'RECORDS.json').write_text(json.dumps(p.clean(records),indent=2,allow_nan=False)+'\n')
    qa=dict(pass_check=True,planned_arms=4,all_planned_seeds=list(SEEDS),development_seeds_pooled=False,
            counts=totals,diagnostic_gaps_or_warnings=rank.diagnostic_coverage_counts(arms),
            shared_support=agg['support'] if agg else None,shared_zoom_end={str(r):zoom_end(arms,r) for r in (1.01,1.05)},
            median_requires_all_four_finite=True,seed_ranges_not_confidence_intervals=True,
            canonical_two_seed_aggregate_unchanged=True,four_seed_medians_independently_checked=True,
            normalized_units_checks=units,no_model_evaluation=True,training_runs=0,visual_review='pending')
    (output/'PLOT_QA.json').write_text(json.dumps(p.clean(qa),indent=2,allow_nan=False)+'\n')
    if any(sha(name)!=digest for name,digest in tracked.items()):raise RuntimeError('Input or source changed during reporting')
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=tracked,
                    inputs_unchanged=True,unavailable_source_paths=unavailable,source_hashes_from_summary=summary['input_sha256'],
                    output_sha256={q.name:sha(q) for q in output.iterdir() if q.is_file()})
    (output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),planned_arms=4,counts=totals)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    write_report(args.summary,args.manifest,args.output)
