"""Display-only physical/normalized clocks for the frozen six-arm cubic pilot.

No model imports. Canonical summaries, candidate membership and physical source
records remain unchanged. Canonical two-seed aggregates are computed once in
physical time; normalized plots scale only copies of their abscissae.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RANK = ROOT.parent/'rank_sweep/analysis_rank/plot_rank.py'
SUMMARIZER = ROOT.parent/'rank_sweep/analysis/summarize_coverage.py'
PLOT = ROOT.parent/'rank_sweep/analysis/plot_coverage.py'
QS = (1., .3, .1)
SEEDS = (641, 642)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(PLOT) != '8362da196bb7ec04cd928fa8f88acb8b903de3b7bae6a73c0bb7a59f877db3fb':
    raise RuntimeError('Canonical plot source changed; review before reporting')
if sha(SUMMARIZER) != '57ca1a05e364a9798e3a04a63968c9994ac26ec8cf96f8f127174d3c02cd2166':
    raise RuntimeError('Canonical summary source changed; review before reporting')
spec = importlib.util.spec_from_file_location('headscale_rank_helpers', RANK)
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)
p = rank.p


def select_groups(summary, manifest):
    jobs = manifest['configs']
    if manifest.get('total_arms') != 6 or len(jobs) != 6:
        raise ValueError('All six declared arms are required')
    expected = {(q, seed) for q in QS for seed in SEEDS}
    if {(j['q'], j['seed']) for j in jobs} != expected or len({j['id'] for j in jobs}) != 6:
        raise ValueError('Require exactly q=1,.3,.1 crossed with reused seeds641/642')
    for j in jobs:
        outer = j['config'][0]; c = outer['args']; q = j['q']
        if (c['seed'],c['link'],outer['rank'],outer['tag'],outer['cell']) != (j['seed'],j['teacher'],j['rank'],j['id'],j['cell']):
            raise ValueError('Outer/config manifest identity disagreement')
        if (j['engine'], j['teacher'], j['rank'], c['d'], c['m'], c['s'], c['head_ratio']) != ('swiglu','h3',8,64,64,.3,q):
            raise ValueError('Unexpected scientific recipe')
        if (c['n_pair'],c['n_diag'],c['n_z'],c['n_x']) != (32,96,48,24):
            raise ValueError('All six arms require common frozen quadrature orders')
        if not math.isclose(c['head_lr'], .01*q*q, rel_tol=1e-14) or not math.isclose(c['t_max']*q,3000):
            raise ValueError('Head rate or clock horizon mismatch')
        if (c['h'],c['alpha'],c['L_stop'],c['max_steps'],c['c']) != (.01,1.,.01,15000,[1.]*8):
            raise ValueError('Changed update, loss-stop or teacher recipe')
        if not math.isclose(c['dt_max']*q,50) or not math.isclose(c['cp_min']*q,.5):
            raise ValueError('Changed frozen clock caps')
    if summary.get('schema') != 'breadth14_saved_coverage_v1' or not summary.get('snapshot_usable') or not summary.get('inputs_unchanged'):
        raise ValueError('Stable canonical saved-data summary required')
    source = summary['arms']; plans = {j['id']:j for j in jobs}
    if len(source)!=6 or len({a['id'] for a in source})!=6 or {a['id'] for a in source}!=set(plans):
        raise ValueError('Dropped, duplicate or unknown planned arm')
    arms=[]; restored=[]
    for original in source:
        arm=copy.deepcopy(original); job=plans[arm['id']]
        if arm.get('config') is not None and arm['config']!=job['config']:
            raise ValueError('Summary/config mismatch')
        fields=[]
        for key,value in rank.metadata(job).items():
            if arm.get(key) is None:
                arm[key]=value
                if value is not None: fields.append(key)
            elif arm[key]!=value:
                raise ValueError(f'Summary/manifest metadata mismatch: {arm["id"]}:{key}')
        if fields:restored.append(dict(id=arm['id'],planning_fields=fields))
        arms.append(arm)
    groups = p.grouping({'arms':arms},paired_study=True)
    cells = {plans[cell[0]['id']]['q']:cell for cell in groups.values()}
    if set(cells)!=set(QS) or len(groups)!=3:
        raise ValueError('Exactly three distinct paired q cells are required')
    return cells,restored


def display_cell(cell, factor):
    """Copy only plotting clocks; candidate objects and raw values stay physical."""
    if not p.finite(factor) or factor<=0:
        raise ValueError('Positive finite display-clock scale required')
    out=copy.deepcopy(cell)
    for arm in out:
        for field in ('history','checkpoints'):
            for row in arm.get(field,[]):
                if p.finite(row.get('time')):row['time']*=factor
        for pref in arm.get('prefixes',{}).values():
            if p.finite(pref.get('end_time')):pref['end_time']*=factor
    return out


def display_aggregate(aggregate, factor):
    """Scale x/support only; preserve every canonical value/flag array exactly."""
    if not p.finite(factor) or factor<=0:
        raise ValueError('Positive finite display-clock scale required')
    if aggregate is None:return None
    out=copy.deepcopy(aggregate)
    out['x']*=factor
    out['support']=[value*factor for value in out['support']]
    return out


def counts(cell):
    return {str(r):{label:sum(p.prefix(a,r).get(field) is not None for a in cell)
                   for label,field in [('original','first_material_candidate'),('qualified','first_numerically_qualified_candidate'),('qualified_later_release','first_qualified_complete_sequence')]}
            for r in (1.01,1.05)}


def render(cells,aggregates,clock,ratio,synthetic=False):
    fig=p.plt.figure(figsize=(18.5,8.6))
    outer=fig.add_gridspec(1,3,left=.07,right=.98,bottom=.26,top=.76,wspace=.32)
    mode='Full observed trajectories' if ratio is None else f'Initial {round(100*(ratio-1))}% loss-prefix zoom'
    title='Cubic Gaussian-head scaling: paired development pilot'
    if synthetic:title='SYNTHETIC FIXTURE ONLY — '+title
    fig.text(.525,.965,title,ha='center',fontsize=14,color='black')
    label='Physical adaptive-flow time t' if clock=='physical' else 'Normalized display time τ = q t'
    fig.text(.525,.918,f'{label} · {mode}',ha='center',fontsize=12,color='black')
    fig.text(.525,.876,'Raw h₃ · r=8, d=64, m=64, s=0.3 · quadrature 32/96/48/24 · head multiplier 0.01 q²',ha='center',fontsize=10,color='black')
    fig.text(.525,.836,'All 6 planned outcomes retained; each q has reused seeds 641/642. E[Y²]=Var(Y)=1; profiled output intercept.',ha='center',fontsize=9.5,color='black')
    original_zoom=p.zoom_end
    try:
        if ratio is not None:p.zoom_end=lambda cell:rank.zoom_end(cell,ratio)
        for i,q in enumerate(QS):
            factor=q if clock=='normalized' else 1.
            cell=display_cell(cells[q],factor);agg=display_aggregate(aggregates[q],factor)
            start=len(fig.axes)
            axes=p.cell_axes(fig,outer[0,i],'swiglu','h3',cell,agg,ratio is not None)
            header=fig.axes[start]
            header.get_subplotspec().get_gridspec().set_height_ratios([.95,1,1,1])
            header.texts[0].set_text(f'q={q:g} · head multiplier={.01*q*q:g}')
            header.texts[0].set_fontsize(10)
            text=header.texts[1].get_text().replace('Censored/unavailable','Loss-prefix censored/unavailable')
            gaps=rank.diagnostic_coverage_counts(cells[q])
            text+=f"\nDiagnostic gaps/warnings 1%/5%: {gaps['1.01']}/2, {gaps['1.05']}/2"
            header.texts[1].set_text(text);header.texts[1].set_y(.75);header.texts[1].set_fontsize(7)
            axes[-1].set_xlabel('Physical adaptive-flow time t' if clock=='physical' else r'Normalized display time $\tau=q\,t$',fontsize=8,color='black')
    finally:p.zoom_end=original_zoom
    handles=[p.Line2D([0],[0],color=p.COLORS[k],ls='--' if k=='A_mean' else '-',label=v)
             for k,v in [('loss','Raw MSE'),('A_min','AGOP minimum'),('A_mean','AGOP mean'),('refit','Refit raw MSE')]]
    handles += [p.Patch(facecolor=p.COLORS['prefix5'],label='Shared initial 5% prefix'),p.Patch(facecolor=p.COLORS['prefix1'],alpha=.3,label='Shared initial 1% prefix'),
                p.Line2D([0],[0],color=p.COLORS['warning'],marker='|',lw=0,label='Unresolved'),p.Line2D([0],[0],color=p.COLORS['warning'],marker='x',lw=0,label='Missing diagnostic')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.525,.143),ncol=4,frameon=False,fontsize=9)
    fig.text(.07,.113,'Bold: two-seed median on shared finite support. Bands: seed ranges, not confidence intervals. Faint curves retain individual tails.',fontsize=9,color='black')
    fig.text(.07,.080,'Only the displayed clock changes between sheets. Raw loss/refit, thresholds, candidate membership and numerical/censoring flags are unchanged.',fontsize=9,color='black')
    fig.text(.07,.047,'Changing q changes the dynamics; normalized-time plots do not establish equivalence or a longer plateau. Cutoff-envelope refits are numerical evidence.',fontsize=9,color='black')
    return fig


def report(cells,synthetic=False):
    lines=['# Cubic head-scaling saved-data report','']
    if synthetic:lines+=['**SYNTHETIC FIXTURE ONLY. No measured pilot outcomes or initialization verification.**','']
    lines+=['All six fixed arms remain, with two reused development seeds per q. No q values or independent seeds are pooled. Physical-time source records are preserved in RECORDS.json.','',
            '| q | Seed | Process / stop | 1% original / qualified / later | 5% original / qualified / later | End t, 1% / 5% | End τ, 1% / 5% | Diagnostic gaps/warnings |',
            '|---:|---:|---|---|---|---|---|---|']
    for q in QS:
        for a in cells[q]:
            c=counts([a]);ends=[p.prefix(a,r).get('end_time') for r in (1.01,1.05)]
            outcomes=[' / '.join(str(c[str(r)][k])+'/1' for k in ('original','qualified','qualified_later_release')) for r in (1.01,1.05)]
            physical=' / '.join(p.fmt(t) for t in ends);normalized=' / '.join(p.fmt(q*t if p.finite(t) else None) for t in ends)
            lines.append(f"| {q:g} | {a['seed']} | {a.get('process_state','unavailable')} / {p.status_text(a)} | {outcomes[0]} | {outcomes[1]} | {physical} | {normalized} | {rank.diagnostic_coverage_counts([a])} |")
    lines+=['','Both raw prediction/refit MSE and loss divided by target variance are unchanged by the display clock. This fixed h3 target has Var(Y)=1. Median arrays are computed once with unchanged canonical two-seed aggregation in physical time; τ plots multiply only copied abscissae and prefix boundaries by q. Missing values remain gaps, unresolved finite values retain warnings, and there is no one-seed median.','',
            'Initial windows use all recorded loss updates with max/min thresholds 1.01 and 1.05. Original and numerically qualified same-state candidates, and qualified later loss release, are taken verbatim from the canonical summary. Qualified later-release counts use first_qualified_complete_sequence; interpolation never decides a candidate.','',
            'Paired initialization requires saved initial P,V to agree and a/q to match the q=1 reference within each reused seed. Consult INITIALIZATION_AUDIT.json; a planned shared seed is not itself proof. Missing or mismatched initial states remain unresolved or failed, without dropping their outcomes.','',
            'This pilot changes head amplitude and block learning rate, so the trajectories need not be exact time rescalings. Scaling the clock cannot establish a longer plateau. AGOP may change through head dynamics even at fixed hidden features; material feature-refit improvement remains required. Common finite quadrature and cutoff screens are not formal population certificates. Promising states need separate higher-order, block-weighted update and geometry/refit checks; none are performed here. Two reused seeds are development evidence, not fresh-seed confirmation.']
    return '\n'.join(lines)+'\n'


def write_plots(summary_path,manifest_path,output,synthetic=False):
    summary_path,manifest_path,output=(Path(q).resolve() for q in (summary_path,manifest_path,output))
    if output.exists() or not output.is_relative_to(HERE) or output==HERE:
        raise ValueError('Choose a fresh output directory under headscale/analysis')
    tracked={str(q):sha(q) for q in [summary_path,manifest_path,Path(__file__),RANK,PLOT,SUMMARIZER,ROOT/'bundle/PROTOCOL.snapshot.md']}
    data=json.loads(summary_path.read_text());manifest=json.loads(manifest_path.read_text())
    is_fixture=data.get('fixture_provenance',{}).get('kind')=='synthetic'
    if synthetic!=is_fixture:raise ValueError('Synthetic mode and explicit fixture input must agree')
    if not synthetic and (data.get('script_sha256')!=sha(SUMMARIZER) or data.get('input_sha256',{}).get(str(manifest_path))!=sha(manifest_path)):
        raise ValueError('Exact manifest and unchanged canonical summary source required')
    cells,restored=select_groups(data,manifest)
    aggregates={q:p.aggregate(cells[q]) for q in QS}
    units=[p.normalized_units_check(a) for cell in cells.values() for a in cell]
    unavailable=[]
    for name,digest in data.get('input_sha256',{}).items():
        if Path(name).is_file():
            if sha(name)!=digest:raise RuntimeError(f'Changed summary input: {name}')
            tracked[name]=digest
        else:unavailable.append(name)
    if synthetic:
        initialization=dict(status='not_checked_synthetic_fixture',no_scientific_result=True)
    else:
        from paired_initialization import audit
        initialization=audit(manifest_path,Path(data['execution_directory']))
        tracked[str(HERE/'paired_initialization.py')]=sha(HERE/'paired_initialization.py')
        tracked.update(initialization.get('input_sha256',{}))
    output.mkdir(parents=True)
    p.configure();p.plt.rcParams.update({'text.color':'black','xtick.color':'black','ytick.color':'black'})
    for clock in ('physical','normalized'):
        with p.PdfPages(output/f'{clock}_sheets.pdf') as pdf:
            for ratio,suffix in [(None,'full'),(1.01,'initial_1pct'),(1.05,'initial_5pct')]:
                fig=render(cells,aggregates,clock,ratio,synthetic)
                for ext in ('png','svg'):fig.savefig(output/f'{clock}_{suffix}.{ext}',dpi=140,facecolor='white')
                pdf.savefig(fig,facecolor='white');p.plt.close(fig)
    for q,agg in aggregates.items():
        if agg is not None:
            p.np.savez_compressed(output/f'aggregate_q{round(q*100):03d}.npz',x_t=agg['x'],x_tau=q*agg['x'],
                                 **{f'{metric}_{key}':value for metric,info in agg['metrics'].items() for key,value in info.items()})
    records=dict(synthetic_fixture_only=synthetic,planned_arms=6,physical_source_records=True,planning_metadata_restored=restored,
                 cells=[dict(q=q,counts=counts(cells[q]),arms=cells[q]) for q in QS])
    qa=dict(pass_check=True,synthetic_fixture_only=synthetic,planned_arms=6,q_order=list(QS),seeds=list(SEEDS),
            clock_only_scaling_on_copies=True,canonical_two_seed_aggregation_unchanged=True,raw_values_and_candidate_membership_unchanged=True,
            paired_initialization_status=initialization.get('status'),normalized_units_checks=units,
            shared_support_t={str(q):aggregates[q]['support'] if aggregates[q] else None for q in QS},
            no_model_evaluation=True,training_runs=0,visual_review='pending')
    for name,value in [('RECORDS.json',records),('PLOT_QA.json',qa),('INITIALIZATION_AUDIT.json',initialization)]:
        (output/name).write_text(json.dumps(p.clean(value),indent=2,allow_nan=False)+'\n')
    (output/'REPORT.md').write_text(report(cells,synthetic))
    if any(sha(name)!=digest for name,digest in tracked.items()):raise RuntimeError('Inputs changed while plotting')
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=tracked,inputs_unchanged=True,
                    unavailable_source_paths=unavailable,source_hashes_from_summary=data.get('input_sha256',{}),
                    output_sha256={f.name:sha(f) for f in output.iterdir() if f.is_file()})
    (output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),synthetic=synthetic,planned_arms=6)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path);parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--synthetic-fixture',action='store_true')
    args=parser.parse_args();write_plots(args.summary,args.manifest,args.output,args.synthetic_fixture)
