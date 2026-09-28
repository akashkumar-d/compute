"""Secondary exploratory, loss-only windows; no model evaluation or training."""
from collections import deque
from pathlib import Path
import argparse
import hashlib
import json
import math

RULE = ('For each 1%/5% loss ratio, choose the longest observed-clock interval '
        'in [.5R,1.05V] or [.5R,1.05R], where V is target variance and R its '
        'nonlinear Hermite energy. Ties: earliest start, earliest end. Require '
        '50 updates and10% of the observed eligible clock. Compare the first '
        'and last required saved checkpoints inside the selected interval; '
        'never move endpoints using feature/refit outcomes. Require50% clock '
        'coverage, both numerical screens, deltaAmin>=.5 and refit gain>=.1V. '
        'These are benchmark loss bands, not proof an affine predictor was fitted.')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x): return isinstance(x,(int,float)) and math.isfinite(x)

def longest(loss,clock,ratio,band):
    low,high=band; mins=deque(); maxs=deque(); left=0; best=None
    eligible=[i for i,x in enumerate(loss) if finite(x) and 0<x and low<=x<=high]
    span=clock[eligible[-1]]-clock[eligible[0]] if len(eligible)>1 else 0.
    for right,x in enumerate(loss):
        if not finite(x) or x<=0 or not low<=x<=high:
            mins.clear(); maxs.clear(); left=right+1; continue
        while mins and loss[mins[-1]]>=x: mins.pop()
        while maxs and loss[maxs[-1]]<=x: maxs.pop()
        mins.append(right); maxs.append(right)
        while loss[maxs[0]]/loss[mins[0]]>ratio:
            if mins[0]==left: mins.popleft()
            if maxs[0]==left: maxs.popleft()
            left+=1
        choice=(clock[right]-clock[left],-left,-right)
        if best is None or choice>best: best=choice
    return (None if best is None else (-best[1],-best[2])),span

def select(summary,energies):
    if not summary['snapshot_usable']: raise ValueError('Mixed input snapshot')
    moments={r['teacher']:r for r in energies['rows']}; rows=[]
    for arm in summary['arms']:
        history=arm.get('history',[]); row=dict(id=arm['id'],windows=[])
        if not history or arm.get('issues'):
            row['status']='missing_or_inconsistent_history';rows.append(row);continue
        e=moments.get(arm['teacher'])
        if e is None or arm.get('r')!=energies['rank']:
            row['status']='teacher_moments_unavailable';rows.append(row);continue
        V,R=e['target_variance'],e['nonlinear_energy']
        if not finite(arm.get('target_variance')) or abs(V-arm['target_variance'])>1e-10:
            row['status']='normalization_inconsistent';rows.append(row);continue
        loss=[p['loss_raw'] for p in history]; clock=[p['time'] for p in history]
        if any(not finite(t) for t in clock) or any(b<a for a,b in zip(clock,clock[1:])):
            row['status']='invalid_clock';rows.append(row);continue
        row.update(status='selected_from_loss_only',target_variance=V,nonlinear_energy=R)
        for ratio in (1.01,1.05):
            for kind,band in [('after_mean',[.5*R,1.05*V]),('after_affine',[.5*R,1.05*R])]:
                bounds,span=longest(loss,clock,ratio,band)
                w=dict(kind=kind,ratio=ratio,band=band,selected=bounds is not None,eligible_clock_span=span)
                if bounds is not None:
                    a,b=bounds; duration=clock[b]-clock[a]
                    w.update(start_step=history[a]['step'],end_step=history[b]['step'],start_time=clock[a],end_time=clock[b],duration=duration,
                        observed_loss_ratio=max(loss[a:b+1])/min(loss[a:b+1]),duration_pass=b-a>=50 and duration>=.1*span,
                        starts_after_initialization=a>0,touches_observed_horizon=b==len(history)-1)
                row['windows'].append(w)
        rows.append(row)
    return dict(rule=RULE,selection_depends_on_features=False,exploratory=True,
        prior_partial_outcomes_known=True,primary_initial_prefix_criterion_unchanged=True,runs=rows)

def evaluate(selection,summary):
    arms={a['id']:a for a in summary['arms']}
    for row in selection['runs']:
        arm=arms[row['id']]
        for w in row['windows']:
            w.update(joint_candidate=False,later_loss_release=False)
            if not w['selected']:continue
            inside=[p for p in arm['checkpoints'] if w['start_step']<=p['step']<=w['end_step']]
            w['required_checkpoint_count']=len(inside)
            if len(inside)<2:w['assessment']='insufficient_saved_endpoints';continue
            a,b=inside[0],inside[-1];w['endpoint_steps']=[a['step'],b['step']]
            w['missing_inside']=sum(p['diagnostic_status']!='ok' for p in inside)
            coverage=(b['time']-a['time'])/w['duration'] if w['duration']>0 else 0.
            w['diagnostic_clock_coverage']=coverage
            needed=('A_min','refit_low_raw','refit_high_raw')
            if any(p['diagnostic_status']!='ok' or p.get('issues') or any(not finite(p.get(k)) for k in needed) for p in (a,b)):
                w['assessment']='endpoint_diagnostic_unavailable';continue
            if any(p['refit_low_raw']>p['refit_high_raw']+1e-10 for p in (a,b)):
                w['assessment']='inconsistent_refit_interval';continue
            gain=a['refit_low_raw']-max(0.,b['refit_high_raw']); delta=b['A_min']-a['A_min'];V=row['target_variance']
            guards=arm.get('snapshot_integrity_verified') is True and all(p.get('agop_screen') is True and p.get('refit_screen') is True and p.get('raw_refit_nonnegative_screen') is True for p in (a,b))
            joint=w['duration_pass'] and coverage>=.5 and guards and delta>=.5 and gain>=.1*V
            later=[p['loss_raw'] for p in arm['history'] if p['step']>b['step'] and finite(p['loss_raw'])]
            drop=b['loss_raw']-min(later) if later else None
            w.update(delta_A_min=delta,refit_gain_numerical_or_cutoff_envelope_raw=gain,refit_gain_over_variance=gain/V,
                both_endpoint_screens=guards,joint_candidate=bool(joint),later_loss_drop_raw=drop,
                later_loss_release=bool(joint and drop is not None and drop>=.1*V),
                comparison_class=arm.get('comparison_class'),certificate=arm.get('certificate'),
                assessment='observed_secondary_candidate' if joint else 'no_qualified_endpoint_candidate')
    return selection

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True,type=Path);p.add_argument('--energies',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    digests={str(f.resolve()):sha(f) for f in (args.summary,args.energies)}
    summary=json.loads(args.summary.read_text());energy=json.loads(args.energies.read_text())
    selected=select(summary,energy);selected['input_sha256']=digests
    (args.output/'LOSS_ONLY_SELECTION.json').write_text(json.dumps(selected,indent=2,allow_nan=False)+'\n')
    result=evaluate(json.loads(json.dumps(selected)),summary)
    if any(sha(Path(f))!=h for f,h in digests.items()):raise RuntimeError('Inputs changed')
    result['script_sha256']=sha(Path(__file__))
    (args.output/'ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(runs=len(result['runs']),candidate_window_entries=sum(w['joint_candidate'] for r in result['runs'] for w in r['windows']),
        unique_runs_with_candidates=sum(any(w['joint_candidate'] for w in r['windows']) for r in result['runs']),
        unique_runs_with_later_start_candidates=sum(any(w['joint_candidate'] and w.get('starts_after_initialization') for w in r['windows']) for r in result['runs']),no_training=True)))

if __name__=='__main__':main()
