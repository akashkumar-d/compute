"""Synthetic numerical plotting guards; no new experiment results or model code."""
from pathlib import Path
import copy,importlib.util,json
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('coverage_plot',HERE/'plot_coverage.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def arm(seed,stop=4,missing=False):
 variance=.02
 checkpoints=[dict(step=i,time=float(i),A_min=.1+.1*i,A_mean=.3+.1*i,refit_raw=.018-.002*i,
                   agop_screen=(i!=3),refit_screen=(i!=2),diagnostic_status='ok',issues=[]) for i in range(stop+1)]
 if missing:checkpoints[1].update(A_min=None,A_mean=None,refit_raw=None,diagnostic_status='not_evaluated_runtime_budget')
 return dict(id=f'SYNTHETIC_{seed}',student='swiglu',teacher='h3',cell='SYNTHETIC_TEST',seed=seed,r=2,d=6,m=4,
  target_variance=variance,profiled_intercept=True,head_lr=.01,history=[dict(step=i,time=float(i),loss_raw=variance*(1-.001*i),loss_over_target_variance=1-.001*i) for i in range(stop+1)],
  checkpoints=checkpoints,prefixes={'1.01':dict(end_time=min(2,stop),end_step=min(2,stop),right_censored=False),
  '1.05':dict(end_time=min(3,stop),end_step=min(3,stop),right_censored=stop<3)})
cell=[arm(1,4,True),arm(2,3,False)];before=copy.deepcopy(cell)
agg=p.aggregate(cell);assert agg['support']==[0.,3.]
x=agg['x'];between=(x>0)&(x<2)
for metric in ('A_min','A_mean','refit'):
 assert np.isnan(agg['metrics'][metric]['range_median'][:,between]).all()
 assert np.isfinite(agg['metrics'][metric]['range_median'][:,x==0]).all()
 assert np.isfinite(agg['metrics'][metric]['range_median'][:,x==2]).all()
assert not agg['metrics']['A_min']['flags'][:,x==3].any()
assert not agg['metrics']['refit']['flags'][:,x==2].any()
assert p.zoom_end(cell)==3
changed=copy.deepcopy(cell)
for a in changed:
 for q in a['checkpoints']:q.update(A_min=100,A_mean=-100,refit_raw=999)
assert p.zoom_end(changed)==p.zoom_end(cell)
assert cell==before
missing=copy.deepcopy(cell);missing[1]['history']=[]
assert p.aggregate(missing) is None and p.zoom_end(missing) is None
values,valid,screen=p.interpolate([1,2],[.3,.4],[True,False],np.array([0,1,1.5,2,3]))
assert np.isnan(values[[0,4]]).all() and not valid[[0,4]].any()
assert valid[2] and not screen[2]
x,y,flags,count=p.collapse_duplicate_times([0,1,1,2],[.2,.3,.4,.5],[True]*4)
assert count==1 and np.isnan(y[1]) and not flags[1]
for a in cell:assert p.normalized_units_check(a)['checked']==len(a['history'])
wrong=copy.deepcopy(cell[0]);wrong['history'][0]['loss_raw']=1
try:p.normalized_units_check(wrong)
except ValueError:pass
else:raise AssertionError('Unit mismatch not detected')
try:p.grouping({'arms':cell[:1]},True)
except ValueError:pass
else:raise AssertionError('Missing planned seed was pooled as one-seed median')
assert p.status_text(dict(termination_reason=None,stop_reason='training_budget',process_state='exit_zero'))=='training budget'
wrong_clock=copy.deepcopy(cell);wrong_clock[0]['time_unit']='accepted_force_time'
try:p.grouping({'arms':wrong_clock},True)
except ValueError:pass
else:raise AssertionError('Incompatible clocks were accepted')
wrong_recipe=copy.deepcopy(cell);wrong_recipe[0]['scale']=.1;wrong_recipe[1]['scale']=.01
try:p.grouping({'arms':wrong_recipe},True)
except ValueError:pass
else:raise AssertionError('Different initial scales were pooled')
unstarted=copy.deepcopy(cell);unstarted[0].update(history=[],checkpoints=[],target_mean=None,target_variance=None,process_state='unstarted')
assert len(p.grouping({'arms':unstarted},True))==1 and p.aggregate(unstarted) is None
for key in ('r','d','m','scale','head_lr','profiled_intercept'):unstarted[0].pop(key,None)
unstarted[0]['time_unit']=None
assert len(p.grouping({'arms':unstarted},True))==1 and p.aggregate(unstarted) is None
result=dict(pass_check=True,model_evaluations=0,training_runs=0,synthetic_only=True,
 missing_expected_points_remain_nan=True,no_interpolation_bridges=True,common_support_only=True,
 unresolved_flags_propagate=True,no_extrapolation=True,zoom_invariant_to_alignment_and_refit=True,
 no_one_seed_median=True,raw_variance_units_checked=True,ambiguous_duplicate_times_become_nan=True,
 input_records_unchanged=True,stop_reason_preserved=True,clock_mismatch_rejected=True,recipe_mismatch_rejected=True,missing_arm_normalization_retained=True,exception_fallback_arm_metadata_retained=True)
out=HERE/'plot_validation';out.mkdir(exist_ok=True)
(out/'SYNTHETIC_QA.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
