"""Pure saved-record diagnostic-order validation: zero model evaluations/updates."""
from pathlib import Path
import copy,importlib.util,json,math
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('breadth_runner',ROOT/'code/run_one.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
records=[]
for name,loss,steps in [('monotone',[1-.001*i for i in range(80)],[0,3,6,10,14,20,26,35,40,50,60,79]),
                       ('nonmonotone',[1,.999,1.006,.997,.988,.982,.97,.94],[0,1,2,3,4,5,6,7]),
                       ('censored',[1]*51,[0,10,20,30,40,50])]:
 raw={'rows':[{'step':x,'t':float(x)} for x in steps],'Lhist':[[float(i),v] for i,v in enumerate(loss)]};before=copy.deepcopy(raw)
 legacy={str(d):r.prefix_index(raw,d) for d in (.001,.0001)}
 ratio={str(v):r.ratio_prefix_index(raw,v) for v in (1.01,1.05)}
 selected=list(dict.fromkeys([0]+[p[key] for p in ratio.values() for key in ('checkpoint_index','crossing_checkpoint_index') if p[key] is not None]+[v[1] for v in legacy.values() if v[1] is not None]+[len(steps)-1]))
 original=selected+[i for i in range(len(steps)) if i not in selected]
 for fractions in (None,[]):
  got,order,interior=r.diagnostic_order(raw,legacy,ratio,fractions)
  assert got==selected and order==original and interior==[]
 got,order,interior=r.diagnostic_order(raw,legacy,ratio,[.25,.5,.75])
 assert got==selected and order[:len(selected)]==selected and sorted(order)==list(range(len(steps)))
 for choice in interior:
  end=ratio[choice['prefix']]['last_admissible_update'];target=choice['fraction']*end
  expected=sorted([i for i,x in enumerate(steps) if x<=end],key=lambda i:(abs(steps[i]-target),steps[i],i))[0]
  assert choice['selected_index']==expected
 assert raw==before
 records.append({'history':name,'omitted_default_order_identical':True,'snapshots_histories_unchanged':True,'all_original_states_retained':True,'original_priorities_first':True,'interior_indices':[v['selected_index'] for v in interior]})
# Explicit equal-distance tie: update25 chooses20 rather than30.
raw={'rows':[{'step':x} for x in (0,20,30,50,100)]}
_,_,choice=r.diagnostic_order(raw,{}, {'1.05':{'last_admissible_update':100,'checkpoint_index':4,'crossing_checkpoint_index':None}}, [.25])
assert choice[0]['selected_update']==20
for bad in ([0],[1],[-.2],[True],[math.nan],[math.inf],['.5'],'.5'):
 try:r.validate_diagnostic_priority_fractions(bad)
 except ValueError:pass
 else:raise AssertionError(f'Invalid fractions accepted: {bad}')
result={'pass_check':True,'model_evaluations':0,'parameter_updates':0,'tie_earlier_pass':True,'invalid_fraction_guard_pass':True,'cases':records}
(ROOT/'DIAGNOSTIC_PRIORITY_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'pass_check':True,'model_evaluations':0,'parameter_updates':0}))
