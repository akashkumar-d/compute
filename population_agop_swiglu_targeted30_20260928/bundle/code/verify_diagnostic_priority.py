import json,random
from pathlib import Path
from diagnostic_priority import diagnostic_order
rng=random.Random(17633)
for _ in range(300):
    observed=rng.randint(2,2000);states=sorted(set([0,observed]+[rng.randint(0,observed) for _ in range(30)]))
    endpoint=rng.randint(0,observed);strict_exit=rng.choice([None,rng.randint(1,endpoint+1)])
    priority=[0,strict_exit-1 if strict_exit is not None else -1,strict_exit if strict_exit is not None else -1,endpoint,endpoint+1,observed]+sorted(states)
    legacy=list(dict.fromkeys(s for s in priority if s in states))
    assert diagnostic_order(states,observed,endpoint,strict_exit)==legacy
    assert diagnostic_order(states,observed,endpoint,strict_exit,[])==legacy
    new=diagnostic_order(states,observed,endpoint,strict_exit,[.25,.5,.75])
    assert set(new)==set(states) and len(new)==len(states)
    boundaries=list(dict.fromkeys(s for s in priority[:6] if s in states))
    assert new[:len(boundaries)]==boundaries
assert diagnostic_order([0,10,20,30,40,41,60,61,100],100,60,41,[.25,.5,.75])==[0,40,41,60,61,100,10,20,30]
# Censored strictprefix uses observed endpoint; ties snap to earlierstep.
assert diagnostic_order([0,10,30,40],40,40,None,[.5])==[0,40,10,30]
report={'status':'PASS','legacy_omitted_and_empty_fraction_comparisons':300,'all_saved_states_retained':True,'boundary_first':True,'tie_break':'earlier','no_training_executed':True}
Path(__file__).with_name('DIAGNOSTIC_PRIORITY_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
