import ast,copy,importlib.util,json
from pathlib import Path
import plot_coverage as p
from test_plot_coverage import arm
cells=[]
for recipe,scale in [('low',.1),('high',.3)]:
 for seed in [641,642]:
  x=arm(seed,4,False);x.update(cell=recipe,scale=scale,id=f'{recipe}_{seed}');cells.append(x)
assert len(p.grouping({'arms':cells},paired_study=True))==2
try:p.grouping({'arms':cells})
except ValueError:pass
else:raise AssertionError('Default full14-grid check changed')
try:p.grouping({'arms':cells[:-1]},paired_study=True)
except ValueError:pass
else:raise AssertionError('Missing paired seed accepted')
print('Paired-study grouping checks passed; original numerical helpers unchanged by patch.')
