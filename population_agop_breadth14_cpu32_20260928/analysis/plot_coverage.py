"""Scientific breadth14 plots from a unified saved-data summary only.

No model or training module is imported. Two planned seeds define each cell;
missing arms remain missing, and NaN diagnostic gaps are never interpolated over.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/agop-breadth14-plots')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
CANONICAL=('h2','h3','h4','h5','relu','leaky_relu','abs','softplus','silu','gelu','sine','tanh','erf','gaussian_rbf')
COLORS={'loss':'#25395B','A_min':'#007C83','A_mean':'#BB7518','refit':'#814B9C',
        'prefix5':'#DDEDEC','prefix1':'#598C89','warning':'#A32C28'}
ALIASES={'he2':'h2','he3':'h3','he4':'h4','he5':'h5','sin':'sine','rbf':'gaussian_rbf'}
LABELS={'h2':r'$h_2$','h3':r'$h_3$','h4':r'$h_4$','h5':r'$h_5$','relu':'ReLU',
 'leaky_relu':'Leaky ReLU(0.1)','abs':r'$|z|$','softplus':'Softplus','silu':'SiLU','gelu':'GELU',
 'sine':r'$\sin z$','tanh':r'$\tanh z$','erf':r'$\mathrm{erf}(z)$','gaussian_rbf':'Gaussian RBF'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(x):return isinstance(x,(float,int,np.floating,np.integer)) and not isinstance(x,(bool,np.bool_)) and math.isfinite(x)
def number(x):return float(x) if finite(x) else np.nan
def fmt(x):return f'{x:.3g}' if finite(x) else 'missing'
def clean(value):
 if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
 if isinstance(value,(bool,np.bool_)):return bool(value)
 if isinstance(value,(int,np.integer)):return int(value)
 if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
 return value

def configure():
 plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif',
  'font.size':8,'axes.labelsize':8,'axes.labelcolor':'black','axes.titlesize':9,
  'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':8,'axes.linewidth':.55,
  'lines.linewidth':1.15,'axes.spines.top':False,'axes.spines.right':False,
  'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5,
  'svg.fonttype':'none','svg.hashsalt':'agop-breadth14-v1','path.simplify':False})

def prefix(arm,ratio):
 return arm.get('prefixes',{}).get(str(ratio),{}) or {}

def status_text(arm):
 value=arm.get('termination_reason') or arm.get('stop_reason') or arm.get('process_state') or 'missing receipt'
 if isinstance(value,dict):value=value.get('reason',value.get('status',json.dumps(value)))
 return str(value).replace('_',' ')

def canonical_name(name):return ALIASES.get(name,name)

def grouping(data,allow_historical=False):
 groups={}
 for arm in data['arms']:
  if arm.get('supplemental',False):continue
  student=str(arm['student']).lower();teacher=canonical_name(arm['teacher'])
  if teacher not in CANONICAL and not allow_historical:continue
  expected_unit='accepted_force_time' if student=='relu' else 'adaptive_population_flow_time'
  if arm.get('time_unit') is not None and arm['time_unit']!=expected_unit:raise ValueError(f'Unexpected clock units: {arm["id"]}')
  key=(student,teacher,str(arm.get('cell',teacher)))
  groups.setdefault(key,[]).append(arm)
 for key,cell in groups.items():
  if len(cell)!=2 or len({a['seed'] for a in cell})!=2:
   raise ValueError(f'Exactly two planned seeds are required in each plotted cell: {key} has{len(cell)}')
  cell.sort(key=lambda a:a['seed'])
  for field in ('r','d','m','scale','head_lr','profiled_intercept'):
   if len({str(a[field]) for a in cell if a.get(field) is not None})>1:raise ValueError(f'Within-cell metadata mismatch: {key}:{field}')
  for field in ('target_mean','target_variance'):
   present=[float(a[field]) for a in cell if finite(a.get(field))]
   if len(present)==2 and not math.isclose(*present,rel_tol=1e-12,abs_tol=1e-12):raise ValueError(f'Within-cell normalization mismatch: {key}:{field}')
 if not allow_historical:
  for student in sorted({key[0] for key in groups}):
   teachers=[key[1] for key in groups if key[0]==student]
   if sorted(teachers)!=sorted(CANONICAL):raise ValueError(f'{student}: all14 canonical teacher cells must be present exactly once')
 return groups

def collapse_duplicate_times(x,y,flags):
 """Keep ambiguous zero-clock evolution as a NaN, not a fabricated interpolation."""
 x=np.asarray(x,float);y=np.asarray(y,float);flags=np.asarray(flags,bool)
 if not len(x):return x,y,flags,0
 if not np.isfinite(x).all() or np.any(np.diff(x)<0):raise ValueError('Saved clocks must be finite and nondecreasing')
 starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1];ends=np.r_[starts[1:],len(x)]
 if len(starts)==len(x):return x,y,flags,0
 out_y=[];out_flags=[]
 for a,b in zip(starts,ends):
  values=y[a:b];same=np.isfinite(values).all() and np.all(values==values[0])
  out_y.append(values[0] if same else np.nan);out_flags.append(bool(same and flags[a:b].all()))
 return x[starts],np.asarray(out_y),np.asarray(out_flags),len(x)-len(starts)

def metric(arm,name):
 if name=='loss':
  rows=arm.get('history',[]);x=[p['time'] for p in rows];y=[number(p.get('loss_raw')) for p in rows];flags=[True]*len(rows)
 else:
  rows=arm.get('checkpoints',[]);x=[p['time'] for p in rows]
  key={'A_min':'A_min','A_mean':'A_mean','refit':'refit_raw'}[name]
  y=[number(p.get(key)) if p.get('diagnostic_status')=='ok' else np.nan for p in rows]
  flags=[p.get('agop_screen') is True if name.startswith('A_') else p.get('refit_screen') is True for p in rows]
 return collapse_duplicate_times(x,y,flags)

def interpolate(x,y,flags,grid):
 x=np.asarray(x,float);y=np.asarray(y,float);grid=np.asarray(grid,float);flags=np.asarray(flags,bool)
 out=np.full(grid.shape,np.nan);valid=np.zeros(grid.shape,bool);screen=np.zeros(grid.shape,bool)
 if not len(x):return out,valid,screen
 if np.any(np.diff(x)<=0):raise ValueError('Interpolation requires strictly increasing condensed clocks')
 right=np.searchsorted(x,grid,side='left');hi=np.minimum(right,len(x)-1)
 exact=(right<len(x))&(x[hi]==grid);lo=np.where(exact,hi,np.maximum(hi-1,0))
 valid=(grid>=x[0])&(grid<=x[-1])&np.isfinite(y[lo])&np.isfinite(y[hi])
 out[valid]=np.interp(grid[valid],x,y)
 screen=valid&flags[lo]&flags[hi]
 return out,valid,screen

def independent_interpolation(x,y,grid):
 out=np.full(len(grid),np.nan)
 for i,(t,v) in enumerate(zip(x,y)):
  if np.isfinite(v):out[grid==t]=v
  if i+1<len(x) and np.isfinite(v) and np.isfinite(y[i+1]):
   selected=(grid>=t)&(grid<=x[i+1]);out[selected]=v+(y[i+1]-v)*(grid[selected]-t)/(x[i+1]-t)
 return out

def support(cell):
 history=[a.get('history',[]) for a in cell]
 if len(cell)!=2 or any(not h for h in history):return None
 low=max(h[0]['time'] for h in history);high=min(h[-1]['time'] for h in history)
 return [low,high] if high>=low else None

def aggregate(cell):
 bounds=support(cell)
 if bounds is None:return None
 low,high=bounds
 extras=[p['time'] for a in cell for p in a.get('checkpoints',[]) if low<=p['time']<=high]
 extras += [prefix(a,r).get('end_time') for a in cell for r in (1.01,1.05)
            if finite(prefix(a,r).get('end_time')) and low<=prefix(a,r)['end_time']<=high]
 grid=np.unique(np.r_[np.linspace(low,high,601),extras]);result={'support':bounds,'x':grid,'metrics':{}}
 for name in ('loss','A_min','A_mean','refit'):
  series=[metric(a,name) for a in cell]
  rows=[interpolate(x,y,f,grid) for x,y,f,_ in series]
  values=np.array([r[0] for r in rows]);valid=np.array([r[1] for r in rows]);flags=np.array([r[2] for r in rows])
  both=valid.all(axis=0);q=np.full((3,len(grid)),np.nan)
  q[:,both]=np.quantile(values[:,both],[0,.5,1],axis=0)
  independent=np.array([independent_interpolation(x,y,grid) for x,y,_,_ in series])
  np.testing.assert_allclose(values,independent,rtol=0,atol=2e-12,equal_nan=True)
  expected=np.full_like(q,np.nan);both_independent=np.isfinite(independent).all(axis=0)
  expected[:,both_independent]=np.array([np.min(independent[:,both_independent],axis=0),np.mean(independent[:,both_independent],axis=0),np.max(independent[:,both_independent],axis=0)])
  np.testing.assert_allclose(q,expected,rtol=0,atol=2e-12,equal_nan=True)
  result['metrics'][name]={'values':values,'valid':valid,'flags':flags,'range_median':q,
    'duplicate_time_points':sum(row[3] for row in series)}
 return result

def zoom_end(cell):
 bounds=support(cell)
 ends=[prefix(a,1.05).get('end_time') for a in cell]
 if bounds is None or any(not finite(t) for t in ends):return None
 return min(bounds[1],*ends)

def normalized_units_check(arm):
 variance=arm.get('target_variance');checked=0
 if not finite(variance) or variance<=0:return dict(id=arm['id'],status='variance unavailable',checked=0)
 for p in arm.get('history',[]):
  raw,norm=p.get('loss_raw'),p.get('loss_over_target_variance')
  if finite(raw) and finite(norm):
   if not math.isclose(raw,norm*variance,rel_tol=2e-12,abs_tol=2e-12):raise ValueError(f'Loss-unit mismatch: {arm["id"]}')
   checked+=1
 return dict(id=arm['id'],status='checked',checked=checked,target_variance=variance)

def draw_metric(ax,name,cell,agg):
 color=COLORS[name];style='--' if name=='A_mean' else '-'
 for arm in cell:
  x,y,flags,_=metric(arm,name);ax.plot(x,y,color=color,ls=style,lw=.65,alpha=.32)
  if len(y) and np.isfinite(y[-1]):ax.plot(x[-1],y[-1],'>',color=color,mfc='white',ms=2.5)
 if agg is not None:
  info=agg['metrics'][name];x=agg['x'];q=info['range_median']
  ax.fill_between(x,q[0],q[2],color=color,alpha=.16,lw=0)
  ax.plot(x,q[1],color=color,ls=style,lw=1.7)
 if name in ('A_min','refit'):
  # Every actual unresolved checkpoint is marked, including individual tails.
  for arm in cell:
   x,y,flags,_=metric(arm,name);bad=np.isfinite(y)&~flags
   ax.scatter(x[bad],np.full(bad.sum(),.027),transform=ax.get_xaxis_transform(),marker='|',s=22,color=COLORS['warning'],lw=1,zorder=7)
   missing=[p['time'] for p in arm.get('checkpoints',[]) if p.get('diagnostic_status')!='ok' or not finite(p.get('A_min' if name=='A_min' else 'refit_raw')) or (name=='A_min' and not finite(p.get('A_mean')))]
   missing=sorted(set(missing+[float(t) for t,v in zip(x,y) if not np.isfinite(v)]))
   ax.scatter(missing,np.full(len(missing),.027),transform=ax.get_xaxis_transform(),marker='x',s=12,color=COLORS['warning'],lw=.65,zorder=8)

def limits(cell,name,right,zoom):
 values=[]
 names=(name,'A_mean') if name=='A_min' else (name,)
 for key in names:
  for arm in cell:
   x,y,_,_=metric(arm,key);values.extend(y[(x<=right)&np.isfinite(y)].tolist())
 var=max([a.get('target_variance',1) for a in cell if finite(a.get('target_variance'))],default=1)
 low,high=min(values,default=0),max(values,default=var)
 if name=='loss' and zoom:
  span=max(high-low,.002*var,1e-12);return low-.1*span,high+.1*span
 if name=='A_min':return min(-.06,low-.02),max(1.03,high+.02)
 span=max(var,high-low,1e-10)
 return min(-.04*span,low-.025*span),max(var*1.04,high+.025*span)

def cell_axes(fig,slot,student,teacher,cell,agg,zoom):
 inner=slot.subgridspec(3,1,hspace=.13);axes=[fig.add_subplot(inner[i,0]) for i in range(3)]
 for ax in axes[:-1]:ax.tick_params(labelbottom=False)
 ends=[a['history'][-1]['time'] for a in cell if a.get('history')]
 end=zoom_end(cell) if zoom else max(ends,default=0)*1.02
 available=end is not None and end>0
 right=end if available else 1
 for ax in axes:
  ax.set_xlim(0,right);ax.grid(axis='y',color='#DCE3EE',lw=.55)
  for ratio,color,alpha in ((1.05,'prefix5',.65),(1.01,'prefix1',.17)):
   boundary=[prefix(a,ratio).get('end_time') for a in cell]
   if all(finite(t) for t in boundary):ax.axvspan(0,min(boundary),color=COLORS[color],alpha=alpha,zorder=0)
  if available:
   for a in cell:
    boundary=prefix(a,1.05).get('end_time')
    if finite(boundary):ax.axvline(boundary,color=COLORS['prefix1'],ls=':',lw=.5,alpha=.65)
  else:ax.text(.5,.5,'No paired initial window' if zoom else 'No recorded trajectory',ha='center',va='center',transform=ax.transAxes,fontsize=8)
 if available:
  for ax,name in zip(axes,('loss','A_min','refit')):
   draw_metric(ax,name,cell,agg);ax.set_ylim(*limits(cell,name,right,zoom))
  draw_metric(axes[1],'A_mean',cell,agg)
 label=LABELS.get(teacher,teacher)
 first=cell[0];var=first.get('target_variance');head=first.get('head_lr')
 label+=f" (s={fmt(first.get('scale'))}"+(f", head={fmt(head)}" if student=='swiglu' else '')+')'
 recipe=f"r={first.get('r','?')}, d={first.get('d','?')}, m={first.get('m','?')}; Var(Y)={fmt(var)}"
 counts=[sum(prefix(a,r).get('first_material_candidate') is not None for a in cell) for r in (1.01,1.05)]
 qualified=[sum(prefix(a,r).get('first_numerically_qualified_candidate') is not None for a in cell)
            if all('first_numerically_qualified_candidate' in prefix(a,r) for a in cell) else None for r in (1.01,1.05)]
 qualified_text=', '.join(f'{q}/2' if q is not None else 'unavailable' for q in qualified)
 censored=[sum(not prefix(a,r) or bool(prefix(a,r).get('right_censored')) for a in cell) for r in (1.01,1.05)]
 axes[0].set_title(f'{label}\n{recipe}',fontsize=9,pad=5)
 axes[0].text(.025,.055,f"Original criterion 1%/5%: {counts[0]}/2, {counts[1]}/2\nNumerically qualified 1%/5%: {qualified_text}\nCensored/unavailable 1%/5%: {censored[0]}/2, {censored[1]}/2",transform=axes[0].transAxes,fontsize=6.2,
              bbox=dict(facecolor='white',edgecolor='none',alpha=.78))
 for ax,text in zip(axes,('Raw MSE',r'$A_{\min}, A_{\rm mean}$','Refit raw MSE')):ax.set_ylabel(text,color='black',fontsize=8)
 axes[-1].set_xlabel('Accepted force clock'+r' $\sum h_k$' if student=='relu' else 'Adaptive flow time'+r' $\sum\Delta t_k$',fontsize=8)
 stop_lines=[]
 for arm in cell:
  end_time=arm['history'][-1]['time'] if arm.get('history') else None
  stop_lines.append(f"{arm['seed']}: {status_text(arm)}, t={fmt(end_time)}")
 axes[-1].text(.02,.94,'\n'.join(stop_lines),va='top',transform=axes[-1].transAxes,fontsize=6.1,
               bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
 return axes

def information_card(fig,slot,student,zoom,historical):
 ax=fig.add_subplot(slot);ax.set_axis_off()
 handles=[Line2D([0],[0],color=COLORS['loss'],label='Raw prediction MSE'),
  Line2D([0],[0],color=COLORS['A_min'],label=r'Minimum AGOP alignment $A_{\min}$'),
  Line2D([0],[0],color=COLORS['A_mean'],ls='--',label=r'Mean AGOP alignment $A_{\rm mean}$'),
  Line2D([0],[0],color=COLORS['refit'],label='Actual same-class refit MSE'),
  Patch(facecolor=COLORS['prefix5'],label='Observed two-seed common 5% prefix'),
  Patch(facecolor=COLORS['prefix1'],alpha=.3,label='Observed two-seed common 1% prefix'),
  Line2D([0],[0],color=COLORS['warning'],marker='|',lw=0,label='Unresolved/sensitivity flag'),
  Line2D([0],[0],color=COLORS['warning'],marker='x',lw=0,label='Missing expected diagnostic')]
 ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=8,handlelength=2.7,labelspacing=.45)
 readout='ReLU refits retain the saved bounded head class.' if student=='relu' else 'SwiGLU uses unrestricted cutoff 10⁻¹² refits.'
 text=('Bold: two planned seeds’ median.\nBands: seed range, not confidence intervals.\nFaint curves: individual observed tails.\nMedians end at each cell’s shared support.\nMissing points remain NaN gaps.\n\n'
       'Original counts use saved per-run criteria.\nQualification adds numerical screens.\nNeither is a median-curve claim.\nAll failed/censored runs remain planned.\n\n'+readout+'\nRaw MSE uses the stored teacher variance.\nClocks/readout classes differ by student.\n\n')
 text+=('Zoom ends at the two-seed common initial\n5% loss-prefix endpoint; no alignment selection.' if zoom else 'Loss-only initial 1%/5% shading uses all\nrecorded updates, retaining right-censoring.')
 if historical:text+='\n\nHISTORICAL VALIDATION: no new batch results.'
 ax.text(.02,.68,text,va='top',fontsize=7.6,linespacing=1.26)


def render_page(student,entries,aggregates,page,zoom,title,historical):
 fig=plt.figure(figsize=(17.5,13.5));outer=fig.add_gridspec(2,4,left=.065,right=.98,bottom=.075,top=.845,wspace=.35,hspace=.28)
 mode='Initial loss-only prefix zoom' if zoom else 'Full observed trajectories'
 profile=sorted({str(a.get('profiled_intercept')) for _,cell in entries for a in cell})
 heads=sorted({fmt(a.get('head_lr')) for _,cell in entries for a in cell})
 scales=sorted({fmt(a.get('scale')) for _,cell in entries for a in cell})
 fig.text(.52,.975,title,ha='center',fontsize=14)
 fig.text(.52,.945,f'{student.upper()} · {mode} · page {page}',ha='center',fontsize=12)
 extra=('; head-rate multiplier='+','.join(heads)) if student=='swiglu' else ''
 fig.text(.52,.917,'Profiled intercept='+','.join(profile)+'; Gaussian scale='+','.join(scales)+extra,ha='center',fontsize=10)
 fig.text(.52,.886,'Two planned seeds per cell; fixed per-run criteria; raw teacher mean retained; E[Y²]=1',ha='center',fontsize=10)
 axes=[]
 for i,(key,cell) in enumerate(entries):axes.extend(cell_axes(fig,outer[i//4,i%4],student,key[1],cell,aggregates[key],zoom))
 information_card(fig,outer[1,3],student,zoom,historical)
 if len(entries)<7:
  for i in range(len(entries),7):
   ax=fig.add_subplot(outer[i//4,i%4]);ax.set_axis_off()
 fig.text(.065,.038,'Variance differs across links; raw MSE and MSE/Var(Y) coincide only when Var(Y)=1. No model evaluation or training is performed by this renderer.',fontsize=9)
 if historical:fig.text(.065,.016,'Historical fixture validation only. Source cohorts and any deterministic seed selection are recorded in provenance; these are not breadth14 observations.',fontsize=9,color=COLORS['warning'])
 fig.canvas.draw()
 return fig


def write_plots(summary_path,out,title=None,historical=False):
 summary_path=summary_path.resolve();out=out.resolve()
 if not out.is_relative_to(HERE):raise ValueError('Output must remain under breadth14/analysis')
 if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve prior plots; choose a fresh output directory')
 before=sha(summary_path);renderer_before=sha(Path(__file__));data=json.loads(summary_path.read_text())
 if data.get('snapshot_usable') is False:raise ValueError('Summary snapshot is unusable; obtain a stable saved snapshot')
 if data.get('inputs_unchanged') is False:raise ValueError('Summary input files changed while being read; obtain a stable saved snapshot')
 groups=grouping(data,historical)
 if not groups:raise ValueError('No planned canonical cells to plot')
 if historical and not title:raise ValueError('Historical validation requires an explicit truthful title')
 title=title or 'Fourteen scalar teachers: saved population-learning results'
 configure();aggregates={key:aggregate(cell) for key,cell in groups.items()}
 unit_checks=[normalized_units_check(a) for cell in groups.values() for a in cell]
 out.mkdir(parents=True,exist_ok=True);outputs=[]
 for student in sorted({key[0] for key in groups}):
  keys=sorted([key for key in groups if key[0]==student],key=lambda k:(CANONICAL.index(k[1]) if k[1] in CANONICAL else 99,k[2]))
  entries=[(key,groups[key]) for key in keys]
  for zoom in (False,True):
   mode='initial_prefix_zoom' if zoom else 'full';pdfpath=out/f'{student}_{mode}.pdf'
   with PdfPages(pdfpath) as pdf:
    for i,start in enumerate(range(0,len(entries),7),1):
     fig=render_page(student,entries[start:start+7],aggregates,i,zoom,title,historical)
     stem=f'{student}_{mode}_page{i:02d}'
     for ext in ('png','svg'):
      path=out/(stem+'.'+ext);fig.savefig(path,dpi=165,facecolor='white');outputs.append(path)
     pdf.savefig(fig,facecolor='white');plt.close(fig)
   outputs.append(pdfpath)
 for i,(key,agg) in enumerate(aggregates.items()):
  if agg is not None:
   path=out/f'aggregate_{i:02d}.npz';np.savez_compressed(path,x=agg['x'],**{f'{metric}_{name}':v for metric,info in agg['metrics'].items() for name,v in info.items()});outputs.append(path)
 provenance_hashes=data.get('input_sha256',data.get('input_hashes',{}))
 unavailable=[]
 if isinstance(provenance_hashes,dict):
  for p,h in provenance_hashes.items():
   if not isinstance(h,str):continue
   path=Path(p)
   if path.exists():
    if sha(path)!=h:raise RuntimeError(f'Saved source input changed: {p}')
   else:unavailable.append(p)
 assert sha(summary_path)==before
 assert sha(Path(__file__))==renderer_before, 'Renderer changed during rendering'
 cells=[]
 for key,cell in groups.items():
  agg=aggregates[key]
  cells.append(dict(student=key[0],teacher=key[1],cell=key[2],planned_ids=[a['id'] for a in cell],planned_seeds=[a['seed'] for a in cell],stops=[dict(id=a['id'],reason=status_text(a),process_state=a.get('process_state')) for a in cell],
   support=agg['support'] if agg else None,initial_zoom_end=zoom_end(cell),
   metric_support=None if agg is None else {name:dict(paired_finite_points=int(info['valid'].all(axis=0).sum()),duplicate_time_points=info['duplicate_time_points']) for name,info in agg['metrics'].items()},
   per_run_criteria_passed_through=[dict(id=a['id'],prefixes=a.get('prefixes',{})) for a in cell]))
 qa=dict(pass_check=True,no_model_evaluation=True,exactly_two_planned_seeds_per_cell=True,
  medians_independently_checked=True,seed_ranges_are_not_confidence_intervals=True,
  missing_diagnostics_preserved_as_nan=True,no_interpolation_across_nan=True,no_extrapolation=True,
  initial_zoom_selected_only_from_loss_prefixes=True,teachers_and_students_not_pooled=True,
  raw_normalized_loss_checks=unit_checks,visual_review='pending',cells=cells)
 (out/'PLOT_QA.json').write_text(json.dumps(clean(qa),indent=2,allow_nan=False)+'\n')
 provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),renderer_sha256=renderer_before,
  summary_path=str(summary_path),summary_sha256=before,source_hashes_from_summary=provenance_hashes,
  source_paths_unavailable_for_local_recheck=unavailable,historical_validation=historical,
  fixture_provenance=data.get('fixture_provenance'),excluded_supplemental_ids=[a['id'] for a in data['arms'] if a.get('supplemental',False)],
  colors=COLORS,font='DejaVu Serif',label_color='black',output_sha256={p.name:sha(p) for p in outputs})
 (out/'PLOT_PROVENANCE.json').write_text(json.dumps(clean(provenance),indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(output=str(out),cells=len(cells),planned_arms=2*len(cells),historical_validation=historical,no_model_evaluation=True)))
 return qa,provenance


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--summary',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--title');ap.add_argument('--historical-validation',action='store_true')
 args=ap.parse_args();write_plots(args.summary,args.output,args.title,args.historical_validation)

if __name__=='__main__':main()
