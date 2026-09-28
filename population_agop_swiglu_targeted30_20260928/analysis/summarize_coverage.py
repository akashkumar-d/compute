"""Read-only all-arm criteria audit. Reads JSON/JSONL/NPY only; imports no models."""
from __future__ import annotations
import argparse, collections, datetime, hashlib, io, json, math, os, zipfile
from pathlib import Path
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import numpy as np

RATIOS=(1.01,1.05)
GUARD=1e-9
REFIT_TOL=1e-7
SCHEMA='breadth14_saved_coverage_v1'

def finite(x):return isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_)) and math.isfinite(x)
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    return x

def tri(checks):
    if any(x is False for x in checks):return False
    return True if checks and all(x is True for x in checks) else None

def test_numeric(x,predicate):return bool(predicate(x)) if finite(x) else None

def json_unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate JSON key: '+k)
        out[k]=v
    return out

class Reader:
    def __init__(self):self.hashes={};self.errors=[];self.missing=[];self.reread_conflicts=[]
    def remember_hash(self,path,value):
        key=str(Path(path).resolve())
        if key in self.hashes and self.hashes[key]!=value:
            if key not in self.reread_conflicts:self.reread_conflicts.append(key)
        else:self.hashes.setdefault(key,value)
    def read(self,path,kind='json',default=None):
        path=Path(path).resolve()
        if not path.exists():self.missing.append(str(path));return default
        try:
            raw=path.read_bytes();self.remember_hash(path,hashlib.sha256(raw).hexdigest())
            if kind=='json':return json.loads(raw,object_pairs_hook=json_unique)
            if kind=='jsonl':return [json.loads(s,object_pairs_hook=json_unique) for s in raw.splitlines() if s.strip()]
            if kind=='npy':return np.load(io.BytesIO(raw),allow_pickle=False)
            if kind=='npz':
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    return {k:z[k].copy() for k in ('history','accepted','step') if k in z}
            raise ValueError(kind)
        except Exception as e:self.errors.append({'path':str(path),'error':repr(e)});return default
    def file_hash(self,path):
        path=Path(path).resolve()
        if not path.exists():self.missing.append(str(path));return None
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        value=h.hexdigest();self.remember_hash(path,value);return value
    def verify_hashes(self,folder,mapping):
        issues=[]
        for name,expected in mapping.items():
            p=(folder/name).resolve()
            if not p.is_relative_to(folder.resolve()):issues.append('recorded_file_escapes_arm:'+name);continue
            actual=self.file_hash(p)
            if actual is None:issues.append('recorded_file_missing:'+name)
            elif actual!=expected:issues.append('recorded_file_hash_mismatch:'+name)
        return issues
    def snapshot_metadata(self,path):
        if not path.exists():self.missing.append(str(path.resolve()));return None
        self.file_hash(path)
        try:
            shapes={}
            with zipfile.ZipFile(path) as z:
                for name in ('t','P','V','a'):
                    with z.open(name+'.npy') as f:
                        version=np.lib.format.read_magic(f)
                        shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                        if dtype.hasobject:raise ValueError('Object dtype in snapshot')
                        shapes[name]=list(shape)
            with np.load(path,allow_pickle=False) as z:times=z['t'].tolist()
            return {'shapes':shapes,'times':times}
        except Exception as e:self.errors.append({'path':str(path),'error':repr(e)});return None
    def unchanged(self):
        self.changed_paths=list(self.reread_conflicts)
        for name,expected in self.hashes.items():
            p=Path(name);h=hashlib.sha256()
            if p.exists():
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if (not p.exists() or h.hexdigest()!=expected) and name not in self.changed_paths:self.changed_paths.append(name)
        return not self.changed_paths

def exact_prefix(history,ratio):
    """No interpolation, rounding epsilon, or first-to-last ratio substitution."""
    if not history:return dict(end_step=None,end_time=None,right_censored=True,exit_observed=False,invalid_loss=False,actual_ratio=None,first_exit_step=None)
    lo=math.inf;hi=-math.inf;end=-1;invalid=False;crossed=False
    for p in history:
        loss=p['loss_raw']
        if not finite(loss) or loss<=0:invalid=True;break
        lo=min(lo,loss);hi=max(hi,loss)
        if hi>ratio*lo:crossed=True;break
        end=p['step']
    used=[p['loss_raw'] for p in history if p['step']<=end]
    return dict(end_step=end,end_time=history[end]['time'] if end>=0 else None,right_censored=not crossed,
                exit_observed=crossed,invalid_loss=invalid,actual_ratio=max(used)/min(used) if used else None,
                endpoint_loss=used[-1] if used else None,first_exit_step=end+1 if crossed else None,
                first_exit_time=history[end+1]['time'] if crossed else None,ratio_limit=ratio)

def config_of(job):
    c=job['config'];c=c[0] if isinstance(c,list) else c
    return c,c.get('args',c)

def process_state(cid,status,complete,has_result):
    receipts={p['id']:p for p in status.get('completed',[])+status.get('finished',[])+complete.get('outcomes',[])}
    active={p['id'] if isinstance(p,dict) else p for p in status.get('active',[])}
    if cid in active:return 'active_snapshot',receipts.get(cid)
    if cid in status.get('pending',[]):return 'pending_snapshot',receipts.get(cid)
    if cid in receipts:
        r=receipts[cid];code=r.get('returncode',r.get('exit_code'))
        return ('failed_or_terminated' if code not in (None,0) else 'exit_zero' if code==0 and has_result else 'exit_zero_missing_result' if code==0 else 'unstarted_or_unreconciled_receipt'),r
    return ('result_present_launcher_unreconciled' if has_result else 'not_started'),None

def expected_target(name,rank,moments):
    scalar=moments.get('links',{}).get(name)
    if scalar:
        mu,s2=scalar['raw_mean'],scalar['raw_second'];den=s2+(rank-1)*mu*mu
        ey=math.sqrt(rank)*mu/math.sqrt(den)
        return ey,(s2-mu*mu)/den,'frozen_scalar_moments'
    if name in ('h2','he2','h3','he3','h4','h5','sine','tanh','h3_plus_h5','h3_plus_sine','damped_2_3','three_atom'):return 0.,1.,'documented_legacy_zero_mean'
    if name in ('relu','abs'):
        mu,s2=(1/math.sqrt(2*math.pi),.5) if name=='relu' else (math.sqrt(2/math.pi),1.)
        den=s2+(rank-1)*mu*mu;return math.sqrt(rank)*mu/math.sqrt(den),(s2-mu*mu)/den,'documented_legacy_exact_moments'
    return None,None,'unknown'

def empty_point(step,time=None):
    return dict(step=step,time=time,loss_raw=None,A_min=None,A_mean=None,A_top=None,refit_raw=None,
                refit_low_raw=None,refit_high_raw=None,agop_screen=None,refit_screen=None,
                diagnostic_status='not_evaluated',issues=[],quadrature_double_checked=False)

def relu_point(row):
    p=empty_point(row['step']);g=row.get('agop_diagnostic',{});f=row.get('refit_info',{})
    p.update(loss_raw=row.get('loss'),A_min=row.get('A_min'),A_mean=row.get('A_mean'),A_top=row.get('A_top'),
             refit_raw=row.get('refit'),refit_low_raw=row.get('refit_lower'),refit_high_raw=row.get('refit'),diagnostic_status='ok',
             saved_agop_valid=row.get('agop_valid'),rank_relative_eigenvalue=row.get('relative_lambda_r'),rank_relative_gap=row.get('relative_gap'),
             refit_gap_raw=row.get('refit_gap'),refit_optimizer_success=f.get('success'),refit_first_order_gap=f.get('first_order_gap'),
             refit_gap_tolerance=f.get('gap_tolerance'),refit_l1_cap=f.get('l1_cap'),refit_l2_cap=f.get('l2_cap'),
             quadrature_warning_count=g.get('quadrature_warning_count'),arithmetic_screen=g.get('numerical_screen_passed'),
             original_ratio_screens_passed=g.get('original_ratio_screens_passed'),substantial_negative_eigenvalue=g.get('substantial_negative_eigenvalue'))
    p['agop_screen']=tri([row.get('agop_valid'),g.get('numerical_screen_passed'),g.get('original_ratio_screens_passed'),
        test_numeric(row.get('relative_lambda_r'),lambda v:v>GUARD),test_numeric(row.get('relative_gap'),lambda v:v>GUARD),
        g.get('substantial_negative_eigenvalue') is False if 'substantial_negative_eigenvalue' in g else None])
    gap=f.get('gap_tolerance'); valid_bounds=all(finite(p[k]) for k in ('refit_raw','refit_low_raw'))
    p['raw_refit_nonnegative_screen']=test_numeric(p['refit_raw'],lambda v:v>=-1e-10)
    p['refit_screen']=tri([f.get('success'),test_numeric(row.get('refit_gap'),lambda v:0<=v<=REFIT_TOL),
        test_numeric(gap,lambda v:0<v<=REFIT_TOL),test_numeric(f.get('first_order_gap'),lambda v:finite(gap) and v<=gap),
        bool(p['refit_low_raw']<=p['refit_raw']+1e-12 and p['refit_raw']>=-1e-10) if valid_bounds else None])
    if finite(p['refit_low_raw']) and finite(p['refit_raw']) and p['refit_low_raw']>p['refit_raw']+1e-12:p['issues'].append('refit_interval_order_invalid')
    if finite(p['refit_low_raw']) and finite(p['refit_raw']) and finite(p['refit_gap_raw']) and p['refit_raw']-p['refit_low_raw']>p['refit_gap_raw']+1e-8*max(1.,abs(p['refit_raw'])):p['issues'].append('refit_interval_gap_inconsistent')
    if finite(f.get('numerical_lower_bound')) and finite(p['refit_low_raw']) and abs(f['numerical_lower_bound']-p['refit_low_raw'])>1e-10:p['issues'].append('row_vs_optimizer_lower_bound_mismatch')
    if finite(f.get('objective_gap_bound')) and finite(p['refit_gap_raw']) and abs(f['objective_gap_bound']-p['refit_gap_raw'])>1e-10:p['issues'].append('row_vs_optimizer_gap_mismatch')
    return p

def swiglu_point(source,row,variance):
    p=empty_point(source['step'],source.get('t'));ok=row.get('status')=='ok';pinv=row.get('pinv',{})
    risks=[x.get('actual_mse') for x in pinv.values()];residuals=[x.get('relative_normal_residual') for x in pinv.values()]
    complete=all(any(finite(float(k)) and float(k)==v for k in pinv) for v in (1e-8,1e-10,1e-12,1e-14)) if pinv else False
    risks_ok=bool(complete and all(finite(x) for x in risks));res_ok=bool(complete and all(finite(x) for x in residuals))
    risk=next((v.get('actual_mse') for k,v in pinv.items() if float(k)==1e-12),None)
    spread=max(risks)-min(risks) if risks_ok else None
    raw=lambda x:x*variance if finite(x) and finite(variance) else None
    p.update(loss_raw=raw(source.get('L')),A_min=row.get('agop_Amin') if ok else None,A_mean=row.get('agop_A') if ok else None,A_top=row.get('agop_Atop') if ok else None,
        refit_raw=raw(risk) if ok else None,refit_low_raw=raw(min(risks)) if risks_ok and ok else None,refit_high_raw=raw(max(risks)) if risks_ok and ok else None,
        diagnostic_status=row.get('status','not_evaluated'),saved_agop_valid=row.get('agop_full_rank_resolved'),rank_relative_eigenvalue=row.get('agop_rank_r_eigenvalue_relative'),
        rank_relative_gap=row.get('agop_rank_r_relative_gap'),original_guard=row.get('agop_resolution_guard'),
        refit_sensitivity_spread_normalized=spread,refit_normal_residual_max=max(residuals) if res_ok else None,
        equilibrated_lambda_ratio=row.get('equilibrated_lambda_ratio'),cutoff_count=len(pinv),
        diagnostic_recorded_loss=row.get('L'),diagnostic_recorded_time=row.get('t'))
    p['agop_screen']=tri([ok if row else None,row.get('agop_full_rank_resolved'),row.get('agop_psd_resolved'),
        test_numeric(row.get('agop_resolution_guard'),lambda x:x==GUARD),
        test_numeric(row.get('agop_rank_r_eigenvalue_relative'),lambda x:x>GUARD),test_numeric(row.get('agop_rank_r_relative_gap'),lambda x:x>GUARD)])
    p['raw_refit_nonnegative_screen']=test_numeric(min(risks) if risks_ok else None,lambda v:v>=-1e-8)
    p['refit_screen']=tri([ok if row else None,risks_ok if row else None,res_ok if row else None,
        test_numeric(spread,lambda x:x<=REFIT_TOL),test_numeric(min(risks) if risks_ok else None,lambda x:x>=-1e-8),
        test_numeric(max(residuals) if res_ok else None,lambda x:x<=REFIT_TOL),test_numeric(row.get('equilibrated_lambda_ratio'),lambda x:x>0)])
    return p

def load_arm(job,folder,reader,moments):
    outer,c=config_of(job);engine=job['engine'];name=c.get('teacher',c.get('link'));rank=c.get('r',outer.get('rank'))
    a=dict(id=job['id'],cell=job.get('cell',engine+'_'+name),student=engine,teacher={'he2':'h2','he3':'h3'}.get(name,name),seed=c['seed'],r=rank,d=c['d'],m=c['m'],
        scale=c.get('scale',c.get('s')),head_lr=c.get('head_lr',1.) if engine=='swiglu' else None,profiled_intercept=bool(c.get('profile_intercept',False)) if engine=='relu' else True,
        supplemental=bool(job.get('supplemental',False)),algorithm=job.get('algorithm'),config=job['config'],execution_directory=str(folder),history=[],checkpoints=[],issues=[],prefixes={})
    expected_mean,expected_variance,provenance=expected_target(a['teacher'],rank,moments)
    expected=set();points={};duplicates=[];norm_samples=[];time_samples=[];dense_partial=False
    if engine=='relu':
        meta=reader.read(folder/'meta.json',default={});done=reader.read(folder/'DONE.json',default={});train=reader.read(folder/'TRAIN_RESULT.json',default={})
        rows=reader.read(folder/'rows.jsonl','jsonl',[]);loss=reader.read(folder/'loss_every_step.npy','npy');accepted=reader.read(folder/'accepted_h.npy','npy')
        if loss is None:
            resume=reader.read(folder/'resume.npz','npz',{});loss=resume.get('history',np.array([]));accepted=resume.get('accepted',np.array([]));dense_partial=bool(len(loss))
        state_manifest=reader.read(folder/'checkpoint_manifest.json',default={})
        expected.update(int(k) for k in state_manifest)
        expected.update(done.get('required_diagnostic_steps',[]))
        state_issues=[]
        for state in state_manifest.values():state_issues.extend(reader.verify_hashes(folder,{state['path']:state['sha256']}))
        state_issues.extend(reader.verify_hashes(folder,done.get('output_sha256',{})))
        a['issues'].extend(state_issues)
        a['snapshot_integrity_verified']=bool(state_manifest) and not state_issues
        a['state_manifest_steps']=len(state_manifest)
        variance=next((x['target_variance'] for x in rows if finite(x.get('target_variance'))),expected_variance)
        mean=next((x['teacher_mean'] for x in rows if finite(x.get('teacher_mean'))),expected_mean)
        time=np.r_[0.,np.cumsum(accepted)] if accepted is not None else np.array([])
        if len(loss) and len(time)!=len(loss):a['issues'].append('accepted_clock_length_mismatch')
        for i,l in enumerate(loss):a['history'].append(dict(step=i,time=float(time[i]) if i<len(time) else None,loss_raw=float(l),loss_over_target_variance=float(l)/variance if finite(variance) and variance>0 else None))
        for row in rows:
            step=row['step'];p=relu_point(row);p['time']=float(time[step]) if step<len(time) else None
            if step in points:duplicates.append(step)
            points[step]=p
            if finite(row.get('target_variance')):norm_samples.append(row['target_variance'])
            if finite(row.get('time_eta')) and p['time'] is not None:time_samples.append((step,2*row['time_eta']/c['m']-p['time']))
        reason=train.get('stop_reason',done.get('training_stop_reason'));saved=meta.get('config')
        a.update(has_result=bool(done),completion_record=done,training_record=train,recorded_diagnostics_complete=done.get('diagnostic_complete'),time_unit='accepted_force_time',
                 normalization_note='ReLU loss and bounded same-budget refit are raw MSE.',refit_comparator='normalized bounded readout: l2<=32,l1<=64*sqrt(r); profiled intercept follows learner',
                 training_wall_censored=train.get('training_censored',done.get('training_censored',bool(len(loss)) and (reason=='training_budget' or str(reason).startswith('signal_')))),stop_reason=reason,
                 declared_horizon_reached=train.get('declared_horizon_or_loss_target_reached',train.get('complete',done.get('training_complete',False))),source_hashes_recorded=meta.get('source_sha256',{}),
                 teacher_moment_method=meta.get('teacher_moment_method'),configuration_matches=saved==job['config'] if saved else None)
        a['required_grid_provenance']='checkpoint_manifest_and_DONE_required_steps' if state_manifest or done.get('required_diagnostic_steps') else 'observed_rows_only'
        if train.get('observed_steps') is not None and train['observed_steps']!=len(loss)-1:a['issues'].append('dense_history_length_vs_training_record_mismatch')
    else:
        report=reader.read(folder/'result.json',default={});raw=reader.read(folder/(job['id']+'.json'),default={})
        if not raw:raw=reader.read(folder/'partials'/(job['id']+'.json'),default={});dense_partial=bool(raw)
        ds=report.get('diagnostics')
        if ds is None:ds=reader.read(folder/'diagnostics_partial.json',default={}).get('rows',[])
        rawrows=raw.get('rows',[]);db={};rowmap={}
        for x in ds:
            if x['step'] in db:duplicates.append(x['step'])
            db[x['step']]=x
        for x in rawrows:
            if x['step'] in rowmap:duplicates.append(x['step'])
            rowmap[x['step']]=x
        variance=raw.get('V');mean=raw.get('EY')
        for i,(t,l) in enumerate(raw.get('Lhist',[])):a['history'].append(dict(step=i,time=t,loss_raw=l*variance if finite(variance) else None,loss_over_target_variance=l))
        expected.update(rowmap);expected.update(db)
        raw_hash_issues=reader.verify_hashes(folder,report.get('raw_files',{}));a['issues'].extend(raw_hash_issues)
        snapshot=folder/(job['id']+'_snaps.npz')
        if not snapshot.exists() and dense_partial:snapshot=folder/'partials'/(job['id']+'_snaps.npz')
        sm=reader.snapshot_metadata(snapshot);a['snapshot_metadata']=sm
        a['snapshot_integrity_verified']=False
        if rawrows:
            if sm is None:a['issues'].append('snapshot_metadata_missing_or_invalid')
            else:
                n=len(rawrows);expected_shapes={'t':[n],'P':[n,c['d']+1,c['m']],'V':[n,c['d']+1,c['m']],'a':[n,c['m']]}
                if sm['shapes']!=expected_shapes:a['issues'].append('snapshot_shape_count_mismatch')
                if len(sm['times'])!=n or any(not finite(t) or abs(t-r['t'])>1e-9*max(1,abs(r['t'])) for t,r in zip(sm['times'],rawrows)):a['issues'].append('snapshot_training_row_time_mismatch')
                a['snapshot_integrity_verified']=bool(report.get('raw_files')) and sm['shapes']==expected_shapes and len(sm['times'])==n and not raw_hash_issues and not any(k.startswith('snapshot_') for k in a['issues'])
        doubled={x['step'] for x in report.get('quadrature_double',[]) if x.get('status')=='ok'}
        for step in expected:
            original=rowmap.get(step)
            if original is None:
                p=empty_point(step);p['issues'].append('diagnostic_without_saved_training_row')
            else:
                p=swiglu_point(original,db.get(step,{}),variance);p['quadrature_double_checked']=step in doubled
                if finite(db.get(step,{}).get('normalization_V')):norm_samples.append(db[step]['normalization_V'])
                if p.get('diagnostic_recorded_loss') is not None and finite(original.get('L')):
                    if abs(p['diagnostic_recorded_loss']-original['L'])>1e-9:p['issues'].append('diagnostic_training_loss_mismatch')
                if p.get('diagnostic_recorded_time') is not None and abs(p['diagnostic_recorded_time']-original['t'])>1e-9:p['issues'].append('diagnostic_training_time_mismatch')
            points[step]=p
        term=report.get('termination',raw.get('termination',{}));reason=term.get('reason');saved=report.get('cfg',raw.get('cfg'))
        a.update(has_result=bool(report),completion_record={k:v for k,v in report.items() if k not in ('diagnostics','quadrature_double')},training_record=term,
                 recorded_diagnostics_complete=bool(ds) and len(ds)==report.get('checkpoint_count',len(rawrows)) and all(x.get('status')=='ok' for x in ds),
                 time_unit='adaptive_population_flow_time',normalization_note='SwiGLU dense loss and every pinv actual_mse multiplied once by saved Var(Y) to raw MSE.',
                 refit_comparator='unrestricted readout; initial-min/current-max across four pseudoinverse cutoffs; sensitivity envelope, not a mathematical lower bound',
                 training_wall_censored=reason in ('external_stop','exception_durable_partial') or (bool(a['history']) and not reason),stop_reason=reason,
                 declared_horizon_reached=reason in ('t_max','loss_stop','max_steps'),source_hashes_recorded=report.get('source_files_sha256',{}),
                 configuration_matches=all(saved.get(k)==v for k,v in c.items()) if saved else None,required_grid_provenance='all_saved_training_rows')
        if report.get('all_update_count') is not None and report['all_update_count']!=len(a['history']):a['issues'].append('dense_history_length_vs_result_mismatch')
        if report.get('checkpoint_count') is not None and report['checkpoint_count']!=len(rawrows):a['issues'].append('saved_checkpoint_count_mismatch')
    a.update(target_mean=mean,target_variance=variance,target_second_moment=1.,target_variance_expected=expected_variance,target_mean_expected=expected_mean,
             moment_provenance=provenance,dense_history_partial=dense_partial)
    if finite(variance) and finite(expected_variance) and abs(variance-expected_variance)>1e-9:a['issues'].append('saved_variance_vs_frozen_moments_mismatch')
    if finite(mean) and finite(expected_mean) and abs(mean-expected_mean)>1e-9:a['issues'].append('saved_mean_vs_frozen_moments_mismatch')
    if any(abs(v-variance)>1e-10 for v in norm_samples):a['issues'].append('inconsistent_per_state_variance')
    if any(abs(diff)>1e-8*max(1,abs(a['history'][step]['time'])) for step,diff in time_samples):a['issues'].append('recorded_clock_vs_accepted_clock_mismatch')
    if duplicates:a['issues'].append('duplicate_checkpoint_steps');a['duplicate_steps']=sorted(set(duplicates))
    if a['configuration_matches'] is False:a['issues'].append('saved_config_mismatch')
    if a['history'] and (not finite(variance) or variance<=0):a['issues'].append('invalid_target_variance')
    if any(not finite(p['loss_raw']) or p['loss_raw']<=0 for p in a['history']):a['issues'].append('nonpositive_or_nonfinite_dense_loss')
    times=[p['time'] for p in a['history']]
    if times and (not finite(times[0]) or abs(times[0])>1e-12):a['issues'].append('dense_history_does_not_start_at_initial_time')
    if times and (any(not finite(t) for t in times) or any(times[i]<=times[i-1] for i in range(1,len(times)))):a['issues'].append('invalid_or_nonmonotone_dense_clock')
    for ratio in RATIOS:
        pr=exact_prefix(a['history'],ratio);a['prefixes'][str(ratio)]=pr
        if pr['end_step'] is not None and pr['end_step']>=0:expected.add(pr['end_step']);expected.add(0)
    expected.update(points)
    for step in sorted(expected):
        if not isinstance(step,int) or step<0:a['issues'].append('invalid_checkpoint_step');continue
        p=points.get(step,empty_point(step,a['history'][step]['time'] if step<len(a['history']) else None))
        if step>=len(a['history']):p['issues'].append('checkpoint_outside_dense_history')
        elif p['diagnostic_status']=='ok':
            if not finite(p['loss_raw']) or abs(p['loss_raw']-a['history'][step]['loss_raw'])>1e-9:p['issues'].append('diagnostic_dense_loss_mismatch')
            if not finite(p['time']) or abs(p['time']-a['history'][step]['time'])>1e-8*max(1,abs(p['time'] or 0)):p['issues'].append('diagnostic_dense_time_mismatch')
        a['checkpoints'].append(p)
    a['diagnostics_complete']=bool(a['checkpoints']) and all(p['diagnostic_status']=='ok' and not p['issues'] for p in a['checkpoints'])
    a['diagnostic_censored']=not a['diagnostics_complete'] if a['history'] else None
    a['observed_updates']=len(a['history'])-1 if a['history'] else None
    a['initial_loss_raw']=a['history'][0]['loss_raw'] if a['history'] else None;a['final_loss_raw']=a['history'][-1]['loss_raw'] if a['history'] else None
    a['final_time']=a['history'][-1]['time'] if a['history'] else None
    a['physical_horizon_reached']=bool(finite(a['final_time']) and a['final_time']>=c.get('force_time_max',c.get('t_max',math.inf)))
    a['update_cap_reached']=bool(a['observed_updates'] is not None and a['observed_updates']>=c.get('steps',c.get('max_steps',math.inf)))
    a['loss_stop_reached']=bool(a['history'] and (a['final_loss_raw']<=c.get('loss_stop',-math.inf) if engine=='relu' else a['history'][-1]['loss_over_target_variance']<=c.get('L_stop',-math.inf)))
    a['declared_horizon_reached']=bool(a['declared_horizon_reached'] or a['physical_horizon_reached'] or a['update_cap_reached'] or a['loss_stop_reached'])
    return a

def evaluate_arm(a):
    for p in a['checkpoints']:
        if finite(p.get('refit_low_raw')) and finite(p.get('refit_high_raw')) and p['refit_low_raw']>p['refit_high_raw']+1e-12:
            if 'refit_interval_order_invalid' not in p['issues']:p['issues'].append('refit_interval_order_invalid')
    initial=next((p for p in a['checkpoints'] if p['step']==0),None);var=a['target_variance']
    integrity=bool(not a['issues'] and a['configuration_matches'] is True and finite(var) and var>0)
    a['material_refit_threshold_raw']=.1*var if finite(var) and var>0 else None
    a['initial_refit_feasible_upper_raw']=initial['refit_high_raw'] if initial else None
    a['effect_size_eligibility']=('unknown_initial_refit_or_guard' if not initial or initial['refit_screen'] is not True or not finite(var) or var<=0 else 'initial_refit_already_below_material_threshold' if finite(initial['refit_high_raw']) and initial['refit_high_raw']<.1*var else 'initial_refit_not_excluded_by_threshold')
    a['effect_size_eligibility_note']='Actual saved feasible initial refit only; oracle affine/nonlinear residual is not used as a feasibility bound. Denominator and criterion unchanged.'
    a['comparison_class']='bounded_normalized_readout_numerical_lower' if a['student']=='relu' else 'unrestricted_cutoff_sensitivity_envelope'
    a['certificate']='numerical_solver_bound_not_formal_population_certificate' if a['student']=='relu' else 'notformaloptimalriskbound'
    for p in a['checkpoints']:
        p['A_min_gain']=p['A_min']-initial['A_min'] if initial and finite(p['A_min']) and finite(initial['A_min']) else None
        p['unadjusted_refit_gain_lower_raw']=initial['refit_low_raw']-p['refit_high_raw'] if initial and finite(initial['refit_low_raw']) and finite(p['refit_high_raw']) else None
        p['refit_gain_lower_raw']=initial['refit_low_raw']-max(0.,p['refit_high_raw']) if initial and finite(initial['refit_low_raw']) and finite(p['refit_high_raw']) else None
        p['roundoff_risk_floor_for_criterion']=bool(finite(p['refit_high_raw']) and p['refit_high_raw']<0)
        p['risk_floor_note']='Raw values retained; risk within tolerance belowzero is floored only for conservative improvement arithmetic, never used to manufacture threshold crossing.'
        p['refit_gain_lower_over_variance']=p['refit_gain_lower_raw']/var if finite(p['refit_gain_lower_raw']) and finite(var) and var>0 else None
        p['refit_gain_raw']=initial['refit_raw']-p['refit_raw'] if initial and finite(initial['refit_raw']) and finite(p['refit_raw']) else None
        p['cutoff_envelope_gain_raw']=p['refit_gain_lower_raw'] if a['student']=='swiglu' else None
        p['numerical_lower_gain_raw']=p['refit_gain_lower_raw'] if a['student']=='relu' else None
        p['comparison_class']=a['comparison_class'];p['certificate']=a['certificate']
        p['legacy_material_candidate_before_refit_qualification']=bool(integrity and initial and not initial['issues'] and not p['issues'] and initial['agop_screen'] is True and p['agop_screen'] is True and finite(p['A_min_gain']) and p['A_min_gain']>=.5 and finite(p['refit_gain_lower_raw']) and p['refit_gain_lower_raw']>=.1*var)
        p['same_checkpoint_material_candidate_numerically_qualified']=bool(integrity and a.get('snapshot_integrity_verified',True) is True and initial and not initial['issues'] and not p['issues'] and initial['agop_screen'] is True and p['agop_screen'] is True and initial['refit_screen'] is True and p['refit_screen'] is True and finite(p['A_min_gain']) and p['A_min_gain']>=.5 and finite(p['refit_gain_lower_raw']) and p['refit_gain_lower_raw']>=.1*var)
        p['same_checkpoint_material_candidate_original']=bool(p['legacy_material_candidate_before_refit_qualification'] and a['effect_size_eligibility']!='initial_refit_already_below_material_threshold' and initial and initial.get('raw_refit_nonnegative_screen') is True and p.get('raw_refit_nonnegative_screen') is True)
        p['same_checkpoint_material_candidate']=p['same_checkpoint_material_candidate_original']
        p['same_checkpoint_material_candidate_numerically_qualified'] &= p['same_checkpoint_material_candidate_original']
        later=[x for x in a['history'] if x['step']>p['step'] and finite(x['loss_raw'])]
        best=min(later,key=lambda x:x['loss_raw'],default=None)
        p['quadrature_followup_audit_status']='saved_double_check_present_requires_tolerance_review' if p['quadrature_double_checked'] else 'not_recomputed_at_this_checkpoint'
        p['later_loss_min_raw']=best['loss_raw'] if best else None;p['later_loss_min_step']=best['step'] if best else None
        p['later_loss_drop_raw']=p['loss_raw']-best['loss_raw'] if best and finite(p['loss_raw']) else None
        p['later_loss_drop_at_least_0p1_variance']=bool(finite(p['later_loss_drop_raw']) and finite(var) and p['later_loss_drop_raw']>=.1*var)
    for key,pr in a['prefixes'].items():
        subset=[p for p in a['checkpoints'] if pr['end_step'] is not None and p['step']<=pr['end_step']]
        candidates=[p for p in subset if p['same_checkpoint_material_candidate']]
        qualified=[p for p in candidates if p['same_checkpoint_material_candidate_numerically_qualified']]
        missing=[p['step'] for p in subset if p['diagnostic_status']!='ok']
        unresolved=[p['step'] for p in subset if p['diagnostic_status']=='ok' and (p['agop_screen'] is not True or p['refit_screen'] is not True or p['issues'])]
        assessment=('observed_material_candidate' if candidates else 'not_started' if not a['history'] else 'input_inconsistent' if not integrity else 'initial_refit_already_below_material_threshold' if a['effect_size_eligibility']=='initial_refit_already_below_material_threshold' else 'window_right_censored' if pr['right_censored'] else 'required_diagnostic_grid_unknown' if a.get('required_grid_provenance')=='observed_rows_only' else 'diagnostics_missing' if missing else 'numerically_unresolved' if unresolved else 'not_observed_on_fully_diagnosed_saved_grid')
        pr.update(assessment=assessment,original_criterion_assessment=assessment,numerical_qualification_assessment='qualified_candidate_observed' if qualified else 'original_candidate_numerically_unqualified' if candidates else assessment,candidate_count=len(candidates),qualified_candidate_count=len(qualified),first_numerically_qualified_candidate=qualified[0] if qualified else None,first_qualified_complete_sequence=next((p for p in qualified if p['later_loss_drop_at_least_0p1_variance']),None),first_material_candidate=candidates[0] if candidates else None,
                  first_complete_sequence=next((p for p in candidates if p['later_loss_drop_at_least_0p1_variance']),None),
                  legacy_candidate_steps_before_refit_qualification=[p['step'] for p in subset if p['legacy_material_candidate_before_refit_qualification']],qualified_candidate_steps=[p['step'] for p in qualified],candidate_steps=[p['step'] for p in candidates],missing_diagnostic_steps=missing,unresolved_diagnostic_steps=unresolved,
                  required_saved_points=len(subset),diagnosed_saved_points=sum(p['diagnostic_status']=='ok' for p in subset),
                  boundary_diagnosed=any(p['step']==pr['end_step'] and p['diagnostic_status']=='ok' for p in subset),
                  no_event_is_not_proof_of_absence=True)
    a['warning_counts']=dict(missing_diagnostics=sum(p['diagnostic_status']!='ok' for p in a['checkpoints']),agop_unresolved=sum(p['agop_screen'] is False for p in a['checkpoints']),agop_unassessed=sum(p['agop_screen'] is None for p in a['checkpoints']),refit_unresolved=sum(p['refit_screen'] is False for p in a['checkpoints']),refit_unassessed=sum(p['refit_screen'] is None for p in a['checkpoints']),checkpoint_inconsistency=sum(bool(p['issues']) for p in a['checkpoints']),quadrature_double_checked=sum(p['quadrature_double_checked'] for p in a['checkpoints']))
    return a

def build_cells(arms):
    groups={}
    for a in arms:
        if a['supplemental']:continue
        groups.setdefault((a['student'],a['teacher'],a['cell']),[]).append(a)
    return [dict(cell=k[2],student=v[0]['student'],teacher=v[0]['teacher'],comparison_class=v[0].get('comparison_class'),certificate=v[0].get('certificate'),planned_seeds=[a['seed'] for a in v],run_ids=[a['id'] for a in v],windows={str(r):dict(planned=len(v),material_candidates=sum(bool(a['prefixes'][str(r)].get('first_material_candidate')) for a in v),numerically_qualified_candidates=sum(bool(a['prefixes'][str(r)].get('first_numerically_qualified_candidate')) for a in v),complete_sequences=sum(bool(a['prefixes'][str(r)].get('first_complete_sequence')) for a in v),assessments=dict(collections.Counter(a['prefixes'][str(r)]['assessment'] for a in v))) for r in RATIOS}) for k,v in groups.items()]

def summarize(manifest_path,execution_path):
    manifest_path=Path(manifest_path).resolve();execution_path=Path(execution_path).resolve();reader=Reader();m=reader.read(manifest_path)
    if not m or not isinstance(m.get('configs'),list):raise ValueError('Manifest configs list required')
    ids=[j['id'] for j in m['configs']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate manifest IDs')
    status=reader.read(execution_path/'STATUS.json',default={});complete=reader.read(execution_path/'COMPLETE.json',default={})
    moments=reader.read(manifest_path.parent/'TEACHERS.json',default={})
    data=execution_path/'data' if (execution_path/'data').is_dir() else execution_path
    # Current server layout remains execution/data even before execution exists.
    if not execution_path.exists() and manifest_path.parent.name=='bundle':data=execution_path/'data'
    arms=[]
    for job in m['configs']:
        e=len(reader.errors)
        try:
            a=load_arm(job,data/job['id'],reader,moments)
            for name,h in a.get('source_hashes_recorded',{}).items():
                key=('code/'+name if not name.startswith('diagnostics/') else name) if job['engine']=='relu' else 'swiglu/code/'+name
                expected=m.get('source_sha256',{}).get(key)
                if expected is not None and expected!=h:a['issues'].append('recorded_source_hash_mismatch:'+key)
            if len(reader.errors)>e:a['issues'].append('input_read_error')
            evaluate_arm(a)
        except Exception as exc:
            outer,c=config_of(job);a=dict(id=job['id'],cell=job.get('cell',job['id']),student=job['engine'],teacher=c.get('teacher',c.get('link')),seed=c.get('seed'),supplemental=bool(job.get('supplemental')),has_result=False,history=[],checkpoints=[],prefixes={str(r):dict(assessment='input_read_error',first_material_candidate=None,first_complete_sequence=None,right_censored=True) for r in RATIOS},issues=[repr(exc)],target_variance=None)
        a['process_state'],a['process_receipt']=process_state(job['id'],status,complete,a.get('has_result'))
        arms.append(a)
    cells=build_cells(arms)
    out=dict(schema=SCHEMA,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest=str(manifest_path),execution_directory=str(execution_path),runtime_allocation=m.get('runtime'),planned_arms=len(arms),canonical_arms=sum(not a['supplemental'] for a in arms),supplemental_arms=sum(a['supplemental'] for a in arms),cells=cells,arms=arms,
             status_snapshot=status,source_note='Saved records only; no model imports/evaluations/training. Snapshot is not a process liveness probe.',
             criteria={'loss_prefix_max_min':list(RATIOS),'minimum_direction_gain':.5,'lower_or_sensitivity_refit_gain_over_variance':.1,'later_same_run_raw_loss_drop_over_variance':.1,'original_relative_agop_guard':GUARD,'refit_solver_or_sensitivity_tolerance':REFIT_TOL,'refit_comparators_differ':True,'numerical_candidates_not_certificates':True,'same_checkpoint_required':True},
             counts={kind:{'planned':sum((not a['supplemental']) if kind=='canonical' else a['supplemental'] for a in arms),'process_states':dict(collections.Counter(a['process_state'] for a in arms if ((not a['supplemental']) if kind=='canonical' else a['supplemental'])))} for kind in ('canonical','supplemental')},
             input_sha256=reader.hashes,input_files_missing=sorted(set(reader.missing)),reader_errors=reader.errors,inputs_unchanged=reader.unchanged(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out['changed_input_paths']=getattr(reader,'changed_paths',[])
    if not out['inputs_unchanged']:
        for a in arms:
            a['issues'].append('inputs_changed_during_read')
            for p in a['checkpoints']:
                for key in ('same_checkpoint_material_candidate','same_checkpoint_material_candidate_original','same_checkpoint_material_candidate_numerically_qualified','legacy_material_candidate_before_refit_qualification'):p[key]=False
            for pr in a['prefixes'].values():pr.update(assessment='input_snapshot_changed',original_criterion_assessment='input_snapshot_changed',numerical_qualification_assessment='input_snapshot_changed',first_material_candidate=None,first_complete_sequence=None,first_numerically_qualified_candidate=None,first_qualified_complete_sequence=None,candidate_count=0,qualified_candidate_count=0,candidate_steps=[],qualified_candidate_steps=[],legacy_candidate_steps_before_refit_qualification=[])
        out['cells']=build_cells(arms);out['snapshot_usable']=False
    else:out['snapshot_usable']=True
    return clean(out)

def markdown(out):
    def value(a,key):
        p=a['prefixes'][key];c=p.get('first_material_candidate');seq=p.get('first_complete_sequence')
        if c:return 'candidate@'+str(c['step'])+('; later drop observed' if seq else '; later drop not yet observed')+(' [window open]' if p.get('right_censored') else '')
        return p['assessment']
    lines=['# Saved-data coverage and same-checkpoint criteria','',f"All {out['planned_arms']} declared arms retained: {out['canonical_arms']} canonical + {out['supplemental_arms']} separate extras. Numerical candidates, not population certificates. No training or model evaluation.",'',
           'ReLU refit is bounded by the original readout budget; SwiGLU uses an unrestricted pseudoinverse cutoff-sensitivity envelope. The SwiGLU initial-min/current-max cutoff envelope is not a bound on improvement of the unrestricted optimal risk. Raw SwiGLU loss and refit multiply saved variance exactly once. These comparators and student recipes differ. The minimum over all teacher directions is required; mean/top1 alignment cannot substitute.','',
           '| Canonical student / teacher | Planned seed | Process | 1% initial window | 5% initial window | Training / diagnostics |','|---|---:|---|---|---|---|']
    detail_header=lines[-2:];lines=lines[:-2]
    lines+=['| Canonical cell | Refit comparison | Planned seeds | Seed-by-seed 1% /5% outcome |','|---|---|---|---|']
    for cell in out['cells']:
        members=[a for a in out['arms'] if a['id'] in cell['run_ids']]
        text='; '.join(str(a['seed'])+': '+value(a,'1.01')+' / '+value(a,'1.05') for a in members)
        lines.append('| '+cell['student']+' / '+cell['teacher']+' | '+str(cell['comparison_class'])+' | '+','.join(map(str,cell['planned_seeds']))+' | '+text+' |')
    lines+=['','## Per-run process and censoring','']+detail_header
    for a in out['arms']:
        if a['supplemental']:continue
        lines.append(f"| {a['student']} / {a['teacher']} | {a['seed']} | {a['process_state']} | {value(a,'1.01')} | {value(a,'1.05')} | {a.get('stop_reason','—')} / {'complete' if a.get('diagnostics_complete') else 'missing/censored'} |")
    lines+=['','## Separate supplemental pure-sine arms','','| Cell | Seed | Process | 1% | 5% |','|---|---:|---|---|---|']
    for a in out['arms']:
        if a['supplemental']:lines.append(f"| {a['cell']} | {a['seed']} | {a['process_state']} | {value(a,'1.01')} | {value(a,'1.05')} |")
    lines+=['','## Interpretation','','A candidate must use one diagnosed checkpoint inside the exact all-update initial max/min window, with delta A_min>=0.5, passing initial/current original AGOP numerical screens, and initial-low minus current-high refit gain>=0.1Var(Y). A subsequent raw loss drop>=0.1Var(Y) is measured from that candidate and reported separately. Every candidate can occur before a still-open window ends; this does not establish where the window ends.','',
            'An unfinished prefix is right-censored. Missing diagnostics, missing guards, failed optimizer/sensitivity checks, inconsistent records or invalid loss do not become negative evidence. A complete measured grid with no candidate supports only a saved-grid nonobservation. Original frozen criterion and additional numerical qualification (refit solver/total-gap or cutoff-spread/residual checks) have separate named outcomes; extra checks do not overwrite historical flags. Numerical integration certification is not implied by the spectral/cutoff guards; saved doubled-order checks are recorded separately.','',
            f"Input snapshot unchanged: {out['inputs_unchanged']}; files hashed: {len(out['input_sha256'])}; reader errors: {len(out['reader_errors'])}. All missing files and per-state warnings remain in the JSON. No curves or criteria use interpolation."]
    return '\n'.join(lines)+'\n'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--execution',required=True,type=Path);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'SUMMARY.json');a=p.parse_args(argv)
    output=a.output.resolve();manifest=a.manifest.resolve();execution=a.execution.resolve()
    if output.is_relative_to(execution) or output==manifest or output.is_relative_to(manifest.parent):raise ValueError('Output must be outside immutable bundle and execution input directories')
    out=summarize(manifest,execution);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');output.with_suffix('.md').write_text(markdown(out))
    print(json.dumps({'output':str(output),'arms':out['planned_arms'],'canonical':out['canonical_arms'],'extras':out['supplemental_arms'],'snapshot_usable':out['snapshot_usable'],'reader_errors':len(out['reader_errors'])}))
if __name__=='__main__':main()
