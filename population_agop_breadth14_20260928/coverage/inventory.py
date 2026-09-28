#!/usr/bin/env python3
"""Saved-data inventory only. No scientific imports, training, or model evaluation."""
from pathlib import Path
import json, hashlib, datetime, collections, math
OUT = Path(__file__).resolve().parent
W = OUT.parents[2]
INPUTS = {}
def read(p):
    p = Path(p); b=p.read_bytes(); INPUTS[str(p.relative_to(W))]=hashlib.sha256(b).hexdigest()
    return json.loads(b)
def seen(p):
    p=Path(p);INPUTS[str(p.relative_to(W))]=hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(W))
def light(x):
    if not isinstance(x,dict): return x
    return {k:v for k,v in x.items() if not isinstance(v,(list,dict))}
def clip_point(x):
    if x is None:return None
    keep=['step','t','clock_force','loss_raw','A_min','A_mean','A_min_gain','refit_raw','refit_gain_raw','refit_gain_lower_or_sensitivity_raw','refit_gain_over_target_variance','refit_gain_envelope_raw','refit_gain_envelope_normalized','same_checkpoint_primary_candidate','same_checkpoint_gain_0.1_target_variance','same_checkpoint_material_candidate','subsequent_raw_loss_min','later_loss_drop','later_loss_drop_at_least_point1_variance','agop_numerical_screen','refit_optimizer_tolerance_met']
    return {k:x[k] for k in keep if k in x}
STAGES=[('v3','broader_teachers_dataprep_v3'),('v4','numerical_followup_v4'),('v4_h05','numerical_followup_v4/learning_rate_followup_v1'),('v5','numerical_followup_v5'),('v6_relu','goal_followup_v6/relu_confirmation'),('v6_swi_horizon','goal_followup_v6/swiglu_review/matched_horizon_he3'),('v6_swi_headrate','goal_followup_v6/swiglu_head_rate_v1'),('v6_h3_sine','goal_followup_v6/lightning_sine_confirmation')]
V3ROOTS=[W/'broader_teachers_dataprep_v3/remote_final/experiments/agop_broader_teachers_dataprep_20260928/execution',W/'broader_teachers_dataprep_v3/remote_concurrency16_final/experiments/agop_dataprep_concurrency16_20260928/continuation/execution']
V3={};V3EX={}
for p in V3ROOTS:
    for a in read(p/'SUMMARY.json')['records']:V3[a['id']]=a;V3EX[a['id']]=p
    read(p/'STATUS.json')
RUNS=[]
STATUSES={}
for label,stage in STAGES:
    root=W/stage; mf=root/'design/MANIFEST.json';mf=mf if mf.exists() else root/'MANIFEST.json'
    manifest=read(mf)
    stat=root/('remote_results/execution/STATUS.json' if label=='v6_h3_sine' else 'execution/STATUS.json')
    if stat.exists(): STATUSES[label]=read(stat)
    summaries={};sf=root/'analysis/SUMMARY.json'
    if sf.exists():summaries={a['id']:a for a in read(sf)['arms']}
    gf=root/'analysis/GOAL_METRICS.json'
    if label=='v6_h3_sine':gf=W/'goal_followup_v6/lightning_sine_analysis/analysis/GOAL_METRICS.json'
    goals={a['id']:a for a in read(gf)['runs']} if gf.exists() else {}
    for j in manifest['configs']:
        c=j['config']; c=c[0] if isinstance(c,list) else c; eng=j['engine'];args=c.get('args',c);name=args.get('teacher',args.get('link'));canonical={'he2':'h2','he3':'h3'}.get(name,name)
        cp=mf.parent/j['config_path'];read(cp)
        rr={'stage':label,'stage_directory':stage,'id':j['id'],'student':eng,'teacher_source_name':name,'teacher_canonical_name':canonical,'seed':args['seed'],'r':args.get('r',c.get('rank',len(args.get('c',[])))),'d':args['d'],'m':args['m'],'scale':args.get('scale',args.get('s')),'h':args['h'],'head_ratio':args.get('head_ratio'),'head_lr':args.get('head_lr',1.) if eng=='swiglu' else None,'profiled_intercept':bool(args.get('profile_intercept',False)) if eng=='relu' else True,'algorithm':j.get('algorithm','joint_GD_no_output_intercept' if eng=='relu' else 'joint_adaptive_population_gradient_flow_profiled_intercept'),'manifest':rel(mf),'config_file':rel(cp),'config':c,'material_criterion':'same saved state: initial positive-loss max/min <=1.01 or1.05, delta A_min>=0.5, safeguarded refit gain>=0.1 target variance; original numerical guards retained','scope_note':'A repeated seed across recipes is development reuse, not independent confirmation.'}
        ex=V3EX[j['id']]/'data'/j['id'] if label=='v3' else root/('remote_results/execution/data' if label=='v6_h3_sine' else 'execution')/j['id']
        rr['execution_directory']=rel(ex)
        done=ex/('DONE.json' if eng=='relu' else 'result.json'); raw=read(done) if done.exists() else {};rr['result_file']=rel(done);rr['result_file_exists']=done.exists()
        if eng=='relu':
            meta=read(ex/'meta.json');rr['source_sha256_recorded']=meta.get('source_sha256',{});rr['raw_configuration_matches_manifest']=meta.get('config')==c
            rr['completion']={k:raw.get(k) for k in ['complete','training_complete','training_censored','training_stop_reason','diagnostic_complete','diagnostic_censored','prefix_diagnostics_complete','observed_steps','planned_steps','plateau_right_censored','true_plateau_endpoint_diagnosed','elapsed_seconds','final_loss']}
            rr['completion']['process_exit_zero']=None
        else:
            rr['source_sha256_recorded']=raw.get('source_files_sha256',{});rr['raw_configuration_matches_manifest']=raw.get('cfg')==args
            rr['completion']={k:raw.get(k) for k in ['completed','status','training_wall_seconds','diagnostic_wall_seconds','wall_seconds','diagnostic_success_count','checkpoint_count','wall_stop_cause','raw_partial','termination']}
        a=summaries.get(j['id'])
        if label=='v3':
            a=V3[j['id']];s=a['scientific'];diag=s.get('diagnostics',{})
            rr['legacy_v3_record']={'execution_state_from_stale_summary':a['execution']['state'],'outcome_category_from_legacy_summary':a.get('outcome_category'),'plateau':s.get('plateau'),'joint_in_own_plateau':s.get('joint_in_own_plateau'),'diagnostic_counts':{k:v for k,v in diag.items() if k not in ['required_steps','recorded_steps','initial','primary_prefix_endpoint','primary_prefix_last_saved','terminal']},'recorded_endpoint_fields':s.get('recorded_endpoint_fields'),'normalization_recorded':s.get('normalization_recorded')}
            rr['legacy_v3_record']['joint_in_own_plateau'].pop('first_joint_checkpoint',None)
            rr['completion']['stale_launcher_summary_warning']=a['execution']['state'] in ['active','not_started']
            rr['normalization']=s.get('normalization_recorded')
            rr['windows']={'1.01':{'material_assessment':'not harmonized in this inventory; legacy metric retained'},'1.05':{'material_assessment':'not harmonized in this inventory; legacy metric retained'}}
            if eng=='relu':
                rec=a['execution'].get('completion_receipt') or {};rr['completion']['process_exit_zero']=rec.get('returncode')==0 if rec.get('returncode') is not None else None
        elif a:
            rr['summary_source']=rel(sf)
            rr['summary_scalars']=light(a)
            rr['normalization']={k:a.get(k) for k in ['target_mean','target_second_moment','target_variance','loss_baseline','normalization_note','normalization'] if k in a}
            rr['warning_counts']=a.get('warning_counts',{})
            rr['process_receipt']=a.get('launcher_outcome',a.get('process_receipt'))
            if eng=='relu' and rr['process_receipt']:rr['completion']['process_exit_zero']=rr['process_receipt'].get('exit_code')==0
            if 'prefixes' in a:
                rr['windows']={}
                for r,k in [('1.01','ratio_1pct'),('1.05','ratio_5pct')]:
                    ca=a.get('candidates',{}).get(k,{})
                    rr['windows'][r]={'prefix':a['prefixes'].get(k),'material_assessment':ca.get('assessment'),'first_material_candidate':clip_point(ca.get('first_material_candidate')),'first_positive_candidate':clip_point(ca.get('first_positive_candidate'))}
            else:
                mat=[p for p in a.get('checkpoints',[]) if p.get('same_checkpoint_gain_0.1_target_variance')]
                rr['windows']={'1.01':{'material_assessment':'not separately reported in prior summary'},'1.05':{'prefix':a.get('primary_ratio_prefix'),'primary_assessment':a.get('primary_candidate_assessment'),'first_material_candidate':clip_point(mat[0]) if mat else None,'first_material_step_recorded':a.get('first_gain_01_target_variance_step'),'maximum_screened_A_min_gain':a.get('maximum_prefix_raw_min_gain'),'maximum_refit_gain_raw':a.get('maximum_prefix_refit_gain_raw')}}
        if j['id'] in goals:
            g=goals[j['id']];rr['goal_metric_source']=rel(gf);rr['normalization']={**(rr.get('normalization') or {}),'target_variance':g.get('target_variance')}
            rr['windows']={r:{**light(v),'first_material_candidate':clip_point(v.get('first_candidate')),'first_complete_sequence':clip_point(v.get('first_complete_sequence'))} for r,v in g['windows'].items()}
            rr['goal_scalars']=light(g)
        if rr.get('normalization') is None and eng=='relu':
            mu = 1/math.sqrt(2*math.pi) if name=='relu' else math.sqrt(2/math.pi) if name=='abs' else 0.
            s2 = .5 if name=='relu' else 1.
            den=s2+(rr['r']-1)*mu*mu
            mean=math.sqrt(rr['r'])*mu/math.sqrt(den)
            rr['normalization']={'target_mean':mean,'target_second_moment':1.,'target_variance':1-mean*mean,'loss_baseline':1-mean*mean if rr['profiled_intercept'] else 1.,'provenance':'Exact documented source normalization; zero-mean unit-second-moment normalized links otherwise.'}
        if label=='v3':
            rec=V3[j['id']]['execution'].get('completion_receipt')
            if rec:rr['process_receipt']=rec
        if label in STATUSES:
            rs=STATUSES[label].get('completed', STATUSES[label].get('finished',[]))
            if isinstance(rs,list):
                rec=next((x for x in rs if x['id']==j['id']),None)
                if rec:rr['process_receipt']=rec
        if rr.get('process_receipt'):
            rec=rr['process_receipt']; code=rec.get('exit_code',rec.get('returncode'))
            rr['completion']['process_exit_zero']=code==0 if code is not None else None
        if eng=='swiglu':
            ds=raw.get('diagnostics',[])
            rr['completion']['diagnostic_complete']=bool(ds) and all(x.get('status')=='ok' for x in ds)
        RUNS.append(rr)
CANONICAL=[
 ('h2','(z^2-1)/sqrt(2)','sqrt(2)*z','even polynomial; first Hermite degree2'),
 ('h3','(z^3-3*z)/sqrt(6)','(3*z^2-3)/sqrt(6)','odd polynomial; first Hermite degree3'),
 ('h4','(z^4-6*z^2+3)/sqrt(24)','(4*z^3-12*z)/sqrt(24)','even polynomial; first Hermite degree4'),
 ('h5','(z^5-10*z^3+15*z)/sqrt(120)','(5*z^4-30*z^2+15)/sqrt(120)','odd polynomial; first Hermite degree5'),
 ('relu','max(z,0)','1(z>0); value at0 immaterial under Gaussian law','nonsmooth positive homogeneous'),
 ('leaky_relu','z if z>=0 else 0.1*z','1 if z>0 else0.1; value at0 immaterial','negative slope0.1; no trainable parameter'),
 ('abs','abs(z)','sign(z); value at0 immaterial','even nonsmooth'),
 ('softplus','log(1+exp(z)), evaluated as logaddexp(0,z)','sigmoid(z)','beta1, no centering/offset'),
 ('silu','z*sigmoid(z)','sigmoid(z)+z*sigmoid(z)*(1-sigmoid(z))','beta1'),
 ('gelu','z*Phi(z)','Phi(z)+z*phi(z)','exact Gaussian CDF; no tanh approximation'),
 ('sine','sin(z)','cos(z)','frequency1'),
 ('tanh','tanh(z)','1-tanh(z)^2','scale1'),
 ('erf','erf(z)','2/sqrt(pi)*exp(-z^2)','scale1, not erf(z/sqrt(2))'),
 ('gaussian_rbf','exp(-z^2/2)','-z*exp(-z^2/2)','bandwidth1, centered at0')]
DEFINITIONS=[]
for name,q,dq,note in CANONICAL:
    it={'name':name,'q':q,'dq':dq,'convention':note,'teacher':'Y=sum_{i=1}^r q(X_i)/sqrt(r*(s2+(r-1)*mu^2)), X~N(0,I_d), mu=E[q(Z)], s2=E[q(Z)^2]','coefficient_vector':[1.]*8,'teacher_directions':'first8 standard basis vectors in R64','raw_target_second_moment':1,'normalization_not_centered':True}
    for eng in ['relu','swiglu']:
        rs=[a for a in RUNS if a['student']==eng and a['teacher_canonical_name']==name]
        it[eng]={'executed_arms':len(rs),'distinct_seeds':sorted(set(a['seed'] for a in rs)),'widths':sorted(set(a['m'] for a in rs)),'stages':sorted(set(a['stage'] for a in rs)),'intercepts':sorted(set('profiled' if a['profiled_intercept'] else 'none' for a in rs)),'run_keys':[a['stage']+'/'+a['id'] for a in rs]}
        it[eng]['prior_cli_supported']=name in (['h2','h3','h4','relu','abs','sine'] if eng=='relu' else ['h2','h3','relu','abs','tanh'])
        it[eng]['implementation_gap']=not it[eng]['prior_cli_supported']
    DEFINITIONS.append(it)
SOURCE_FILES=['goal_followup_v6/relu_confirmation/code/teachers_extra.py','goal_followup_v6/relu_confirmation/code/teachers_product.py','goal_followup_v6/relu_confirmation/code/population.py','goal_followup_v6/relu_confirmation/code/scaling_run.py','goal_followup_v6/swiglu_head_rate_v1/swiglu/code/engine/swsmall_periodic.py','goal_followup_v6/swiglu_head_rate_v1/swiglu/code/engine/swpop.py','goal_followup_v6/swiglu_head_rate_v1/swiglu/code/run_one.py','goal_followup_v6/sine_stable_audit/REPORT.md','goal_followup_v6/sine_stable_audit/NEXT_EXPERIMENT.md']
for p in SOURCE_FILES:
    if (W/p).exists():seen(W/p)
counts={eng:{'arms':sum(a['student']==eng for a in RUNS),'distinct_underlying_links':sorted(set(a['teacher_canonical_name'] for a in RUNS if a['student']==eng)),'canonical14_executed_count':sum(d[eng]['executed_arms']>0 for d in DEFINITIONS),'canonical14_missing':[d['name'] for d in DEFINITIONS if not d[eng]['executed_arms']]} for eng in ['relu','swiglu']}
obj={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Read-only inventory of frozen configs, raw completion/metadata receipts and existing saved analyses. No new model evaluation, training, threshold selection or harmonization of legacy v3 candidate metrics.','scope':'Rank8 current v3/v4/v5/v6 cohorts only. No old rank1/2 evidence, no v7 unrun bundles.','all_declared_arms_retained':len(RUNS)==84,'total_arms':len(RUNS),'distinct_seed_numbers':sorted(set(a['seed'] for a in RUNS)),'counting_rule':'One cohort/config id once; v3 final continuation replaces missing original4 outputs. Copies, windows, centered/raw labels and head/width/rate variants do not create teacher links or independent seeds.','counts':counts,'stage_counts':dict(collections.Counter(a['stage'] for a in RUNS)),'canonical14':DEFINITIONS,'runs':RUNS,'normalization':{'raw_second_moment':1,'formula':'D=s2+(r-1)*mu^2; EY=sqrt(r)*mu/sqrt(D); Var(Y)=(s2-mu^2)/D','intercept':'Old ReLU raw baseline1 or explicit profiled controls baseline Var(Y). SwiGLU output intercept profiled. Proposed breadth14 profiles both students for every link; this changes ReLU algorithm even for meanzero targets, so historical raw successes are context, not exact recipe replications.','raw_units':'ReLU stored loss/refit are raw MSE; SwiGLU stored loss/refit are divided by Var(Y), multiply once.','swi_quadrature':'Current source uses piecewise Gauss-Legendre teacher moments on[-10,10], breaks[-6,-4,-2,0,2,4,6] plus link kinks, n_x12; finite-order approximations, new links need independent moment/derivative/doubled-order checks.'},'proposed_screen':{'frozen_by_root':True,'teachers':[x[0] for x in CANONICAL],'seeds':[641,642],'r':8,'d':64,'arms':56,'cpu_workers':4,'threads_per_worker':1,'total_cap_seconds_per_arm':210,'training_allowance_seconds':150,'diagnostic_allowance_seconds':50,'cleanup_allowance_seconds':10,'nominal_waves':14,'nominal_worst_arm_wave_seconds':2940,'global_compute_cap_seconds':3300,'wall_budget_note':'3300s leaves300s for setup in a1h wall ceiling; setup/QA must be bounded separately. Per-armcaps do not guarantee observed scientific horizons or complete diagnostics.','relu':{'m':256,'scale':.01,'h':.5,'profile_intercept':True},'swiglu':{'m':64,'s':.1,'head_ratio':1,'head_lr':.01,'h':.01,'profile_intercept':True},'interpretation':'Two-seed exploratory, horizon-limited screening; neither confirmation nor a matched student comparison (widths, parameterizations and rates differ).'},'source_definitions':SOURCE_FILES,'input_sha256':INPUTS}

obj['proposed_screen']['conditional_allocations']={
  'if_verified_4_CPU':{'workers':4,'arms':56,'waves':14,'per_arm_seconds':210,'training_seconds':150,'diagnostic_seconds':50,'cleanup_seconds':10,'nominal_waves_seconds':2940,'global_compute_seconds':3300,'setup_max_seconds':120},
  'if_verified_64_CPU_and_sufficient_memory':{'workers':60,'reserved_cores':4,'arms':60,'canonical_arms':56,'pure_sine_extra_arms':4,'waves':1,'threads_per_worker':1,'per_arm_seconds':900,'training_seconds':710,'diagnostic_seconds':180,'cleanup_seconds':10,'nominal_wave_seconds':900,'global_compute_seconds':1200,'setup_max_seconds':120,'conditions':['Remote lrun CPU count must report at least64 allocated cores','At least24GiB available memory and10GiB disk required by parent guard;60-worker peak memory remains unbenchmarked','Bound setup and shutdown; no dependency waits within compute budget'],'caveat':'Prior4-worker throughput does not validate60-worker memory bandwidth/CPU contention.180s diagnostics is less than250-261s observed for74-75 m256 states.'}}
obj['proposed_screen']['preferred_allocation']='64CPU only if actually allocated and memory/admission checks pass;4CPU fallback remains a separately stated shorter screen'
obj['proposed_screen']['training_allowance_seconds_4CPU']=obj['proposed_screen'].pop('training_allowance_seconds')
obj['historical_process_exit_counts']=dict(collections.Counter(str(a['completion'].get('process_exit_zero')) for a in RUNS))
obj['historical_diagnostic_complete_counts']=dict(collections.Counter(str(a['completion'].get('diagnostic_complete')) for a in RUNS))

assert len(RUNS)==84 and len({(a['stage'],a['id']) for a in RUNS})==84
assert all(a['r']==8 and a['d']==64 for a in RUNS)
obj['inputs_unchanged']=all(hashlib.sha256((W/p).read_bytes()).hexdigest()==v for p,v in INPUTS.items())
assert obj['inputs_unchanged']
(OUT/'coverage.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
(OUT/'LINK_DEFINITIONS.json').write_text(json.dumps({'canonical14':DEFINITIONS,'normalization':obj['normalization'],'parent_frozen_screen':obj['proposed_screen']},indent=2,allow_nan=False)+'\n')
print(json.dumps({'total':len(RUNS),'counts':counts,'inputs':len(INPUTS),'unchanged':obj['inputs_unchanged'],'configuration_mismatch_keys':[a['stage']+'/'+a['id'] for a in RUNS if not a['raw_configuration_matches_manifest']]},indent=2))
