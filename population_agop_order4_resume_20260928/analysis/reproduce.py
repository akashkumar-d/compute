"""Saved-file audit of exact order4 continuation; no model imports or evaluation."""
from pathlib import Path
import hashlib, importlib.util, json, math, datetime
import numpy as np

HERE=Path(__file__).resolve().parent
W=HERE.parents[1]
V9=W/'goal_followup_v9'
B=V9/'continuation_order4'
OLD=V9/'recovered_after_stop/population_agop_swiglu_ordercheck_20260928/execution_server_order24_a01'
NEW=V9/'continuation_order4_results/population_agop_order4_resume_20260928/execution_resume_a01'
T4='v9_swiglu_h2_headslow_order4_r8_d64_m64_seed642'
T2=T4.replace('order4','order2')
CANON=W/'goal_followup_v7/breadth14/analysis/summarize_coverage.py'
EXPECTED_CANON='fe0aedb926eafff5adcc966c3dc5042e2263da83e7e33ff6b5210f36a9e6ae0c'
tracked={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p=Path(p).resolve(); h=sha(p)
    if str(p) in tracked:assert tracked[str(p)]==h,('changed during reread',str(p))
    tracked[str(p)]=h;return p
def read(p):return json.loads(pin(p).read_text())
def arrays(p):
    with np.load(pin(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def maximum(records,key):
    r=max(records,key=lambda x:x[key]);return {'step':r['step'],'value':r[key]}
def scheduler(raw,include_final_stop):
    cfg=raw['cfg'];next_cp=0.;last_dL=1e-8;last_L=1.;steps=[]
    for h in raw['step_history']:
        step,t,L=h['step'],h['t'],h['L'];dL=1-L
        stop=include_final_stop and (t>=cfg['t_max'] or L<=cfg['L_stop'] or step>=cfg['max_steps'])
        if t>=next_cp or stop or (dL>1e-8 and dL>cfg['dl_ratio']*last_dL and L>cfg['L_stop']) or abs(L-last_L)>.02:
            steps.append(step);last_dL=max(dL,1e-8);last_L=L;next_cp=max(t*cfg['cp_ratio'],t+cfg['cp_min'])
    return dict(natural_grid_steps=steps,next_cp=next_cp,last_dL=last_dL,last_L=last_L)

def main():
    assert sha(pin(CANON))==EXPECTED_CANON
    spec=importlib.util.spec_from_file_location('saved_summary',CANON);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    sm4=s.summarize(B/'MANIFEST.json',NEW)
    sm2=s.summarize(V9/'trajectory_validation/MANIFEST.json',OLD)
    for name,sm in [('ORDER4_SUMMARY',sm4),('ORIGINAL_EXECUTION_SUMMARY',sm2)]:
        assert sm['snapshot_usable'] and not sm['reader_errors']
        tracked.update(sm['input_sha256']);save(name+'.json',sm);(HERE/(name+'.md')).write_text(s.markdown(sm))
    a4=sm4['arms'][0];a2=next(a for a in sm2['arms'] if a['id']==T2)
    assert not a4['issues'] and not a2['issues'] and a4['snapshot_integrity_verified'] and a2['snapshot_integrity_verified']
    d4=NEW/'data'/T4;d2=OLD/'data'/T2;di=OLD/'data'/T4
    old=read(di/(T4+'.json'));new=read(d4/(T4+'.json'));other=read(d2/(T2+'.json'))
    r4=read(d4/'result.json');r2=read(d2/'result.json');prior_d=read(di/'diagnostics_partial.json')
    zold=arrays(di/(T4+'_snaps.npz'));z4=arrays(d4/(T4+'_snaps.npz'));z2=arrays(d2/(T2+'_snaps.npz'))
    seg=read(d4/'continuation_segment.json');zs=arrays(d4/'continuation_segment_snaps.npz')
    resumed=read(d4/'RESUME_INPUT.json');check=read(B/'INPUT_CHECK.json')
    anchor=old['termination']['step'];nold=len(old['rows']);nfinal=len(new['rows'])
    preservation={k:new[k][:len(old[k])]==old[k] for k in ('rows','Lhist','step_history')}
    preservation['ratio_prefixes_exact']=old['ratio_prefixes']==new['ratio_prefixes']
    preservation['arrays_bitwise_equal']={k:bool(np.array_equal(zold[k],z4[k][:len(zold[k])])) for k in zold}
    preservation['prior_successful_diagnostics_exact']=all(all(r4['diagnostics'][d['index']].get(k)==v for k,v in d.items()) for d in prior_d['rows'] if d['status']=='ok')
    preservation['pinned_input_bytes_match_recovered']={p.name:sha(pin(p))==sha(pin(di/p.name)) for p in (B/'input').iterdir() if p.is_file()}
    assert all(preservation[k] for k in ('rows','Lhist','step_history','ratio_prefixes_exact','prior_successful_diagnostics_exact'))
    assert all(preservation['arrays_bitwise_equal'].values()) and all(preservation['pinned_input_bytes_match_recovered'].values())
    assert new['step_history'][anchor]==old['step_history'][-1] and len(new['step_history'])==new['termination']['step']+1
    assert all(h['step']==i and new['Lhist'][i]==[h['t'],h['L']] for i,h in enumerate(new['step_history']))
    assert all(new['step_history'][i]['t']==new['step_history'][i-1]['t']+new['step_history'][i-1]['dt'] for i in range(1,len(new['step_history'])))
    assert seg['step_history']==new['step_history'][anchor:] and seg['Lhist']==new['Lhist'][anchor:] and seg['rows']==new['rows'][nold-1:]
    assert all(np.array_equal(zs[k],z4[k][nold-1:]) for k in zs)
    assert all(np.isfinite(v).all() for v in list(z4.values())+list(z2.values()))
    assert all(np.array_equal(z2[k][0],z4[k][0]) for k in ('P','V','a'))
    cfgdiff={k:[other['cfg'].get(k),new['cfg'].get(k)] for k in set(other['cfg'])|set(new['cfg']) if other['cfg'].get(k)!=new['cfg'].get(k)}
    assert set(cfgdiff)=={'n_pair','n_diag','n_z','n_x','out'}
    assert all(r4['cfg'][k]==old['cfg'][k] for k in r4['cfg'])
    for order,raw,result,z in [(2,other,r2,z2),(4,new,r4,z4)]:
        assert len(raw['rows'])==result['checkpoint_count']==len(result['diagnostics'])
        assert all(d['status']=='ok' and d['index']==i and d['step']==raw['rows'][i]['step'] and d['t']==raw['rows'][i]['t'] for i,d in enumerate(result['diagnostics']))
        assert z['t'].tolist()==[r['t'] for r in raw['rows']]
        assert all(raw['rows'][i]['L']==raw['step_history'][row['step']]['L'] for i,row in enumerate(raw['rows']))
    ctl=scheduler(old,False)
    assert all(ctl[k]==resumed['scheduler_control'][k]==check['control'][k] for k in ctl)
    newgrid=scheduler(new,True)['natural_grid_steps']
    boundary={v[k] for v in new['ratio_prefixes'].values() for k in ('first_exit_step','last_valid_step')}
    expected=set(newgrid)|boundary|{anchor};actual={r['step'] for r in new['rows']}
    assert actual==expected and anchor not in newgrid
    assert set(actual)-{r['step'] for r in other['rows']}=={437}
    assert r4['continuation']['original_termination']==old['termination'] and r4['continuation']['original_result_completed'] is False
    assert r4['continuation']['independent_seed'] is False
    assert r4['continuation']['resumed_array_sha256']=={k:hashlib.sha256(zold[k][-1].tobytes()).hexdigest() for k in ('P','V','a')}
    assert resumed['adapter_files_sha256']['resume_engine.py']==sha(pin(B/'adapter/resume_engine.py'))
    engine=pin(B/'adapter/resume_engine.py').read_text()
    strict="if L != anchor['L'] or dt != anchor['dt'] or rel != anchor['rel']:" in engine
    assert strict and new['termination']['step']>anchor and r4['completed']
    manifests={};hashcounts={}
    for label,bundle,execution,result in [('order4',B,NEW,r4),('order2',V9/'trajectory_validation',OLD,r2)]:
        m=read(bundle/'MANIFEST.json');pr=read(execution/'PROVENANCE.json');launch=read((d4 if label=='order4' else d2)/'launch.json')
        assert sha(pin(bundle/'MANIFEST.json'))==sha(pin(execution/'MANIFEST.snapshot.json'))==pr['manifest_sha256']
        source=m['source_sha256'];assert all(sha(pin(bundle/k))==v for k,v in source.items())
        assert all(sha(pin(bundle/k))==v for k,v in pr['source_sha256'].items())
        assert all(sha(pin(bundle/'swiglu/code'/k))==v for k,v in result['source_files_sha256'].items())
        assert result['source_files_sha256']==launch['source_files_sha256']
        job=next(j for j in m['configs'] if j['id']==result['tag'])
        assert result['config_sha256']==launch['config_sha256']==job['configuration_sha256']==sha(pin(bundle/job['config_path']))==pr['config_sha256'][job['config_path']]
        for k,v in result['raw_files'].items():assert sha(pin((d4 if label=='order4' else d2)/k))==v
        manifests[label]=sha(bundle/'MANIFEST.json');hashcounts[label]=len(source)
    scientific_same=[]
    for k in read(B/'MANIFEST.json')['source_sha256']:
        if k.startswith('swiglu/'):
            assert sha(pin(B/k))==sha(pin(V9/'trajectory_validation'/k));scientific_same.append(k)
    for p in NEW.rglob('*'):
        if p.is_file():pin(p)
    transfer=read(W/'goal_followup_v10/COMPLETED_TRANSFER_QA_20260928_2308.json')
    assert transfer['continuation']['passed'] and transfer['continuation']['files']==len([p for p in NEW.rglob('*') if p.is_file()])
    dense=[]
    for h2,h4 in zip(other['step_history'],new['step_history']):
        assert h2['step']==h4['step']
        dense.append(dict(step=h4['step'],loss_absolute_difference=abs(h2['L']-h4['L'])*new['V'],time_absolute_difference=abs(h2['t']-h4['t']),dt_relative_difference=abs(h2['dt']-h4['dt'])/h4['dt'],relative_rate_relative_difference=abs(h2['rel']-h4['rel'])/h4['rel']))
    idx2={r['step']:i for i,r in enumerate(other['rows'])};idx4={r['step']:i for i,r in enumerate(new['rows'])};common=sorted(set(idx2)&set(idx4))
    common_records=[]
    cp2={p['step']:p for p in a2['checkpoints']};cp4={p['step']:p for p in a4['checkpoints']}
    for step in common:
        p2=np.concatenate([z2[k][idx2[step]].ravel() for k in ('P','V','a')]);p4=np.concatenate([z4[k][idx4[step]].ravel() for k in ('P','V','a')])
        record=dict(step=step,parameter_relative_difference=float(np.linalg.norm(p2-p4)/np.linalg.norm(p4)))
        for k in ('A_min','A_mean','A_top','refit_raw','cutoff_envelope_gain_raw','loss_raw','time'):
            record[k+'_order2']=cp2[step][k];record[k+'_order4']=cp4[step][k];record[k+'_absolute_difference']=abs(cp2[step][k]-cp4[step][k])
        common_records.append(record)
    outcomes={}
    for label,a,raw,result in [('order2',a2,other,r2),('order4',a4,new,r4)]:
        outcome=dict(termination=raw['termination'],target_mean=raw['EY'],target_variance=raw['V'],gamma=raw['gamma'],checkpoint_count=len(raw['rows']),dense_count=len(raw['Lhist']),snapshot_integrity_verified=a['snapshot_integrity_verified'],process_state=a['process_state'],prefixes={})
        for k,p in a['prefixes'].items():
            keys=['end_step','end_time','first_exit_step','first_exit_time','actual_ratio','right_censored','candidate_count','qualified_candidate_count','candidate_steps','qualified_candidate_steps','assessment','numerical_qualification_assessment','missing_diagnostic_steps']
            prefix={x:p.get(x) for x in keys};prefix['initial_loss_raw']=a['history'][0]['loss_raw'];prefix['endpoint_loss_raw']=a['history'][p['end_step']]['loss_raw'];prefix['first_material_candidate']=p['first_material_candidate'];prefix['first_qualified_complete_sequence']=p['first_qualified_complete_sequence']
            candidate=p['first_material_candidate']
            if candidate:
                hits=[h for h in a['history'] if h['step']>candidate['step'] and candidate['loss_raw']-h['loss_raw']>=.1*a['target_variance']]
                prefix['first_later_material_release']=hits[0] if hits else None
            valid=[c for c in a['checkpoints'] if c['step']<=p['end_step'] and c['agop_screen'] is True]
            prefix['maximum_A_min_gain_on_saved_prefix']=max(c['A_min_gain'] for c in valid)
            prefix['all_saved_prefix_diagnostics_complete']=all(c['diagnostic_status']=='ok' for c in a['checkpoints'] if c['step']<=p['end_step'])
            outcome['prefixes'][k]=prefix
        outcome['all_checkpoint_agop_screens_pass']=all(p['agop_screen'] is True for p in a['checkpoints'])
        outcome['all_checkpoint_refit_screens_pass']=all(p['refit_screen'] is True for p in a['checkpoints'])
        outcomes[label]=outcome
    result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Saved records/array arithmetic only; no model evaluations, training, network or old-data edits.',independent_seed_count=1,quadrature_runs=2,continuation_is_extra_independent_run=False,canonical_summarizer_sha256=EXPECTED_CANON,
        outcomes=outcomes,preservation=preservation,segment={'anchor_step':anchor,'dense_old_states':len(old['Lhist']),'dense_new_states':len(new['Lhist'])-len(old['Lhist']),'segment_dense_count_including_shared_anchor':len(seg['Lhist']),'old_snapshot_count':nold,'merged_snapshot_count':nfinal,'segment_snapshot_count_including_shared_anchor':len(seg['rows']),'merged_anchor_count':sum(h['step']==anchor for h in new['step_history']),'segment_arrays_match_merged':True},
        scheduler={'old_control_replayed':ctl,'full_natural_grid_steps':newgrid,'prefix_boundary_steps':sorted(boundary),'merged_extra_forced_step':anchor,'expected_grid_matches_saved':True,'continuation_diagnostic_policy':r4['continuation_diagnostic_policy']},
        strict_anchor={'code_requires_exact_loss_rel_dt':strict,'recorded_anchor':old['step_history'][-1],'basis':'Pinned runtime code aborts on any mismatch; exit0 and post-anchor progression establish its gate passed. Independent re-evaluation and the original gradient vector are not available in this saved-file audit.','teacher_moments_match':all(old[k]==new[k] for k in ('gamma','EY','V'))},
        source={'manifest_sha256':manifests,'manifest_file_counts':hashcounts,'scientific_files_byte_identical':scientific_same,'config_differences_order2_order4':cfgdiff,'all_launch_and_completion_source_hashes_match':True,'completed_transfer_receipt':str(W/'goal_followup_v10/COMPLETED_TRANSFER_QA_20260928_2308.json')},
        comparison={'common_dense_states':len(dense),'dense_maxima':{k:maximum(dense,k) for k in dense[0] if k!='step'},'common_saved_states':len(common),'common_saved_steps':common,'snapshot_maxima':{k:maximum(common_records,k) for k in common_records[0] if k=='parameter_relative_difference' or k.endswith('_absolute_difference')},'fixed_state_comparisons':[r for r in common_records if r['step'] in (0,257,258,358,380,381,382,619)],'same_step_not_exact_same_time':True,'parameter_norm_denominator':'order4 concatenated P,V,a Euclidean norm'},
        interpretation={'supports':'Same5% material plateau/refit/later-release outcome under two integrated quadrature orders on one reused development seed.','not_supported':['fresh-seed confirmation','1% phenomenon between unsaved checkpoints','formal population integration/AGOP/optimal-head certificate','step-size convergence','exact common-physical-time comparison'],'refit_comparison':'Conservative envelope over four numerical unrestricted pseudoinverse cutoffs, not a certified bound for unrestricted optimum and not same-budget bounded refit.'})
    result['all_inputs_unchanged']=all(sha(p)==h for p,h in tracked.items());assert result['all_inputs_unchanged']
    result['input_sha256']=tracked;save('VALIDATION.json',result);save('COMMON_SNAPSHOT_COMPARISON.json',common_records)
    write_report(result)
    print(json.dumps({'status':result['status'],'inputs':len(tracked),'dense_maxima':result['comparison']['dense_maxima'],'snapshot_maxima':result['comparison']['snapshot_maxima']}))
def write_report(d):
    o2,o4=d['outcomes']['order2'],d['outcomes']['order4']
    c2=o2['prefixes']['1.05']['first_material_candidate'];c4=o4['prefixes']['1.05']['first_material_candidate']
    lines=["# Completed quadrature-trajectory validation", "",
        "The completed order4 continuation preserves the fixed **5% plateau → all-direction learning and refit improvement → later loss decrease** outcome seen at order2. Both orders first meet the unchanged same-checkpoint criterion at step358. Neither has a qualifying saved checkpoint in the initial1% window. This is numerical agreement on **one reused development seed642**, not two independent successes or fresh-seed confirmation.", "",
        "Both runs use the normalized additive h2 teacher (saved mean0, variance1), rank8, dimension64, width64, SwiGLU with trainable gate/value/head, profiled intercept and biases, initial scale0.1, head ratio1, head learning multiplier0.001, adaptive Euler h0.01, and original global limits step3000/time100/normalized loss0.8. Raw and variance-relative losses coincide because Var(Y)=1. Order2 uses pair/diagonal/z/x quadrature32/96/48/24; order4 uses64/192/96/48. Apart from the output path and quadrature orders, configurations agree, and initial parameter arrays are bitwise equal.", "",
        "## Unchanged all-update windows and criteria", "",
        "The canonical reader hashes to `"+EXPECTED_CANON+"`. It uses the exact maximum/minimum ratio over every evaluated update. Last-valid endpoints are included and the first crossing is excluded; windows are neither moved nor feature-selected. A material saved checkpoint requires minimum-direction alignment gain≥0.5 and initial-min/current-max four-cutoff refit-envelope gain≥0.1Var at that same state, with original1e-9 spectral rank/gap guards. The later raw loss drop≥0.1Var is measured strictly after that same candidate. The additional numerical qualification remains separate from the original flag.", "",
        "| Order | Window | Last valid step / time | First exit | Raw endpoint loss | Exact max/min | Candidates original / qualified |",
        "|---|---|---|---|---:|---:|---:|"]
    for order,o in [(2,o2),(4,o4)]:
        for key,label in [('1.01','1%'),('1.05','5%')]:
            p=o['prefixes'][key];lines.append(f"| {order} | {label} | {p['end_step']} / {p['end_time']:.10f} | {p['first_exit_step']} | {p['endpoint_loss_raw']:.12f} | {p['actual_ratio']:.12f} | {p['candidate_count']} / {p['qualified_candidate_count']} |")
    lines += ["", "Both5% candidate lists are exactly **358,380,381**. All candidates are observations from the same trajectory; they do not increase the independent-run count. The initial raw loss is0.9999996472183378. Maximum minimum-direction gain within the fully diagnosed1% saved grids is only0.0104783753/order2 and0.0104784203/order4. This establishes non-observation on those grids, not absence at unsaved states.", "",
        "| First5% candidate metric | Order2 | Order4 continuation |", "|---|---:|---:|"]
    for title,key in [('Step','step'),('Physical time','time'),('Raw training loss','loss_raw'),('Minimum AGOP alignment','A_min'),('Minimum-direction gain','A_min_gain'),('Mean AGOP alignment','A_mean'),('Raw refit risk (cutoff1e-12)','refit_raw'),('Cutoff-envelope gain','cutoff_envelope_gain_raw'),('Later raw loss decrease through final state','later_loss_drop_raw')]:
        lines.append(f"| {title} | {c2[key]:.12g} | {c4[key]:.12g} |")
    lines += ["", "The first subsequent≥0.1 loss decrease is at step586 for both orders: time51.0424818531/loss0.855197469946 for order2 and time51.0445901272/loss0.855183712382 for order4. Both stop at step619 after reaching their original loss target: order2 time53.8267580175/loss0.798819161101; order4 time53.8283908754/loss0.798585319542. Neither complete trajectory is wall-censored, and the initial windows are closed.", "",
        "All53 order2 and54 order4 diagnostics pass the recorded AGOP and refit screens. At the order4 first candidate, relative rank eigenvalue4.410023397e-7 and relative rank gap1.235979208e-7 exceed the unchanged1e-9 guards; the four-cutoff spread is0, maximum normal-equation residual3.020846480e-16, and equilibrated eigenvalue ratio0.08531147647. There are no negative-risk gains or missing diagnostic placeholders in the completed continuation. These are numerical screens, not certified eigenspace errors.", "",
        "The SwiGLU refit number is a conservative envelope over the declared numerical unrestricted cutoff solves. It is **not** a certified lower bound on improvement of unrestricted optimal-head risk and is not a fixed-norm-budget comparator. Candidate diagnostics were not additionally order-doubled within these result files; the two complete trajectories provide the explicitly separate order comparison here.", "",
        "## Exact continuation preservation", "",
        "The original order4 execution stopped externally at437/time38.72140153556326/loss0.9462111101722706. It remains an interrupted execution with no retrospectively invented success receipt. The new exit0 completion is explicitly a merged, segmented continuation.", "",
        "All438 original dense loss/update records, all45 prior checkpoint rows, and every old t/P/V/a snapshot are preserved exactly; arrays are bitwise equal. All29 successful prior diagnostics are retained field-for-field with an origin label. The16 old budget-missing diagnostics are now computed, including exact order4 states358/380. Nine later checkpoints were added. The final merged trajectory has620 evaluated states and54 snapshots; its segment-only view has183 dense states and10 snapshots because it explicitly shares anchor437. The merged history contains that anchor once.", "",
        "Replaying the saved loss/time scheduler independently gives last natural checkpoint431 and next_cp40.53101837127362, last_dL0.05299185929986161, last_L0.9470081407001384. Forced interruption snapshot437 is retained as an extra and does not reset this scheduler. The final54-state grid equals the full natural grid plus exact prefix boundary states plus437; the other53 steps match order2. Every required original and later checkpoint is diagnosed. Both prefix exits remain byte-for-byte unchanged from the recovered segment.", "",
        "The frozen adapter requires exact equality of anchor loss, relative adaptive update rate, and dt before advancing. The pinned code, successful exit, and post-anchor progression establish that its strict gate passed. Anchor values are loss0.9462111101722706, rel0.12413394104468191, dt0.08055814482197507. This audit did not reevaluate that model, and no stored original gradient vector is available for an independent vector comparison. The original total step/time/loss caps and initial balance references remain unchanged.", "",
        "## Numerical agreement and remaining limits", "",
        "Comparison uses equal update numbers, not exactly equal physical times; no interpolation or surrogate states are used. Across620 common dense states, maximum raw-loss difference is0.0002338415582 at619, clock difference0.002291521362 at600, and relative proposed-dt difference0.100402064% at619 (order4 denominator). Across53 common saved states, the maximum relative concatenated P/V/a parameter difference is0.695596449% at619, using the order4 parameter norm; through the5% endpoint the maximum is0.0137616646%. The independent receipt uses the order2 norm denominator, hence its terminal0.695387638% differs slightly without an arithmetic disagreement. The maximum minimum-alignment difference is0.01748139341 at358; maximum mean-alignment difference0.002177793459 is also at358. Maximum refit-risk difference is0.0003140105976 at619. All same-state comparisons are retained in COMMON_SNAPSHOT_COMPARISON.json.", "",
        "The5% threshold verdict survives integration at the finer quadrature order, including the previously missing candidate diagnostics and later release. Alignment remains measurably sensitive, and later parameter separation is larger than in the old truncated comparison. These observations support finite-order trajectory agreement; they do not prove population integration accuracy, step-size convergence, formal AGOP accuracy, or a mathematical refit certificate.", "",
        "## Provenance, artifacts and reproduction", "",
        "The32 continuation manifest hashes and24 original validation manifest hashes match local files and server provenance; all11 files under swiglu/ are byte-identical across bundles. Launch/config/source and final raw-output hashes agree. The parent transfer receipt independently verified all17 continuation execution files; this audit tracked88 input artifacts and checked that none changed. Order4 manifest SHA256: `"+d['source']['manifest_sha256']['order4']+"`; original two-order manifest: `"+d['source']['manifest_sha256']['order2']+"`.", "",
        "- `ORDER4_SUMMARY.json/md`: unchanged canonical reader on the completed merged continuation.",
        "- `ORIGINAL_EXECUTION_SUMMARY.json/md`: canonical reader on the original execution, preserving completedorder2 and interruptedorder4 as originally recovered. Its latter missing-diagnostic and integrity qualifications are historical, not the completed continuation's verdict.",
        "- `VALIDATION.json`: preservation, source, scheduler, strict-anchor, candidate and comparison checks with input hashes.",
        "- `COMMON_SNAPSHOT_COMPARISON.json`: every common-state paired metric.",
        "- `INDEPENDENT_ARITHMETIC.json/md`: independently computed prefix/candidate/guard/grid checks; denominator choices are explicit.", "",
        "Run from this directory with the local NumPy Python runtime: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/legendkiller/anaconda3/bin/python3 reproduce.py`. The script reads saved JSON/NPZ and performs array arithmetic only. It imports the pinned saved-data reader, no scientific engine. It recreates these summaries, numerical audit and report; no training, model evaluations, network operations, original-data edits, or manuscript changes occur.", ""]
    (HERE/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
