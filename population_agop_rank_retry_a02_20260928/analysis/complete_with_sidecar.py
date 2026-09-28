"""Explicit saved-diagnostic overlay; unchanged canonical metrics, no fake result files."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import struct
import report

HERE=Path(__file__).resolve().parent
DIAG=report.V10/'rank_diagnostics_completion'
TAG='v10_rank_swiglu_h2_r2_d64_m64_seed642'
SIDECAR_EXECUTION=DIAG/'remote_results/population_agop_rank2_diagnostics_20260928/execution_remote_a01'
SIDECAR=SIDECAR_EXECUTION/'data'/TAG/'DIAGNOSTICS_COMPLETION.json'


def exact_values(left,right):
    """Dictionary order is irrelevant; numeric types and every float bit must match."""
    if type(left) is not type(right):return False
    if isinstance(left,float):return struct.pack('>d',left)==struct.pack('>d',right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(exact_values(v,right[k]) for k,v in left.items())
    if isinstance(left,list):return len(left)==len(right) and all(exact_values(a,b) for a,b in zip(left,right))
    return left==right


def verify_sidecar(original,raw,sidecar,spec,plan,snapshot_metadata):
    if not (sidecar.get('completion_pass_finished') is True and sidecar.get('completion_pass_complete') is True
            and sidecar.get('input_files_unchanged_after') is True and sidecar.get('new_training_updates')==0
            and sidecar.get('original_process_diagnostics_complete') is False
            and sidecar.get('original_result_mutated') is False and sidecar.get('independent_seed') is False
            and sidecar.get('quadrature_multiplier')==1):
        raise ValueError('Sidecar completion/provenance invariants failed')
    if sidecar['tag']!=spec['tag'] or sidecar['cfg']!=original['cfg'] or sidecar['cfg']!=spec['cfg']:
        raise ValueError('Sidecar identity/config mismatch')
    if sidecar['input_files_before']!=spec['files'] or sidecar['input_files_after']!=spec['files']:
        raise ValueError('Sidecar original input pins changed')
    if sidecar['plan']!=plan or sidecar['attempted_indices']!=plan['full_compute_order']:
        raise ValueError('Sidecar exact planned missing-grid order mismatch')
    if sidecar['original_training_termination']!=original['termination'] or raw['termination']!=original['termination']:
        raise ValueError('Original training termination changed')
    diagnostics=sidecar['diagnostics'];rows=raw['rows'];n=spec['expected_checkpoint_count'];cfg=spec['cfg']
    if len(diagnostics)!=n or len(rows)!=n or sidecar['diagnostic_success_count']!=n or sidecar['missing_indices']:
        raise ValueError('Full original grid required')
    expected_shapes={'t':[n],'P':[n,cfg['d']+1,cfg['m']],'V':[n,cfg['d']+1,cfg['m']],'a':[n,cfg['m']]}
    if snapshot_metadata['shapes']!=expected_shapes or not exact_values(snapshot_metadata['times'],[r['t'] for r in rows]):
        raise ValueError('Original snapshot shape/timestamp metadata mismatch')
    reused=[];added=[]
    for i,(old,new,row) in enumerate(zip(original['diagnostics'],diagnostics,rows)):
        if new['index']!=i or new['step']!=row['step'] or not exact_values(new['t'],row['t']) or new['status']!='ok':
            raise ValueError('Sidecar checkpoint identity/status mismatch')
        if old['status']=='ok':
            if not exact_values(old,new):raise ValueError('An original successful diagnostic value changed')
            if sidecar['diagnostic_row_origins'][str(i)]!='original_interrupted_diagnostic_process':
                raise ValueError('Reused-row provenance mismatch')
            reused.append(i)
        else:
            if sidecar['diagnostic_row_origins'][str(i)]!='diagnostics_only_completion_pass':
                raise ValueError('Completed-row provenance mismatch')
            added.append(i)
    if reused!=spec['expected_success_indices'] or set(added)!=set(plan['full_compute_order']):
        raise ValueError('Preserved/computed partition mismatch')
    if len(sidecar['diagnostic_row_origins'])!=n:raise ValueError('Unexpected diagnostic origin entries')
    return dict(original_successful_rows_exact_typed_float_bits=True,preserved_indices=reused,
        newly_computed_indices=added,complete_original_grid=n,all_snapshot_metadata_exact=True,
        new_training_updates=0,unchanged_config=True)


def overlay_reader_class(canonical,result_path,sidecar):
    class OverlayReader(canonical.Reader):
        def read(self,path,kind='json',default=None):
            original=super().read(path,kind,default)
            if Path(path).resolve()==result_path.resolve():
                if kind!='json' or not isinstance(original,dict):raise ValueError('Original result required')
                # This is an explicit transient data view, never a file or training receipt.
                view=copy.deepcopy(original)
                view['diagnostics']=copy.deepcopy(sidecar['diagnostics'])
                return view
            return original
    return OverlayReader


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-report',type=Path,default=HERE/'report_a02')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    base=args.base_report.resolve();output=args.output.resolve()
    if base.parent!=HERE or output.parent!=HERE or output.exists():raise ValueError('Use preserved base and fresh output within this analysis folder')
    plan=report.read(HERE/'PLAN.json');report.check_hashes(plan['pinned_inputs'])
    base_provenance=report.read(base/'REPORT_PROVENANCE.json')
    if base_provenance.get('pass_check') is not True:raise ValueError('Verified base report required')
    for name,digest in base_provenance['output_sha256'].items():
        if report.sha(base/name)!=digest:raise ValueError('Base report changed: '+name)
    audit=report.read(DIAG/'TRANSFER_QA.json');inventory=report.read(DIAG/'REMOTE_FINAL_INVENTORY.json')
    if audit.get('passed') is not True or audit.get('missing') or audit.get('mismatched'):
        raise ValueError('Verified sidecar transfer required')
    inputs={}
    for name,expected in inventory['files'].items():
        path=(SIDECAR_EXECUTION/name).resolve()
        if not path.is_relative_to(SIDECAR_EXECUTION) or not path.is_file():raise ValueError('Invalid sidecar inventory path')
        if path.stat().st_size!=expected['bytes'] or report.sha(path)!=expected['sha256']:
            raise ValueError('Sidecar inventory mismatch: '+name)
        inputs[str(path)]=expected['sha256']
    sidecar=report.read(SIDECAR);spec=report.read(DIAG/'INPUT_SPEC.json');completion_plan=report.read(DIAG/'COMPLETION_PLAN.json')
    manifest_hash=report.sha(DIAG/'MANIFEST.json');review=report.read(DIAG/'REVIEW.json')
    if sidecar['manifest_sha256']!=manifest_hash or review.get('status')!='approved_for_execution' or review.get('manifest_sha256')!=manifest_hash:
        raise ValueError('Sidecar reviewed source manifest mismatch')
    frozen=report.read(DIAG/'MANIFEST.json')
    if sidecar['source_sha256']!=frozen['source_sha256']:raise ValueError('Runtime sidecar sources differ from reviewed source')
    for name,digest in frozen['source_sha256'].items():
        if report.sha(DIAG/name)!=digest:raise ValueError('Sidecar prepared source changed: '+name)
    original_summary=report.read(report.ORIGINAL_SUMMARY)
    old=next(a for a in original_summary['arms'] if a['id']==TAG)
    original_folder=Path(old['execution_directory'])
    for name,expected in spec['files'].items():
        path=original_folder/name
        if path.stat().st_size!=expected['size'] or report.sha(path)!=expected['sha256']:
            raise ValueError('Original input changed: '+name)
        inputs[str(path)]=expected['sha256']
    original=report.read(original_folder/'result.json');raw=report.read(original_folder/spec['raw_name'])
    canonical=report.module(report.CANONICAL,'unchanged_sidecar_canonical_summary')
    metadata_reader=canonical.Reader();metadata=metadata_reader.snapshot_metadata(original_folder/spec['snapshots_name'])
    verification=verify_sidecar(original,raw,sidecar,spec,completion_plan,metadata)
    status=report.read(SIDECAR_EXECUTION/'STATUS.json')
    if (status.get('status')!='completed' or status.get('active') or status.get('pending')
            or len(status.get('completed',[]))!=1 or status['completed'][0]['id']!=TAG or status['completed'][0]['returncode']!=0):
        raise ValueError('Completed sidecar process receipt required')
    manifest=report.read(report.ORIGINAL_MANIFEST);job=next(c for c in manifest['configs'] if c['id']==TAG)
    reader=overlay_reader_class(canonical,original_folder/'result.json',sidecar)()
    moments=reader.read(report.ORIGINAL_MANIFEST.parent/'TEACHERS.json',default={})
    updated=canonical.load_arm(job,original_folder,reader,moments)
    for name,digest in updated.get('source_hashes_recorded',{}).items():
        if manifest.get('source_sha256',{}).get('swiglu/code/'+name) not in (None,digest):
            raise ValueError('Original recorded scientific source mismatch')
    canonical.evaluate_arm(updated)
    if reader.errors or not reader.unchanged() or not metadata_reader.unchanged():
        raise ValueError('Original/derived saved-data read failed integrity checks')
    for key in ('process_state','process_receipt'):updated[key]=copy.deepcopy(old[key])
    # Original completion/training/process metadata stays exactly as originally admitted.
    for key in ('history','config','completion_record','training_record','process_state','process_receipt'):
        if not exact_values(updated[key],old[key]):raise ValueError('Original non-diagnostic metadata changed: '+key)
    provenance=dict(kind='explicit_in_memory_diagnostic_overlay',sidecar_path=str(SIDECAR),sidecar_sha256=report.sha(SIDECAR),
        original_result_path=str(original_folder/'result.json'),original_result_sha256=report.sha(original_folder/'result.json'),
        original_training_process_receipt_unchanged=True,original_process_diagnostics_complete=False,
        completion_process_receipt=status['completed'][0],virtual_overlay_fields=['diagnostics'],
        fake_result_files_created=False,original_records_written=False,new_training_updates=0,
        verification=verification)
    updated['derived_diagnostic_provenance']=provenance
    combined=report.read(base/'COMBINED32_SUMMARY.json');before=copy.deepcopy(combined)
    index=next(i for i,a in enumerate(combined['arms']) if a['id']==TAG)
    combined['arms'][index]=updated;combined['cells']=canonical.build_cells(combined['arms'])
    combined['created_utc']=datetime.now(timezone.utc).isoformat()
    combined['source_note']='Explicit derived fixed-attempt combination plus one verified frozen-diagnostics sidecar; original training/process receipts retained.'
    combined['derivation']['rank2_completion_sidecar_included']=True
    combined['derivation']['sidecar_overlay']=provenance
    combined['derivation']['pre_sidecar_summary']=str(base/'COMBINED32_SUMMARY.json')
    combined['derivation']['pre_sidecar_summary_sha256']=report.sha(base/'COMBINED32_SUMMARY.json')
    combined['derivation']['arm_records_copied_verbatim']=False
    combined['derivation']['unchanged_other_arm_count']=31
    for i,(a,b) in enumerate(zip(before['arms'],combined['arms'])):
        if i!=index and not exact_values(a,b):raise ValueError('Unrelated selected arm changed')
    for p in [Path(__file__),base/'COMBINED32_SUMMARY.json',base/'REPORT_PROVENANCE.json',
            DIAG/'TRANSFER_QA.json',DIAG/'REMOTE_FINAL_INVENTORY.json',DIAG/'MANIFEST.json',DIAG/'REVIEW.json',
            DIAG/'INPUT_SPEC.json',DIAG/'COMPLETION_PLAN.json']:
        inputs[str(p)]=report.sha(p)
    inputs.update(reader.hashes);inputs.update(metadata_reader.hashes)
    for name,digest in inputs.items():
        if name in combined['input_sha256'] and combined['input_sha256'][name]!=digest:raise ValueError('Overlay input hash conflict')
        combined['input_sha256'][name]=digest
    report.check_hashes(combined['input_sha256'])
    output.mkdir()
    summary_path=output/'COMBINED32_WITH_SIDECAR.json';report.write_json(summary_path,canonical.clean(combined))
    report.write_json(output/'SIDECAR_VERIFICATION.json',verification)
    report.write_json(output/'SIDECAR_PROVENANCE.json',provenance)
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
    plotter=report.module(report.RANK/'analysis_rank/plot_rank.py','unchanged_sidecar_rank_plot')
    plotter.HERE=HERE
    plotter.write_plots(summary_path,report.ORIGINAL_MANIFEST,output/'plots',
        'Rank study: fixed attempts + rank2 diagnostic completion; reused seeds 641/642')
    before_fields={r:before['arms'][index]['prefixes'][r] for r in ('1.01','1.05')}
    after_fields={r:updated['prefixes'][r] for r in ('1.01','1.05')}
    report.write_json(output/'TARGET_PREFIX_COMPARISON.json',dict(before=before_fields,after=after_fields))
    lines=['# Combined rank report with explicit diagnostic completion','',
        'This new derived view preserves the frozen 20-original/12-retry attempt rule and all original training/process receipts. The separate reviewed rank-two quadratic SwiGLU seed642 sidecar supplies only missing frozen-state diagnostics through an explicit in-memory overlay. No result.json or synthetic training/process receipt is written. The original attempt summary and pre-sidecar report_a02 remain unchanged.','',
        'All269 original saved checkpoints have successful diagnostics: five original records preserved with exact types/float-bit values and264 newly computed records. Original snapshot shapes, timestamps, input hashes and configuration match exactly. No training updates or new independent seeds are introduced. The other31 combined arm records remain verbatim.','',
        'The unchanged canonical metric reader and criteria evaluator recompute only the target arm from the original trajectory and verified diagnostic list. Its original completion_record, training_record, process_state, process_receipt, configuration and full training history remain exact. The old active_snapshot process label is a historical original receipt; it does not describe the completed diagnostic sidecar or current liveness.','',
        '| Initial tolerance | Qualified complete sequences, all32 arms |', '|---|---:|']
    for ratio in ('1.01','1.05'):
        lines.append(f"| {ratio} | {sum(a['prefixes'][ratio].get('first_qualified_complete_sequence') is not None for a in combined['arms'])}/32 |")
    lines+=['','Remaining incomplete full diagnostic grids are the two rank-four seed642 retries (quadratic and raw-ReLU teachers). Their gaps, finite unresolved values and wall censoring remain visible. No interpolation crosses missing diagnostic slots; rank/cell/clock/normalization rules are unchanged. Medians use the two prescribed seeds on common finite support, with ranges rather than confidence intervals.','',
        'The six PNG/SVG sheets and two PDFs are in plots/. Numerical/metadata QA is saved in PLOT_QA.json, with full input/output hashes in PLOT_PROVENANCE.json. TARGET_PREFIX_COMPARISON.json records any newly identified earlier canonical candidate without hiding the original partial-grid assessment. Independent and visual reviews remain separate.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    report.check_hashes(combined['input_sha256']);report.check_hashes(plan['pinned_inputs'])
    receipt=dict(pass_check=True,inputs_unchanged=True,original_summary_unchanged=True,
        pre_sidecar_report_unchanged=True,unchanged_other_arm_count=31,no_model_evaluation=True,
        new_training_updates=0,independent_review='pending',visual_review='pending',
        input_sha256=combined['input_sha256'],script_sha256=report.sha(__file__),
        output_sha256={str(p.relative_to(output)):report.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    report.write_json(output/'REPORT_PROVENANCE.json',receipt)
    print(json.dumps(dict(output=str(output),diagnostics_complete=updated['diagnostics_complete'],other_arms_unchanged=31,
        preserved_successful_rows=5,new_diagnostic_rows=264,figures=6)))

if __name__=='__main__':main()
