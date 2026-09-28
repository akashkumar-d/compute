"""Saved-data retry reporting; immutable canonical arm records, fixed attempt rule."""
from __future__ import annotations
import argparse
import collections
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
V10=HERE.parent.parent
RANK=V10/'rank_sweep'
ORIGINAL_MANIFEST=RANK/'bundle/MANIFEST.json'
RETRY_MANIFEST=HERE.parent/'bundle/MANIFEST.json'
SELECTION=HERE.parent/'bundle/RETRY_SELECTION.json'
ORIGINAL_SUMMARY=RANK/'analysis_rank/recovery_a01/SUMMARY.json'
CANONICAL=RANK/'analysis/summarize_coverage.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def record_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def check_hashes(mapping):
    for name,digest in mapping.items():
        if not Path(name).is_file() or sha(name)!=digest:
            raise ValueError('Pinned source or saved input changed: '+name)


def select_records(original,retry,original_manifest,retry_manifest,selection):
    def index(rows):
        indexed={row['id']:row for row in rows}
        if len(indexed)!=len(rows):raise ValueError('Duplicate ID')
        return indexed
    jobs=index(original_manifest['configs']);retry_jobs=index(retry_manifest['configs'])
    old=index(original['arms']);new=index(retry['arms'])
    retry_ids=set(selection['retry_ids']);keep=set(selection['retained_completion_ids'])
    if (len(jobs)!=32 or len(retry_ids)!=12 or len(keep)!=20 or retry_ids&keep or
            retry_ids|keep!=set(jobs) or set(retry_jobs)!=retry_ids or set(old)!=set(jobs) or set(new)!=retry_ids):
        raise ValueError('Exact original32/retry12/retained20 partition is required')
    if len(selection['retry_ids'])!=12 or len(selection['retained_completion_ids'])!=20:
        raise ValueError('Duplicate selection IDs')
    if original['criteria']!=retry['criteria'] or original['script_sha256']!=retry['script_sha256']:
        raise ValueError('Canonical criteria/source mismatch')
    for summary in (original,retry):
        if summary['snapshot_usable'] is not True or summary['inputs_unchanged'] is not True or summary['reader_errors']:
            raise ValueError('Stable canonical source summaries with no reader errors are required')
    for ident in retry_ids:
        if jobs[ident]['config']!=retry_jobs[ident]['config']:
            raise ValueError('Retry configuration differs from original: '+ident)
    selected=[];attempt_map=[]
    for job in original_manifest['configs']:
        ident=job['id'];attempt=2 if ident in retry_ids else 1
        source=new[ident] if attempt==2 else old[ident]
        if source.get('config')!=job['config']:raise ValueError('Summary configuration mismatch: '+ident)
        selected.append(copy.deepcopy(source))
        attempt_map.append(dict(id=ident,attempt=attempt,
            selection_reason='missing_original_terminal_marker' if attempt==2 else 'original_terminal_marker_present',
            selected_record_sha256=record_sha(source),original_record_sha256=record_sha(old[ident]),
            retry_record_sha256=record_sha(new[ident]) if ident in new else None,
            selected_execution_directory=source['execution_directory'],
            canonical_process_state_unchanged=source['process_state']))
    return selected,attempt_map


def compose(original,retry,original_manifest,retry_manifest,selection,canonical,source_paths):
    arms,attempt_map=select_records(original,retry,original_manifest,retry_manifest,selection)
    combined=copy.deepcopy(original)
    combined.update(created_utc=datetime.now(timezone.utc).isoformat(),arms=arms,
        cells=canonical.build_cells(arms),execution_directory=None,runtime_allocation=None,
        status_snapshot=None,source_note='Derived fixed-attempt combination of unchanged canonical arm records; not one execution or a new replicate.',
        source_status_snapshots={'attempt1':original['status_snapshot'],'attempt2':retry['status_snapshot']},
        counts={kind:dict(planned=sum((not a['supplemental']) if kind=='canonical' else a['supplemental'] for a in arms),
            process_states=dict(collections.Counter(a['process_state'] for a in arms if
                ((not a['supplemental']) if kind=='canonical' else a['supplemental'])))) for kind in ('canonical','supplemental')})
    inputs={}
    for summary in (original,retry):
        for name,digest in summary['input_sha256'].items():
            if name in inputs and inputs[name]!=digest:raise ValueError('Source hash conflict: '+name)
            inputs[name]=digest
    inputs.update({str(Path(p).resolve()):sha(p) for p in source_paths})
    combined.update(input_sha256=inputs,input_files_missing=sorted(set(original['input_files_missing']+retry['input_files_missing'])),
        reader_errors=[],inputs_unchanged=True,changed_input_paths=[],snapshot_usable=True,
        derivation=dict(kind='predeclared_fixed_attempt_combination',combiner_sha256=sha(__file__),
            canonical_per_arm_summarizer_sha256=sha(CANONICAL),
            original_attempt_summary=str(ORIGINAL_SUMMARY),source_summaries=[str(p) for p in source_paths[:2]],
            study_denominator=32,retained_original_arms=20,retry_arms=12,new_independent_seeds=0,
            best_attempt_selection=False,arm_records_copied_verbatim=True,
            missing_input_list_scope='Union of both source summaries, including unselected original partial arms; use per-arm flags for selected diagnostic completeness.',
            original_raw_status_is_historical=True,rank2_completion_sidecar_included=False,
            combination_rule='Original terminal-marker availability only, frozen before retry outcomes; attempt1 for20, attempt2 for12.',
            attempt_map=attempt_map))
    return combined,attempt_map


def write_json(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def report_text(retry,combined,attempt_map):
    lines=['# Rank retry and fixed-attempt combined report','',
        'The retry summary contains exactly 12 original study arms. The derived combined view contains all 32: original attempt a01 for the 20 arms with terminal result markers in the frozen stop inventory, retry a02 for exactly the 12 without markers. The choice does not inspect outcomes. These are the same planned seeds 641/642; no new independent seeds or best-attempt selection are introduced.','',
        'Original attempt summaries and raw results remain unchanged. Every combined arm record is copied verbatim from its chosen canonical summary. Cell tables are rebuilt by the unchanged canonical helper. Original process-state labels remain historical: a stale active_snapshot is not a claim of current activity, and a terminal result marker is not an exit-code receipt.','',
        'The original rank-two quadratic SwiGLU seed642 retains its five successful diagnostics and 264 missing rows in this version. The separately reviewed diagnostics completion pass is not silently included. Missing/unresolved states and censoring remain explicit.','',
        '| View | Arms | Has result | Diagnostics complete | Training wall-censored |',
        '|---|---:|---:|---:|---:|']
    for name,summary in [('Retry a02 only',retry),('Fixed-attempt combined',combined)]:
        arms=summary['arms'];lines.append(f"| {name} | {len(arms)} | {sum(bool(a.get('has_result')) for a in arms)} | {sum(a.get('diagnostics_complete') is True for a in arms)} | {sum(a.get('training_wall_censored') is True for a in arms)} |")
    lines+=['','Counts use the unchanged same-checkpoint criteria. O = original material candidate; Q = additionally numerically qualified; L = qualified candidate plus the required later same-run loss release. Two-seed counts are development evidence, not probabilities.','',
        '| Teacher | Student | Rank | Attempt(s) | 1% O / Q / L | 5% O / Q / L | 5% assessments |',
        '|---|---|---:|---|---|---|---|']
    attempts={row['id']:row['attempt'] for row in attempt_map}
    for teacher in ('h2','relu'):
        for student in ('relu','swiglu'):
            for rank in (2,4,8,16):
                cell=sorted([a for a in combined['arms'] if (a['teacher'],a['student'],a['r'])==(teacher,student,rank)],key=lambda a:a['seed'])
                if len(cell)!=2:raise ValueError('Combined cell must preserve two prescribed seeds')
                texts=[]
                for ratio in ('1.01','1.05'):
                    texts.append(' / '.join(f"{sum(a['prefixes'][ratio].get(field) is not None for a in cell)}/2" for field in
                        ('first_material_candidate','first_numerically_qualified_candidate','first_qualified_complete_sequence')))
                which=', '.join(f"{a['seed']}: a0{attempts[a['id']]}" for a in cell)
                assessments='; '.join(f"{a['seed']}: {a['prefixes']['1.05']['assessment']}" for a in cell)
                lines.append(f'| {teacher} | {student} | {rank} | {which} | {texts[0]} | {texts[1]} | {assessments} |')
    lines+=['','No model evaluations, training, new quadrature checks, or changes to success criteria occur here. ReLU and SwiGLU use different clocks/refit comparators and remain separate. Ranks are not pooled. The derived summary preserves both source summaries’ full missing-input lists; some entries concern unselected original partial attempts and are not missing selected-arm results. Per-arm diagnostic/censoring flags govern interpretation.','',
        'Files: RETRY12_SUMMARY.json (unchanged canonical summarizer), COMBINED32_SUMMARY.json (explicit derived combination), ATTEMPT_MAP.json (selection and per-record hashes), REPORT_PROVENANCE.json (all source/summary hashes), and the canonical retry Markdown summary. Figures, if rendered, are a separate fresh gallery using the unchanged rank/canonical plotting helpers.','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',required=True,type=Path)
    parser.add_argument('--download-audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if output.parent!=HERE or output.exists():raise ValueError('Output must be a new direct child of this analysis directory')
    plan=read(HERE/'PLAN.json');check_hashes(plan['pinned_inputs'])
    audit=read(args.download_audit)
    inventory_path=HERE.parent/'REMOTE_FINAL_INVENTORY.json'
    if (audit.get('passed') is not True or audit.get('missing') or audit.get('mismatched') or
            audit.get('inventory_sha256')!=sha(inventory_path)):
        raise ValueError('Parent-approved successful transfer/hash audit is required')
    execution=args.execution.resolve()
    expected_execution=HERE.parent/'remote_results/population_agop_rank_retry_a02_20260928/bundle/execution_remote_a02'
    if execution != expected_execution or not (execution/'data').is_dir():
        raise ValueError('Expected the parent-downloaded retry execution')
    original=read(ORIGINAL_SUMMARY);original_manifest=read(ORIGINAL_MANIFEST);retry_manifest=read(RETRY_MANIFEST);selection=read(SELECTION)
    check_hashes(original['input_sha256'])
    canonical=module(CANONICAL,'unchanged_rank_canonical_summary')
    retry=canonical.summarize(RETRY_MANIFEST,execution)
    # Validate before creating any reporting artifacts.
    select_records(original,retry,original_manifest,retry_manifest,selection)
    check_hashes(retry['input_sha256'])
    status=retry['status_snapshot']
    completed=status.get('completed',[])
    if (status.get('status')!='completed' or status.get('active') or status.get('pending') or len(completed)!=12
            or {row['id'] for row in completed}!={job['id'] for job in retry_manifest['configs']}
            or any(row.get('returncode')!=0 for row in completed)):
        raise ValueError('Require the saved 12/12 exit-zero terminal retry receipt')
    output.mkdir()
    retry_path=output/'RETRY12_SUMMARY.json';write_json(retry_path,retry)
    (output/'RETRY12_SUMMARY.md').write_text(canonical.markdown(retry))
    source_paths=[ORIGINAL_SUMMARY,retry_path,HERE/'PLAN.json',SELECTION,args.download_audit,inventory_path,Path(__file__)]
    combined,attempt_map=compose(original,retry,original_manifest,retry_manifest,selection,canonical,source_paths)
    combined_path=output/'COMBINED32_SUMMARY.json';write_json(combined_path,combined)
    write_json(output/'ATTEMPT_MAP.json',dict(rule=plan['combination_rule'],rows=attempt_map,
        study_denominator=32,new_independent_seeds=0,best_attempt_selection=False))
    (output/'REPORT.md').write_text(report_text(retry,combined,attempt_map))
    if args.plots:
        os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
        plotter=module(RANK/'analysis_rank/plot_rank.py','unchanged_rank_plot_helpers')
        # Presentation output confinement only: the new authorized reporting root.
        # No source file or numerical function is modified.
        plotter.HERE=HERE
        plotter.write_plots(combined_path,ORIGINAL_MANIFEST,output/'plots',
            'Rank study: fixed original/retry attempts; development seeds 641 and 642')
    check_hashes(combined['input_sha256']);check_hashes(plan['pinned_inputs'])
    selected_lookup={a['id']:a for a in combined['arms']}
    assert all(record_sha(selected_lookup[r['id']])==r['selected_record_sha256'] for r in attempt_map)
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),pass_check=True,
        input_sha256=combined['input_sha256'],inputs_unchanged=True,script_sha256=sha(__file__),
        plan_sha256=sha(HERE/'PLAN.json'),canonical_summarizer_sha256=sha(CANONICAL),
        original_summary_sha256=sha(ORIGINAL_SUMMARY),retry_summary_sha256=sha(retry_path),
        combined_summary_sha256=sha(combined_path),no_model_evaluation=True,new_training_updates=0,
        new_independent_seeds=0,best_attempt_selection=False,all_selected_records_verbatim=True,
        original_summary_untouched=True,rank2_completion_sidecar_included=False,
        figures_rendered=args.plots,independent_review='pending',visual_review='pending' if args.plots else 'not_rendered',
        output_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'REPORT_PROVENANCE.json',provenance)
    print(json.dumps(dict(output=str(output),retry_arms=12,combined_arms=32,all_selected_records_verbatim=True,
        original_arms_retained=20,new_independent_seeds=0,figures_rendered=args.plots)))

if __name__=='__main__':main()
