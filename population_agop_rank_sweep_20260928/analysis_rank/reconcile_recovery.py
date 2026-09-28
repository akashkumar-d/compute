"""Separate shutdown reconciliation/report from immutable raw and canonical records.

Reads saved JSON only. Does not infer child exit status from a result marker,
recompute scientific criteria, evaluate a model, or change any input file.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--recovery-receipt', required=True, type=Path)
    args = parser.parse_args()
    output = args.report_dir.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('Report must remain inside analysis_rank')
    targets = ['RECONCILIATION.json', 'RECOVERY_RECEIPT.snapshot.json', 'RECOVERY_RESULTS.md', 'RECOVERY_REPORT_PROVENANCE.json']
    if any((output/name).exists() for name in targets):
        raise FileExistsError('Preserve earlier recovery reports')
    inputs = {}
    def read(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())
    summary = read(output/'SUMMARY.json')
    transfer = read(output/'DOWNLOAD_AUDIT.json')
    assert summary['snapshot_usable'] and summary['inputs_unchanged'] and transfer['pass_check']
    manifest = read(summary['manifest'])
    execution = Path(summary['execution_directory'])
    status = read(execution/'STATUS.json')
    stop = read(args.recovery_receipt)
    checks = stop['direct_checks']
    assert checks['lrun_status']['agop-ranks-2to16-20260928-a01'] == 'stopped'
    scan = checks['remote_process_scan']
    assert scan['exit_code'] == 0 and scan['matching_process_lines'] == []
    jobs = {job['id']: job for job in manifest['configs']}
    arms = {arm['id']: arm for arm in summary['arms']}
    assert len(jobs) == len(arms) == 32 and set(jobs) == set(arms)
    completed = {row['id']: row for row in status.get('completed', []) + status.get('finished', [])}
    active = {row['id']: row for row in status.get('active', [])}
    pending = set(status.get('pending', []))
    assert not (set(completed) & set(active)) and set(completed)|set(active)|pending == set(jobs)
    rows = []
    for ident, job in jobs.items():
        arm = arms[ident]
        folder = execution/'data'/ident
        marker = folder/('DONE.json' if job['engine'] == 'relu' else 'result.json')
        result = read(marker) if marker.exists() else None
        receipt = completed.get(ident)
        code = receipt.get('returncode', receipt.get('exit_code')) if receipt else None
        row = dict(id=ident, student=job['engine'], teacher=job['teacher'], rank=job['rank'], seed=job['seed'],
                   raw_canonical_process_state=arm['process_state'],
                   raw_status_partition='completed' if ident in completed else 'active_at_old_snapshot' if ident in active else 'pending_at_old_snapshot',
                   known_child_exit_code=code, child_exit_code_known=code is not None,
                   terminal_result_marker_present=result is not None,
                   marker_path=str(marker) if result is not None else None,
                   marker_sha256=inputs.get(str(marker)),
                   marker_completed_field=result.get('completed') if result is not None else None,
                   relu_combined_scientific_complete=result.get('complete') if result is not None and job['engine']=='relu' else None,
                   canonical_training_horizon_reached=arm.get('declared_horizon_reached'),
                   canonical_diagnostics_complete=arm.get('diagnostics_complete'),
                   saved_stop_reason=arm.get('stop_reason'),
                   authoritative_task_job_state='stopped',
                   matched_task_process_observed=False, process_scan_utc=scan['utc'],
                   reconciliation='stopped_with_terminal_result_marker' if result is not None else 'stopped_without_terminal_result_marker_partial_observations_retained',
                   observed_updates=arm.get('observed_updates'), final_saved_time=arm.get('final_time'),
                   final_saved_raw_loss=arm.get('final_loss_raw'),
                   diagnostic_warning_counts=arm.get('warning_counts', {}))
        rows.append(row)
    rows.sort(key=lambda a:(a['teacher'],a['student'],a['rank'],a['seed']))
    counts = dict(planned=32, terminal_result_markers=sum(r['terminal_result_marker_present'] for r in rows),
                  without_terminal_marker=sum(not r['terminal_result_marker_present'] for r in rows),
                  known_exit_zero=sum(r['known_child_exit_code']==0 for r in rows),
                  unknown_child_exit_code=sum(not r['child_exit_code_known'] for r in rows),
                  old_receipt_completed=len(completed), old_receipt_active=len(active), old_receipt_pending=len(pending),
                  raw_marker_by_student=dict(Counter(r['student'] for r in rows if r['terminal_result_marker_present'])))
    reconciliation = dict(created_utc=datetime.now(timezone.utc).isoformat(), counts=counts,
        source_status_utc=status['updated_utc'], authoritative_stop_receipt=str(args.recovery_receipt.resolve()),
        process_scan=scan, scope='Original attempt a01 only; any retry remains a separate attempt with the original 32-arm denominator retained.',
        no_exit_code_inferred_from_marker_or_shutdown=True, no_raw_receipt_or_canonical_summary_edits=True,
        receipt_rank_download_complete=stop.get('rank_download_complete'),
        receipt_rank_transfer_exit_code=stop.get('rank_transfer_exit_code'),
        transfer_evidence='DOWNLOAD_AUDIT.json checks every remote-inventory byte count and SHA256 after parent confirmed pull exit0.',
        rows=rows)
    (output/'RECOVERY_RECEIPT.snapshot.json').write_text(json.dumps(stop, indent=2)+'\n')
    (output/'RECONCILIATION.json').write_text(json.dumps(reconciliation, indent=2)+'\n')
    lines = ['# Recovered rank study, original attempt a01', '',
        f"All 32 declared arms are retained. Transfer verification passed for {transfer['verified_files']:,} files and {transfer['actual_bytes']:,} bytes, with no missing, extra or mismatched files. The unchanged canonical summary has {len(summary['reader_errors'])} reader errors, and its input snapshot is stable.", '',
        '## Shutdown reconciliation', '',
        f"The original STATUS.json was last written at {status['updated_utc']}: {len(completed)} exit-zero receipts, {len(active)} then-active arms, and {len(pending)} pending. It is stale. The independent recovery check reports the rank job stopped and no matching task workers at {scan['utc']}. The canonical summary deliberately preserves the stale `active_snapshot` labels; consult RECONCILIATION.json for stopped-process interpretation.", '',
        f"There are {counts['terminal_result_markers']} terminal result markers ({counts['raw_marker_by_student'].get('relu',0)} ReLU DONE.json and {counts['raw_marker_by_student'].get('swiglu',0)} SwiGLU result.json), and {counts['without_terminal_marker']} arms without a terminal result marker. Only {counts['known_exit_zero']} child exit codes are recorded; the other {counts['unknown_child_exit_code']} remain unknown. A result marker is not an exit-code receipt or a convergence certificate. The raw receipts and canonical summary remain unchanged.", '',
        'Quadratic SwiGLU rank 2, seed 642 wrote its result shortly after the last STATUS snapshot. Its saved initial-window candidate remains available, but many later diagnostics were not evaluated. All twelve rank 4/8/16 SwiGLU arms retain partial trajectories and missing frozen diagnostics. Their lack of a diagnosed candidate is not negative scientific evidence.', '',
        '## Same-checkpoint results by rank', '',
        'Counts below use the unchanged canonical rules, separately for each cell and its two planned reused seeds. O = original criterion; Q = additionally numerically qualified; L = qualified candidate with the required later within-run loss release. These are saved numerical observations, not formal population or high-probability certificates.', '',
        '| Teacher | Student | Rank | Terminal markers | 1% O / Q / L | 5% O / Q / L | 5% assessments (both seeds) |',
        '|---|---|---:|---:|---|---|---|']
    cells = []
    for teacher in ['h2','relu']:
        for student in ['relu','swiglu']:
            for rank in [2,4,8,16]:
                cell = sorted([a for a in arms.values() if (a['teacher'],a['student'],a['r'])==(teacher,student,rank)],key=lambda a:a['seed'])
                assert len(cell)==2
                tallies = {}
                for ratio in ['1.01','1.05']:
                    tallies[ratio] = {key:sum(a['prefixes'][ratio].get(field) is not None for a in cell)
                                     for key,field in [('original','first_material_candidate'),('qualified','first_numerically_qualified_candidate'),('qualified_later_release','first_qualified_complete_sequence')]}
                marker_count=sum(r['terminal_result_marker_present'] for r in rows if (r['teacher'],r['student'],r['rank'])==(teacher,student,rank))
                labels=[' / '.join(f"{tallies[ratio][key]}/2" for key in ['original','qualified','qualified_later_release']) for ratio in ['1.01','1.05']]
                assessments='; '.join(str(a['seed'])+': '+a['prefixes']['1.05']['assessment'] for a in cell)
                lines.append(f'| {teacher} | {student} | {rank} | {marker_count}/2 | {labels[0]} | {labels[1]} | {assessments} |')
                cells.append(dict(teacher=teacher,student=student,rank=rank,tallies=tallies))
    lines += ['', 'The ReLU student recovers quadratic teacher directions under the saved criterion at each studied rank (both seeds). For the ReLU student, raw-ReLU teacher cases do not show the intended initial-plateau sequence; at ranks 8 and 16, some 5% initial-window measurements remain numerically unresolved. Quadratic rank-2 SwiGLU has two observed candidates; raw-ReLU rank-2 SwiGLU has one. The other SwiGLU ranks cannot be compared scientifically from loss curves alone because their frozen diagnostics are missing.', '',
        '## Figures and limitations', '',
        '- [Quadratic full](plots/h2_full.png), [1% zoom](plots/h2_initial_1pct.png), [5% zoom](plots/h2_initial_5pct.png).',
        '- [Raw-ReLU full](plots/relu_full.png), [1% zoom](plots/relu_initial_1pct.png), [5% zoom](plots/relu_initial_5pct.png).',
        '- Matching SVGs, two three-page PDFs, complete rank records and plotted numerical arrays are under `plots/`.', '',
        'MSE and refit MSE are raw, with each rank’s target variance printed in the cell. Raw-ReLU variance changes with rank; h2 variance is one. ReLU uses accepted force time and bounded refits; SwiGLU uses adaptive flow time and unrestricted cutoff-sensitive refits. All learners here fit an output intercept for the unchanged raw teacher. No ranks, algorithms or clocks are pooled.', '',
        'Medians use both seeds on shared finite support. Bands are seed ranges, not confidence intervals. Missing frozen diagnostic records stay gaps; red markers disclose missing/unresolved points. No AGOP/refit values are reconstructed from training-only trajectory summaries. Long individual tails remain visible, without extrapolating the shorter run.', '',
        'This recovery analysis performs no new model evaluations, training, quadrature checks or independent Jacobian tests. Existing numerical guards are preserved, but they do not certify population accuracy. Reused seeds and two-seed cells cannot establish a success probability or a general rank scaling law. Initialization matching across ranks has not been re-audited in this reporting step. Incomplete diagnostic coverage and the Studio interruption prevent absence-of-learning conclusions for the twelve partial SwiGLU arms. Any authorized retries are separate attempts; they do not replace the original denominator or its failed/incomplete records.', '',
        'The saved RECOVERY_RECEIPT snapshot records the transfer state as supplied by the coordinating task. DOWNLOAD_AUDIT.json independently verifies all transferred bytes and hashes; no source receipt was edited by this reporting step. The direct stopped-process scan is timestamped evidence, not a new liveness probe. Visual inspection results are recorded separately in VISUAL_QA.json.', '']
    (output/'RECOVERY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    assert all(sha(path)==digest for path,digest in inputs.items()), 'Inputs changed while reporting'
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
                      inputs_sha256=inputs, inputs_unchanged=True, no_model_evaluation=True, training_runs=0,
                      process_counts=counts, cell_counts=cells,
                      output_sha256={name:sha(output/name) for name in targets if name!='RECOVERY_REPORT_PROVENANCE.json'})
    (output/'RECOVERY_REPORT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(counts))


if __name__=='__main__':
    main()
