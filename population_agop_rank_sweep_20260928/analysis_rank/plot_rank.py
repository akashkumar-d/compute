"""Rank-specific presentation of the unchanged canonical saved-data summary.

Imports no model/runner code. Numerical interpolation, aggregation, diagnostics,
unit checking and curve drawing are delegated to the byte-identical v8 plotter.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent / 'analysis/plot_coverage.py'
SUMMARIZER = HERE.parent / 'analysis/summarize_coverage.py'
V8 = HERE.parents[2] / 'goal_followup_v8/targeted30/analysis/plot_coverage.py'
RANKS = (2, 4, 8, 16)
TEACHERS = ('h2', 'relu')
STUDENTS = ('relu', 'swiglu')
SEEDS = (641, 642)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(CANONICAL) != sha(V8):
    raise RuntimeError('Canonical plot copy differs from v8; review before rendering')
spec = importlib.util.spec_from_file_location('rank_canonical_plot', CANONICAL)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def config_of(job):
    outer = job['config'][0] if isinstance(job['config'], list) else job['config']
    return outer, outer.get('args', outer)


def metadata(job):
    outer, c = config_of(job)
    student = job['engine']
    return dict(id=job['id'], cell=job['cell'], student=student,
                teacher=p.canonical_name(c.get('teacher', c.get('link'))),
                seed=c['seed'], r=c.get('r', outer.get('rank')), d=c['d'], m=c['m'],
                scale=c.get('scale', c.get('s')),
                head_lr=c.get('head_lr', 1.) if student == 'swiglu' else None,
                profiled_intercept=bool(c.get('profile_intercept', False)) if student == 'relu' else True,
                supplemental=bool(job.get('supplemental', False)),
                time_unit='accepted_force_time' if student == 'relu' else 'adaptive_population_flow_time')


def validate_manifest(manifest):
    jobs = manifest.get('configs', [])
    if manifest.get('total_arms') != 32 or len(jobs) != 32:
        raise ValueError('Expected the full declared 32-arm rank study')
    records = [metadata(j) for j in jobs]
    ids = [m['id'] for m in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate planned ID')
    expected = set(itertools.product(STUDENTS, TEACHERS, RANKS, SEEDS))
    actual = [(m['student'], m['teacher'], m['r'], m['seed']) for m in records]
    if set(actual) != expected or len(set(actual)) != 32:
        raise ValueError('Manifest must contain each student/teacher/rank/seed exactly once')
    for job, m in zip(jobs, records):
        outer, c = config_of(job)
        if m['student'] == 'swiglu' and len(c.get('c', [])) != m['r']:
            raise ValueError(f'SwiGLU rank/teacher coefficient-count mismatch: {m["id"]}')
        if m['d'] != 64 or m['supplemental'] or m['m'] != (256 if m['student'] == 'relu' else 64):
            raise ValueError(f'Unexpected study dimension, width or supplemental arm: {m["id"]}')
        for outer_key, key in [('student', 'student'), ('teacher', 'teacher'), ('rank', 'r'), ('seed', 'seed'), ('width', 'm')]:
            if job.get(outer_key) != m[key]:
                raise ValueError(f'Outer/config manifest disagreement: {m["id"]}:{outer_key}')
    cells = {}
    for m in records:
        cells.setdefault((m['student'], m['teacher'], m['r']), set()).add(m['cell'])
    if any(len(v) != 1 for v in cells.values()) or len(set.union(*cells.values())) != 16:
        raise ValueError('Each rank cell requires one unique cell ID shared by its two seeds')
    return {m['id']: m for m in records}


def select_groups(data, manifest):
    """Validate all planned identities, recovering only absent planning metadata.

    Canonical exception arms omit rank/recipe fields. Those may be filled from
    the manifest for placement, never with invented histories, metrics or flags.
    """
    planned = validate_manifest(manifest)
    if data.get('schema') != 'breadth14_saved_coverage_v1':
        raise ValueError('Use the unchanged canonical summarize_coverage.py schema')
    if data.get('snapshot_usable') is not True or data.get('inputs_unchanged') is not True:
        raise ValueError('Stable, usable canonical summary required')
    arms = data.get('arms', [])
    ids = [a['id'] for a in arms]
    if len(ids) != 32 or len(set(ids)) != 32 or set(ids) != set(planned):
        raise ValueError('Summary must retain exactly all 32 planned IDs, including unavailable arms')
    result = []
    recovered = []
    jobs = {job['id']: job for job in manifest['configs']}
    for source in arms:
        arm = copy.deepcopy(source)
        plan = planned[arm['id']]
        if arm.get('config') is not None and arm['config'] != jobs[arm['id']]['config']:
            raise ValueError(f'Summary/manifest full configuration mismatch: {arm["id"]}')
        fields = []
        for key, value in plan.items():
            observed = arm.get(key)
            if key == 'teacher' and observed is not None:
                observed = p.canonical_name(observed)
            if observed is None:
                arm[key] = value
                if value is not None:
                    fields.append(key)
            elif observed != value:
                raise ValueError(f'Summary/manifest disagreement: {arm["id"]}:{key}')
            elif key == 'teacher':
                arm[key] = observed
        if fields:
            recovered.append(dict(id=arm['id'], planning_fields=fields))
        result.append(arm)
    # This runs the existing clock, within-cell recipe and normalization guards.
    canonical_groups = p.grouping({'arms': result}, paired_study=True)
    groups = {}
    for (student, teacher, _), cell in canonical_groups.items():
        key = student, teacher, cell[0]['r']
        if key in groups:
            raise ValueError(f'Duplicate rank cell: {key}')
        groups[key] = cell
    expected = set(itertools.product(STUDENTS, TEACHERS, RANKS))
    if set(groups) != expected:
        raise ValueError('Incomplete rank cells')
    return groups, recovered


def zoom_end(cell, ratio):
    """The v8 loss-only shared-prefix rule with an explicit 1% or 5% ratio."""
    bounds = p.support(cell)
    ends = [p.prefix(arm, ratio).get('end_time') for arm in cell]
    if bounds is None or any(not p.finite(t) for t in ends):
        return None
    return min(bounds[1], *ends)


def ordered_keys():
    return list(itertools.product(TEACHERS, STUDENTS, RANKS))


def diagnostic_coverage_counts(cell):
    """Presentation only: count arms with saved prefix gaps/warnings/unavailability."""
    result = {}
    for ratio in (1.01, 1.05):
        count = 0
        for arm in cell:
            prefix = p.prefix(arm, ratio)
            end = prefix.get('end_step')
            points = [point for point in arm.get('checkpoints', [])
                      if end is not None and point.get('step') is not None and point['step'] <= end]
            incomplete = (not prefix or end is None or not points
                          or bool(prefix.get('missing_diagnostic_steps'))
                          or bool(prefix.get('unresolved_diagnostic_steps'))
                          or bool(arm.get('issues'))
                          or any(point.get('diagnostic_status') != 'ok'
                                 or point.get('agop_screen') is not True
                                 or point.get('refit_screen') is not True
                                 or bool(point.get('issues')) for point in points))
            count += int(incomplete)
        result[str(ratio)] = count
    return result


def criterion_text(arm, ratio):
    prefix = p.prefix(arm, ratio)
    original = prefix.get('first_material_candidate')
    qualified = prefix.get('first_numerically_qualified_candidate')
    def event(value):
        return 'step ' + str(value.get('step', '?')) if value is not None else 'none recorded'
    return (f"original {event(original)}; qualified {event(qualified)}; "
            f"{prefix.get('assessment', 'unavailable')}; "
            f"{'open/unavailable' if not prefix or prefix.get('right_censored') else 'exit observed'}")


def rank_report(groups, synthetic=False):
    lines = ['# Rank-specific saved-data report', '']
    if synthetic:
        lines += ['**SYNTHETIC FIXTURE ONLY — no scientific observations.**', '']
    lines += ['All 32 declared arms remain visible. Ranks are ordered 2, 4, 8, 16; each cell contains the two reused development seeds 641/642. No ranks are pooled.', '',
              'Counts and candidates come unchanged from the canonical per-run summary. Plot interpolation never decides an event. Original and numerically qualified candidates are distinct. Refit improvement must be at least 0.1 Var(Y) at the same checkpoint, alongside the minimum-direction gain criterion; later loss release is a separate within-run observation.', '',
              '| Teacher | Student | Rank | Seed | Process / stop | Var(Y) | 1% initial prefix | 5% initial prefix | Diagnostics / issues |',
              '|---|---|---:|---:|---|---:|---|---|---|']
    for teacher, student, rank in ordered_keys():
        for arm in groups[student, teacher, rank]:
            items = [teacher, student, str(rank), str(arm['seed']),
                     str(arm.get('process_state', 'unavailable')) + ' / ' + p.status_text(arm),
                     p.fmt(arm.get('target_variance')), criterion_text(arm, 1.01), criterion_text(arm, 1.05),
                     ('complete' if arm.get('diagnostics_complete') else 'incomplete/unavailable') + '; ' + json.dumps(arm.get('issues', []))]
            lines.append('| ' + ' | '.join(x.replace('|', '\\|').replace('\n', ' ') for x in items) + ' |')
    lines += ['', 'Both prediction and refit rows display raw MSE. The normalized target has E[Y²]=1, while raw ReLU target variance changes with rank. Profiled intercepts fit the raw target mean; no teacher centering is substituted. ReLU uses accepted force time and bounded-head refits; SwiGLU uses adaptive flow time and unrestricted cutoff-sensitivity refits. These clocks and comparator classes differ.', '',
              'Thick curves are two-seed medians only where both seeds have finite interpolable measurements. Bands are seed ranges, not confidence intervals. Finite unresolved values remain raw curves/medians with warning marks; missing diagnostics stay gaps. Missing arms contribute no one-seed median. All stop/censoring/qualification fields remain in RANK_RECORDS.json.', '']
    return '\n'.join(lines)


def render_page(teacher, groups, aggregates, ratio, title, synthetic):
    fig = p.plt.figure(figsize=(18.5, 13.4))
    outer = fig.add_gridspec(2, 4, left=.075, right=.985, bottom=.19, top=.835, wspace=.36, hspace=.39)
    mode = 'Full observed trajectories' if ratio is None else f'Initial {round(100*(ratio-1))}% loss-prefix zoom'
    teacher_label = 'Normalized quadratic h₂' if teacher == 'h2' else 'Normalized raw ReLU'
    fig.text(.53, .976, title, ha='center', fontsize=14, color='black')
    fig.text(.53, .946, f'{teacher_label} teacher · {mode}', ha='center', fontsize=12, color='black')
    fig.text(.53, .917, 'Ranks 2, 4, 8, 16 · two reused development seeds per cell · E[Y²] = 1; Var(Y) shown per rank', ha='center', fontsize=9.5, color='black')
    fig.text(.075, .877, 'ReLU student · bounded refit · accepted force clock', fontsize=11, color='black')
    fig.text(.075, .502, 'SwiGLU student · unrestricted cutoff refit · adaptive flow clock', fontsize=11, color='black')
    original_zoom = p.zoom_end
    try:
        if ratio is not None:
            p.zoom_end = lambda cell: zoom_end(cell, ratio)
        for row, student in enumerate(STUDENTS):
            for column, rank in enumerate(RANKS):
                key = student, teacher, rank
                start = len(fig.axes)
                p.cell_axes(fig, outer[row, column], student, teacher, groups[key], aggregates[key], ratio is not None)
                # Presentation-only distinction: a closed loss window can still
                # have entirely missing or unresolved diagnostic measurements.
                header = fig.axes[start]
                header.get_subplotspec().get_gridspec().set_height_ratios([.82, 1, 1, 1])
                label = header.texts[1]
                text = label.get_text().replace('Censored/unavailable 1%/5%', 'Loss-prefix censored/unavailable 1%/5%')
                coverage = diagnostic_coverage_counts(groups[key])
                text += f"\nDiagnostic gaps/warnings 1%/5%: {coverage['1.01']}/2, {coverage['1.05']}/2"
                label.set_text(text)
                label.set_y(.51)
    finally:
        p.zoom_end = original_zoom
    handles = [p.Line2D([0], [0], color=p.COLORS['loss'], label='Raw MSE'),
               p.Line2D([0], [0], color=p.COLORS['A_min'], label='AGOP minimum'),
               p.Line2D([0], [0], color=p.COLORS['A_mean'], ls='--', label='AGOP mean'),
               p.Line2D([0], [0], color=p.COLORS['refit'], label='Refit raw MSE'),
               p.Patch(facecolor=p.COLORS['prefix5'], label='Shared initial 5% prefix'),
               p.Patch(facecolor=p.COLORS['prefix1'], alpha=.3, label='Shared initial 1% prefix'),
               p.Line2D([0], [0], color=p.COLORS['warning'], marker='|', lw=0, label='Unresolved'),
               p.Line2D([0], [0], color=p.COLORS['warning'], marker='x', lw=0, label='Missing diagnostic')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .105), ncol=4, frameon=False, fontsize=8.5)
    fig.text(.075, .080, 'Bold: two-seed medians on shared support. Bands: seed ranges, not confidence intervals. Faint curves retain individual tails; red markers retain warnings.', fontsize=9, color='black')
    fig.text(.075, .059, 'Missing arms/diagnostics remain unavailable; no one-seed median or interpolation across missing points. Raw unresolved scores are retained, not certified.', fontsize=9, color='black')
    fig.text(.075, .038, 'Windows use every recorded loss update; no alignment/refit-based zoom selection. ReLU and SwiGLU clocks, numerical precision and refit classes differ.', fontsize=9, color='black')
    if synthetic:
        fig.text(.075, .016, 'SYNTHETIC FIXTURE ONLY — layout and schema checks; no measured rank-sweep outcomes.', fontsize=10, color='black', weight='bold')
    fig.canvas.draw()
    return fig


def write_plots(summary_path, manifest_path, output, title, synthetic=False):
    summary_path, manifest_path, output = (Path(x).resolve() for x in (summary_path, manifest_path, output))
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError('Output must be a fresh child directory of analysis_rank')
    if output.exists():
        raise FileExistsError('Preserve previous reports; choose a new output directory')
    tracked = {str(q): sha(q) for q in [summary_path, manifest_path, Path(__file__), CANONICAL, V8, SUMMARIZER]}
    data, manifest = json.loads(summary_path.read_text()), json.loads(manifest_path.read_text())
    is_fixture = data.get('fixture_provenance', {}).get('kind') == 'synthetic'
    if synthetic != is_fixture or (synthetic and 'synthetic' not in title.lower()):
        raise ValueError('Synthetic fixtures require --synthetic-fixture and an explicit SYNTHETIC title')
    if not synthetic:
        if data.get('script_sha256') != sha(SUMMARIZER):
            raise ValueError('Summary was not produced by the current unchanged canonical summarizer')
        if data.get('input_sha256', {}).get(str(manifest_path)) != sha(manifest_path):
            raise ValueError('Summary must hash this exact manifest')
    groups, recovered = select_groups(data, manifest)
    aggregates = {key: p.aggregate(cell) for key, cell in groups.items()}
    units = [p.normalized_units_check(a) for cell in groups.values() for a in cell]
    unavailable = []
    for name, digest in data.get('input_sha256', {}).items():
        path = Path(name)
        if path.is_file():
            if sha(path) != digest:
                raise RuntimeError(f'Source input changed since summarization: {name}')
            tracked[name] = digest
        else:
            unavailable.append(name)
    output.mkdir(parents=True)
    p.configure()
    p.plt.rcParams.update({'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})
    artifacts = []
    for teacher in TEACHERS:
        pdf_path = output / f'{teacher}_rank_sheets.pdf'
        with p.PdfPages(pdf_path) as pdf:
            for ratio, suffix in [(None, 'full'), (1.01, 'initial_1pct'), (1.05, 'initial_5pct')]:
                fig = render_page(teacher, groups, aggregates, ratio, title, synthetic)
                for ext in ['png', 'svg']:
                    path = output / f'{teacher}_{suffix}.{ext}'
                    fig.savefig(path, dpi=140, facecolor='white')
                    artifacts.append(path)
                pdf.savefig(fig, facecolor='white')
                p.plt.close(fig)
        artifacts.append(pdf_path)
    cells = []
    for teacher, student, rank in ordered_keys():
        key = student, teacher, rank
        cell, agg = groups[key], aggregates[key]
        name = f'{teacher}_{student}_r{rank:02d}'
        if agg is not None:
            path = output / f'aggregate_{name}.npz'
            p.np.savez_compressed(path, x=agg['x'], **{f'{metric}_{field}': value for metric, info in agg['metrics'].items() for field, value in info.items()})
            artifacts.append(path)
        cells.append(dict(teacher=teacher, student=student, rank=rank,
                          planned_ids=[a['id'] for a in cell], planned_seeds=[a['seed'] for a in cell],
                          shared_support=agg['support'] if agg else None,
                          zoom_end={str(r): zoom_end(cell, r) for r in (1.01, 1.05)},
                          diagnostic_gaps_or_warnings_by_prefix=diagnostic_coverage_counts(cell),
                          arms=cell))
    report = output / 'RANK_REPORT.md'
    report.write_text(rank_report(groups, synthetic))
    artifacts.append(report)
    records = output / 'RANK_RECORDS.json'
    records.write_text(json.dumps(p.clean(dict(cells=cells, recovered_planning_metadata=recovered)), indent=2, allow_nan=False)+'\n')
    artifacts.append(records)
    if any(sha(Path(name)) != digest for name, digest in tracked.items()):
        raise RuntimeError('An input or analysis source changed during rendering')
    qa = dict(pass_check=True, synthetic_fixture_only=synthetic, no_model_evaluation=True, training_runs=0,
              planned_arms=32, rank_order=list(RANKS), no_rank_pooling=True,
              interpolation_aggregation_and_curve_drawing='unchanged imported v8 functions',
              canonical_plot_sha256=sha(CANONICAL), canonical_summarizer_sha256=sha(SUMMARIZER),
              two_seed_medians_on_shared_finite_support=True, seed_ranges_not_confidence_intervals=True,
              missing_diagnostics_remain_gaps=True, unresolved_finite_scores_retained_with_flags=True,
              zoom_selection='min(shared history endpoint, both saved loss-only prefix endpoints), separately at 1% and 5%',
              normalized_units_checks=units, inputs_unchanged=True, visual_review='pending',
              cells=[{k: v for k, v in c.items() if k != 'arms'} for c in cells])
    (output/'PLOT_QA.json').write_text(json.dumps(p.clean(qa), indent=2, allow_nan=False)+'\n')
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), title=title, synthetic_fixture_only=synthetic,
                      input_sha256=tracked, source_paths_unavailable_for_local_recheck=unavailable,
                      source_hashes_from_summary=data.get('input_sha256', {}),
                      fixture_provenance=data.get('fixture_provenance'), colors=p.COLORS, label_color='black',
                      output_sha256={f.name: sha(f) for f in artifacts})
    (output/'PLOT_PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), planned_arms=32, cells=len(cells), sheets=6, synthetic_fixture=synthetic)))
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--title', required=True)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args()
    write_plots(args.summary, args.manifest, args.output, args.title, args.synthetic_fixture)


if __name__ == '__main__':
    main()
