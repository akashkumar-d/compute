"""Create explicitly synthetic plotting input; no models, training or observations."""
import argparse
import copy
import json
import math
from pathlib import Path
import plot_rank as r


def fixture(manifest):
    arms = []
    for job in manifest['configs']:
        a = r.metadata(job)
        rank, seed = a['r'], a['seed']
        mu = 1/math.sqrt(2*math.pi) if a['teacher'] == 'relu' else 0.
        second = .5 if a['teacher'] == 'relu' else 1.
        denominator = second + (rank-1)*mu*mu
        variance = (second-mu*mu)/denominator
        mean = math.sqrt(rank)*mu/math.sqrt(denominator)
        stop = 7 if seed == 641 else 6
        clock_scale = 5. if a['student'] == 'relu' else 25.
        loss = [1., .999, .997, .994, .988, .971, .940, .85][:stop+1]
        a.update(config=copy.deepcopy(job['config']), target_mean=mean, target_variance=variance,
                 process_state='SYNTHETIC_fixture_exit_zero', stop_reason='SYNTHETIC_training_cap',
                 diagnostics_complete=True, has_result=True, issues=['SYNTHETIC_DATA_NOT_OBSERVATIONS'],
                 history=[dict(step=i, time=clock_scale*i, loss_raw=value*variance,
                               loss_over_target_variance=value) for i, value in enumerate(loss)],
                 checkpoints=[dict(step=i, time=clock_scale*i, diagnostic_status='ok',
                                   A_min=.005*rank+.085*i+.01*(seed-641),
                                   A_mean=.02*rank+.065*i,
                                   refit_raw=variance*(.93-.07*i), agop_screen=i != 4,
                                   refit_screen=i != 5, issues=['SYNTHETIC']) for i in range(stop+1)], prefixes={})
        for ratio, end in [(1.01, 3), (1.05, 5)]:
            a['prefixes'][str(ratio)] = dict(end_step=end, end_time=clock_scale*end,
                right_censored=False, exit_observed=True, assessment='SYNTHETIC_no_candidate',
                first_material_candidate=None, first_numerically_qualified_candidate=None,
                first_complete_sequence=None, first_qualified_complete_sequence=None)
        if rank == 4 and seed == 641:
            a['checkpoints'][2].update(diagnostic_status='SYNTHETIC_missing', A_min=None,
                                      A_mean=None, refit_raw=None, agop_screen=None, refit_screen=None)
            a['diagnostics_complete'] = False
        if rank == 16 and seed == 642:
            a.update(history=[], checkpoints=[], has_result=False, process_state='not_started',
                     diagnostics_complete=False, stop_reason=None)
            for prefix in a['prefixes'].values():
                prefix.update(end_time=None, end_step=None, right_censored=True, exit_observed=False,
                              assessment='SYNTHETIC_unavailable')
            # Reproduce the real canonical error-fallback omission shape.
            for field in ['r', 'd', 'm', 'scale', 'head_lr', 'profiled_intercept', 'time_unit', 'config']:
                a.pop(field, None)
            a['target_mean'] = a['target_variance'] = None
        arms.append(a)
    return dict(schema='breadth14_saved_coverage_v1', snapshot_usable=True, inputs_unchanged=True,
                planned_arms=32, canonical_arms=32, supplemental_arms=0, arms=arms,
                input_sha256={}, fixture_provenance=dict(kind='synthetic',
                    description='Deterministic artificial curves, gaps, flags, unequal tails and absent seed arms. No rank experiment result read.',
                    generated_by=str(Path(__file__).resolve())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=r.HERE.parent/'bundle/MANIFEST.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(r.HERE) or out.exists():
        raise ValueError('Choose a new fixture file within analysis_rank')
    manifest = json.loads(args.manifest.read_text())
    r.validate_manifest(manifest)
    data = fixture(manifest)
    data['fixture_provenance']['manifest_sha256'] = r.sha(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(out)


if __name__ == '__main__':
    main()
