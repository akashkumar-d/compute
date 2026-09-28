"""Schema, selection and inherited numerical guards on synthetic records only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import plot_rank as r
from make_fixture import fixture


class RankPresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((r.HERE.parent/'bundle/MANIFEST.json').read_text())

    def setUp(self):
        self.data = fixture(self.manifest)

    def test_all_planned_ids_and_natural_rank_order(self):
        before = copy.deepcopy(self.data)
        groups, recovered = r.select_groups(self.data, self.manifest)
        self.assertEqual(len(groups), 16)
        self.assertEqual([k[2] for k in r.ordered_keys()[:4]], [2, 4, 8, 16])
        self.assertEqual(len(recovered), 4)
        self.assertEqual(self.data, before)
        self.assertTrue(all([a['seed'] for a in c] == [641, 642] for c in groups.values()))

    def test_dropped_duplicate_and_unknown_ids_rejected(self):
        for kind in ['dropped', 'duplicate', 'unknown']:
            data = copy.deepcopy(self.data)
            if kind == 'dropped': data['arms'].pop()
            if kind == 'duplicate': data['arms'][-1] = copy.deepcopy(data['arms'][0])
            if kind == 'unknown': data['arms'][0]['id'] = 'unexpected'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.select_groups(data, self.manifest)

    def test_rank_seed_recipe_clock_and_config_mismatch_rejected(self):
        for field, bad in [('r', 2), ('seed', 999), ('scale', 999), ('time_unit', 'wrong'), ('supplemental', True),
                           ('teacher', 'h3'), ('student', 'swiglu'), ('cell', 'wrong')]:
            data = copy.deepcopy(self.data)
            data['arms'][0][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.select_groups(data, self.manifest)
        self.data['arms'][0]['config']['scale'] = 999
        with self.assertRaises(ValueError): r.select_groups(self.data, self.manifest)

    def test_unstable_summary_rejected(self):
        for field in ['snapshot_usable', 'inputs_unchanged']:
            data = copy.deepcopy(self.data); data[field] = False
            with self.assertRaises(ValueError): r.select_groups(data, self.manifest)

    def test_duplicate_manifest_cells_and_swiglu_rank_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['configs'][1]['cell'] = manifest['configs'][0]['cell']
        with self.assertRaises(ValueError): r.validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        job = next(j for j in manifest['configs'] if j['engine'] == 'swiglu')
        job['config'][0]['args']['c'].pop()
        with self.assertRaises(ValueError): r.validate_manifest(manifest)

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory(dir=r.HERE) as output:
            with self.assertRaises(FileExistsError):
                r.write_plots(r.HERE/'not_read.json', r.HERE.parent/'bundle/MANIFEST.json',
                              Path(output), 'SYNTHETIC', True)

    def test_no_one_seed_median_or_zoom(self):
        groups, _ = r.select_groups(self.data, self.manifest)
        for student in r.STUDENTS:
            cell = groups[student, 'relu', 16]
            self.assertIsNone(r.p.aggregate(cell))
            self.assertIsNone(r.zoom_end(cell, 1.01))
            self.assertIsNone(r.zoom_end(cell, 1.05))
            self.assertEqual(cell[1]['process_state'], 'not_started')
            self.assertEqual(cell[1]['history'], [])

    def test_exact_imported_aggregation_and_gaps(self):
        groups, _ = r.select_groups(self.data, self.manifest)
        cell = groups['relu', 'h2', 4]
        agg = r.p.aggregate(cell)
        self.assertEqual(agg['support'], [0., 30.])
        gap = (agg['x'] > 5) & (agg['x'] < 15)
        for metric in ['A_min', 'A_mean', 'refit']:
            self.assertTrue(np.isnan(agg['metrics'][metric]['range_median'][:, gap]).all())
        at_flag = agg['x'] == 20
        self.assertTrue(np.isfinite(agg['metrics']['A_min']['range_median'][:, at_flag]).all())
        self.assertFalse(agg['metrics']['A_min']['flags'][:, at_flag].any())

    def test_loss_only_two_zoom_rules(self):
        groups, _ = r.select_groups(self.data, self.manifest)
        cell = groups['relu', 'h2', 2]
        self.assertEqual(r.zoom_end(cell, 1.01), 15.)
        self.assertEqual(r.zoom_end(cell, 1.05), r.p.zoom_end(cell))
        original = [r.zoom_end(cell, ratio) for ratio in (1.01, 1.05)]
        for arm in cell:
            for point in arm['checkpoints']:
                point.update(A_min=999, A_mean=-999, refit_raw=10)
        self.assertEqual(original, [r.zoom_end(cell, ratio) for ratio in (1.01, 1.05)])

    def test_closed_loss_prefix_missing_diagnostics_label(self):
        groups, _ = r.select_groups(self.data, self.manifest)
        cell = groups['relu', 'h2', 2]
        for arm in cell:
            arm['issues'] = []
            for point in arm['checkpoints']:
                point['issues'] = []
        self.assertEqual(r.diagnostic_coverage_counts(cell), {'1.01': 0, '1.05': 2})
        for arm in cell:
            for point in arm['checkpoints']:
                point.update(agop_screen=True, refit_screen=True)
        self.assertEqual(r.diagnostic_coverage_counts(cell), {'1.01': 0, '1.05': 0})
        for arm in cell:
            arm['issues'] = []
            for point in arm['checkpoints']:
                point.update(diagnostic_status='not_evaluated', agop_screen=None, refit_screen=None)
        self.assertTrue(all(not r.p.prefix(arm, ratio)['right_censored'] for arm in cell for ratio in (1.01, 1.05)))
        self.assertEqual(r.diagnostic_coverage_counts(cell), {'1.01': 2, '1.05': 2})

    def test_variance_normalization_and_no_extrapolation(self):
        groups, _ = r.select_groups(self.data, self.manifest)
        arm = groups['swiglu', 'relu', 2][0]
        self.assertGreater(r.p.normalized_units_check(arm)['checked'], 0)
        arm['history'][0]['loss_raw'] *= 2
        with self.assertRaises(ValueError): r.p.normalized_units_check(arm)
        values, valid, _ = r.p.interpolate([1, 2], [.2, .4], [True, True], np.array([0, 1, 2, 3]))
        self.assertTrue(np.isnan(values[[0, 3]]).all())
        self.assertFalse(valid[[0, 3]].any())

    def test_unchanged_canonical_missing_execution_summary(self):
        spec = importlib.util.spec_from_file_location('canonical_summary_for_test', r.SUMMARIZER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        missing = r.HERE/'NO_SUCH_SYNTHETIC_EXECUTION'
        self.assertFalse(missing.exists())
        data = module.summarize(r.HERE.parent/'bundle/MANIFEST.json', missing)
        groups, _ = r.select_groups(data, self.manifest)
        self.assertEqual(sum(len(v) for v in groups.values()), 32)
        self.assertTrue(all(r.p.aggregate(c) is None for c in groups.values()))

    def test_report_keeps_all_arms_and_flags(self):
        groups, _ = r.select_groups(self.data, self.manifest)
        report = r.rank_report(groups, synthetic=True)
        self.assertEqual(sum(line.startswith('| h2 |') or line.startswith('| relu |') for line in report.splitlines()), 32)
        self.assertIn('SYNTHETIC FIXTURE ONLY', report)
        self.assertIn('not_started', report)


if __name__ == '__main__':
    unittest.main()
