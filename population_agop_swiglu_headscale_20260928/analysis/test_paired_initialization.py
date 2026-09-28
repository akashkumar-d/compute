"""Synthetic saved-data checks only; never reads experiment result folders."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import paired_initialization as audit_module


class InitializationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.execution = self.root / 'execution'
        self.manifest_path = self.root / 'MANIFEST.json'
        self.manifest = json.loads((audit_module.HERE.parent / 'bundle/MANIFEST.json').read_text())
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.folders = {}
        bases = {}
        for seed in audit_module.SEEDS:
            rng = np.random.default_rng(seed)
            bases[seed] = (rng.normal(size=(65, 64)), rng.normal(size=(65, 64)),
                           rng.normal(size=64) * 0.3)
        for job in self.manifest['configs']:
            folder = self.execution / 'data' / job['id']
            folder.mkdir(parents=True)
            self.folders[job['seed'], job['q']] = folder
            cfg = job['config'][0]['args']
            # Deliberately unrelated names prove that metadata, not tag naming,
            # selects the archive and its corresponding recorded row index.
            raw = dict(cfg=cfg, rows=[dict(step=0, t=0.0), dict(step=7, t=2.0)], partial=False)
            (folder / 'recorded-history.json').write_text(json.dumps(raw))
            P, V, base_head = bases[job['seed']]
            head = base_head * job['q']
            np.savez_compressed(folder / 'recorded-weights.npz', t=np.array([0.0, 2.0]),
                                P=np.stack([P, P + 1]), V=np.stack([V, V + 1]),
                                a=np.stack([head, head + 1]))
            result = dict(tag=job['id'], cfg=cfg, completed=True, raw_files={})
            (folder / 'result.json').write_text(json.dumps(result))
            self.refresh_receipt(folder)

    def refresh_receipt(self, folder):
        path = folder / 'result.json'
        result = json.loads(path.read_text())
        result['raw_files'] = {name: audit_module.digest(folder / name)
                               for name in ('recorded-history.json', 'recorded-weights.npz')}
        path.write_text(json.dumps(result))

    def result(self):
        return audit_module.audit(self.manifest_path, self.execution)

    def change_snapshot(self, folder, key, change, refresh=True):
        path = folder / 'recorded-weights.npz'
        with np.load(path, allow_pickle=False) as snapshots:
            arrays = {name: snapshots[name] for name in snapshots.files}
        change(arrays[key])
        np.savez_compressed(path, **arrays)
        if refresh:
            self.refresh_receipt(folder)

    def test_paired_arrays_scaled_heads_and_no_input_mutation(self):
        before = {str(p): audit_module.digest(p) for p in self.root.rglob('*') if p.is_file()}
        report = self.result()
        self.assertTrue(report['pass'])
        self.assertEqual(len(report['arms']), 6)
        self.assertEqual(len(report['comparisons']), 4)
        self.assertTrue(all(x['P_exact'] and x['V_exact'] for x in report['comparisons']))
        arms = {(x['seed'], x['q']): x for x in report['arms']}
        self.assertNotEqual(arms[641, 0.3]['raw_head'], arms[641, 1.0]['raw_head'])
        self.assertTrue(all(x['raw_head_equals_q_times_reference'] and
                            x['normalized_head_within_tolerance'] for x in report['comparisons']))
        self.assertEqual(before, {str(p): audit_module.digest(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertTrue(report['inputs_unchanged'])
        json.dumps(report, allow_nan=False)

    def test_hidden_weight_and_head_mismatches_are_separate(self):
        self.change_snapshot(self.folders[641, 0.3], 'P', lambda a: a.__setitem__((0, 64, 0), a[0, 64, 0] + 1e-12))
        self.change_snapshot(self.folders[642, 0.1], 'a', lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-9))
        report = self.result()
        pairs = {(x['seed'], x['q']): x for x in report['comparisons']}
        self.assertFalse(report['pass'])
        self.assertEqual(report['status'], 'mismatch')
        self.assertFalse(pairs[641, 0.3]['P_exact'])  # Includes the bias row.
        self.assertTrue(pairs[641, 0.3]['raw_head_equals_q_times_reference'])
        self.assertFalse(pairs[642, 0.1]['raw_head_equals_q_times_reference'])

    def test_missing_control_retains_six_arms_and_unresolved_pairs(self):
        (self.folders[641, 1.0] / 'recorded-weights.npz').unlink()
        report = self.result()
        self.assertFalse(report['pass'])
        self.assertEqual(len(report['arms']), 6)
        self.assertEqual(sum(x['status'] == 'missing' for x in report['arms']), 1)
        self.assertEqual(sum(x['status'] == 'unresolved' for x in report['comparisons']), 2)

    def test_initial_step_and_time_must_be_recorded(self):
        folder = self.folders[641, 0.3]
        path = folder / 'recorded-history.json'
        raw = json.loads(path.read_text())
        raw['rows'][0]['step'] = 1
        path.write_text(json.dumps(raw))
        self.refresh_receipt(folder)
        self.change_snapshot(self.folders[642, 0.1], 't', lambda a: a.__setitem__(0, 0.01))
        report = self.result()
        self.assertEqual(sum(x['status'] == 'unresolved' for x in report['arms']), 2)
        self.assertFalse(report['pass'])

    def test_nonfinite_and_hash_mismatch_are_unresolved(self):
        self.change_snapshot(self.folders[641, 0.3], 'a', lambda a: a.__setitem__((0, 0), np.nan))
        self.change_snapshot(self.folders[642, 0.1], 'V', lambda a: a.__setitem__((0, 0, 0), 9.0), refresh=False)
        report = self.result()
        self.assertEqual(sum(x['status'] == 'unresolved' for x in report['arms']), 2)
        self.assertFalse(report['pass'])
        json.dumps(report, allow_nan=False)

    def test_changed_input_cannot_pass(self):
        with patch.object(audit_module.Inputs, 'changed', return_value=['synthetic changed input']):
            report = self.result()
        self.assertFalse(report['pass'])
        self.assertFalse(report['inputs_unchanged'])
        self.assertEqual(report['status'], 'input_changed')


if __name__ == '__main__':
    unittest.main()
