"""Synthetic adapter tests; never import or evaluate a scientific engine."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'swiglu/code/engine/swsmall_periodic.py'
RESUME = ROOT / 'adapter/resume_engine.py'
spec = importlib.util.spec_from_file_location('resume_state_test_subject', ROOT / 'adapter/resume_state.py')
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def functions(path, names, namespace):
    """Execute only selected function definitions, never source imports/main."""
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('Requested synthetic test functions were not found')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def synthetic_namespace(stop_at=None):
    env = dict(np=np, math=math, sys=sys, time=time, copy=copy,
               _STOP_REQUESTED=False, _write_partial=lambda *a, **k: None,
               make_link=lambda _: (None, None))

    class Teacher:
        def __init__(self, *args, **kwargs):
            self.gamma, self.EY, self.V = 1., 0., 1.

    class SyntheticEngine:
        """Deterministic artificial values solely to test control flow."""
        def __init__(self, d, teacher, **kwargs):
            self.d, self.T, self.calls = d, teacher, 0

        def evaluate(self, P, V, a, need_grad=True):
            if stop_at is not None and self.calls == stop_at:
                env['_STOP_REQUESTED'] = True
            self.calls += 1
            return dict(L=float(1. + .04 * P.sum()),
                        gP=np.full_like(P, .01), gV=np.full_like(V, .004),
                        ga=np.full_like(a, .002))

        def agop(self, P, V, a, ev):
            return np.diag(np.arange(1, self.d + 1, dtype=float))

        def refit(self, ev):
            return .5 * ev['L']

    env['swpop'] = types.SimpleNamespace(Teacher=Teacher, SwiGLUPop=SyntheticEngine)
    env['vlab'] = types.SimpleNamespace(top_r_alignment=lambda M, r:
                                     dict(A=.5, cos2_min=.25, gap=1.))
    functions(FROZEN, ['validate_head_lr', 'update_directions',
                      'first_ratio_crossing', 'alignment_stats'], env)
    return env


def configuration():
    return dict(d=4, m=4, c=[1., 1.], link='synthetic', alpha=1., s=.1,
                head_ratio=1., head_lr=.001, seed=12, n_x=2, n_pair=2,
                n_diag=2, n_z=2, h=10., dt_max=.1, t_max=4., L_stop=.1,
                max_steps=100, cp_min=.4, cp_ratio=1.06, dl_ratio=1.5,
                out='', verbose=False)


def original_run(cfg, stop_at=None):
    env = synthetic_namespace(stop_at)
    functions(FROZEN, ['run'], env)
    return env['run'](cfg)


def checkpoint_from(raw):
    snapshots = raw['_snaps']
    prior = json.loads(json.dumps({k: v for k, v in raw.items() if k != '_snaps'}))
    return dict(raw=prior, snaps=snapshots,
                control=state.recover_control(prior, prior['cfg']))


def resumed_run(cfg, checkpoint):
    env = synthetic_namespace()
    functions(RESUME, ['run'], env)
    return env['run'](cfg, checkpoint)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()
        self.full = original_run(self.cfg)
        self.stopped = original_run(self.cfg, stop_at=6)
        self.checkpoint = checkpoint_from(self.stopped)

    def test_exact_mock_trajectory_and_clock_without_rng(self):
        before = copy.deepcopy(self.checkpoint)
        with patch.object(np.random, 'default_rng', side_effect=AssertionError('Resume must not draw RNG')):
            resumed = resumed_run(self.cfg, self.checkpoint)
        self.assertGreater(resumed['termination']['step'] - self.stopped['termination']['step'], 10)
        self.assertEqual(resumed['termination']['step'], self.full['termination']['step'])
        self.assertEqual(resumed['termination']['t'], self.full['termination']['t'])
        self.assertEqual(resumed['termination']['L'], self.full['termination']['L'])
        self.assertEqual(resumed['termination']['reason'], self.full['termination']['reason'])
        self.assertEqual(json.loads(json.dumps(resumed['Lhist'])),
                         json.loads(json.dumps(self.full['Lhist'])))
        self.assertEqual(resumed['step_history'], self.full['step_history'])
        self.assertEqual(resumed['ratio_prefixes'], self.full['ratio_prefixes'])
        for actual, expected in zip(resumed['_snaps'][-1][1:], self.full['_snaps'][-1][1:]):
            np.testing.assert_array_equal(actual, expected)
        # Original rows are retained, and every uninterrupted checkpoint survives.
        by_step = {r['step']: r for r in resumed['rows']}
        for row in self.full['rows']:
            self.assertEqual(by_step[row['step']], row)
        self.assertEqual(self.checkpoint['raw'], before['raw'])
        for actual, expected in zip(self.checkpoint['snaps'], before['snaps']):
            self.assertEqual(actual[0], expected[0])
            for x, y in zip(actual[1:], expected[1:]):
                np.testing.assert_array_equal(x, y)

    def test_forced_terminal_does_not_reset_scheduler(self):
        control = self.checkpoint['control']
        self.assertTrue(control['forced_terminal_extra'])
        self.assertNotIn(6, control['natural_grid_steps'])
        terminal = self.stopped['termination']
        artificial_next = max(terminal['t'] * self.cfg['cp_ratio'],
                              terminal['t'] + self.cfg['cp_min'])
        self.assertNotEqual(control['next_cp'], artificial_next)
        resumed = resumed_run(self.cfg, self.checkpoint)
        expected = [r['step'] for r in self.full['rows']]
        actual = [r['step'] for r in resumed['rows']]
        self.assertEqual(actual, sorted(set(expected + [6])))

    def test_anchor_mismatch_refused_before_advancing(self):
        for key in ('L', 'dt', 'rel'):
            with self.subTest(key=key):
                checkpoint = copy.deepcopy(self.checkpoint)
                checkpoint['raw']['step_history'][-1][key] += 1e-7
                with self.assertRaisesRegex(ValueError, 'Anchor reevaluation differs'):
                    resumed_run(self.cfg, checkpoint)

    def test_scientific_configuration_change_refused(self):
        changed = dict(self.cfg, head_lr=.002)
        with self.assertRaisesRegex(ValueError, 'Scientific configuration changed'):
            state.recover_control(self.checkpoint['raw'], changed)

    def test_bad_clock_and_terminal_metadata_refused(self):
        changed = copy.deepcopy(self.checkpoint['raw'])
        changed['step_history'][2]['t'] += .001
        with self.assertRaises(ValueError):
            state.recover_control(changed, self.cfg)
        changed = copy.deepcopy(self.checkpoint['raw'])
        changed['termination']['t'] += .001
        with self.assertRaisesRegex(ValueError, 'Terminal metadata'):
            state.recover_control(changed, self.cfg)

    def test_frozen_input_scheduler_control(self):
        raw_path = next((ROOT / 'input').glob('v9_swiglu*order4*.json'))
        raw = json.loads(raw_path.read_text())
        control = state.recover_control(raw, raw['cfg'])
        self.assertEqual(len(control['natural_grid_steps']), 40)
        self.assertEqual(control['natural_grid_steps'][-1], 431)
        self.assertTrue(control['forced_terminal_extra'])
        self.assertEqual(control['next_cp'], 40.53101837127362)
        self.assertEqual(control['last_dL'], .05299185929986161)
        self.assertEqual(control['last_L'], .9470081407001384)
        self.assertEqual(control['prefix_exits']['ratio_1pct']['first_exit_step'], 258)
        self.assertEqual(control['prefix_exits']['ratio_5pct']['first_exit_step'], 382)

    def test_diagnostic_plan_preserves_old_successes_and_prioritizes_gaps(self):
        raw_path = next((ROOT / 'input').glob('v9_swiglu*order4*.json'))
        raw = json.loads(raw_path.read_text())
        old = json.loads((ROOT / 'input/diagnostics_partial.json').read_text())
        before = copy.deepcopy(old)
        # One artificial new row tests scheduling only; no metrics are evaluated.
        raw['rows'].append(dict(step=438, t=39.))
        computed = [dict(index=i, step=r['step'], t=r['t'], status='not_evaluated')
                    for i, r in enumerate(raw['rows'])]
        original_order = list(reversed(range(len(raw['rows']))))
        order, policy = state.diagnostic_plan(raw, old, computed, original_order)
        self.assertEqual(old, before)
        self.assertEqual([raw['rows'][i]['step'] for i in order[:2]], [358, 380])
        self.assertEqual(len(policy['preserved_success_indices']), 29)
        self.assertEqual(len(policy['missing_original_indices']), 16)
        self.assertEqual(order[-1], 45)
        self.assertEqual(set(order), set(range(46)) - set(policy['preserved_success_indices']))
        self.assertEqual(len(order), len(set(order)))
        for prior in old['rows']:
            if prior['status'] == 'ok':
                actual = dict(computed[prior['index']])
                self.assertEqual(actual.pop('diagnostic_segment'), 'preserved_interrupted_segment')
                self.assertEqual(actual, prior)

    def test_diagnostic_plan_rejects_mismatched_checkpoint(self):
        raw_path = next((ROOT / 'input').glob('v9_swiglu*order4*.json'))
        raw = json.loads(raw_path.read_text())
        old = json.loads((ROOT / 'input/diagnostics_partial.json').read_text())
        old['rows'][0]['step'] = -1
        with self.assertRaisesRegex(ValueError, 'same preserved checkpoint'):
            state.diagnostic_plan(raw, old, [{} for _ in raw['rows']], list(range(len(raw['rows']))))


if __name__ == '__main__':
    unittest.main()
