"""Small synthetic tests of the new display adapter, not scientific outcomes."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import plot_headscale as h


def fixture(manifest):
    arms=[]
    for job in manifest['configs']:
        q=job['q'];seed=job['seed'];last=4 if seed==641 else 3
        arm=h.rank.metadata(job)
        losses=[1.,.999,.992,.965,.85]
        history=[dict(step=i,time=10.*i/q,loss_raw=losses[i],loss_over_target_variance=losses[i]) for i in range(last+1)]
        points=[]
        for row in history:
            i=row['step'];missing=i==2 and seed==641;screen=not(i==3 and seed==642)
            points.append(dict(**row,A_min=None if missing else [.05,.7,.8,.9,.95][i],
                               A_mean=None if missing else [.2,.8,.9,.95,.99][i],
                               refit_raw=None if missing else [.9,.4,.3,.2,.1][i],
                               diagnostic_status='not_evaluated' if missing else 'ok',
                               agop_screen=None if missing else screen,refit_screen=None if missing else screen,issues=[]))
        candidate=copy.deepcopy(points[1]);candidate.update(A_min_gain=.65,refit_gain_lower_over_variance=.5)
        prefixes={str(r):dict(end_step=end,end_time=10.*end/q,right_censored=seed==642 and r==1.05,
                             assessment='SYNTHETIC_candidate',first_material_candidate=copy.deepcopy(candidate),
                             first_numerically_qualified_candidate=copy.deepcopy(candidate),first_qualified_complete_sequence=None,
                             missing_diagnostic_steps=[2] if seed==641 else [],unresolved_diagnostic_steps=[3] if seed==642 and end>=3 else [])
                  for r,end in [(1.01,2),(1.05,3)]}
        arm.update(config=copy.deepcopy(job['config']),history=history,checkpoints=points,prefixes=prefixes,
                   target_mean=0.,target_variance=1.,process_state='synthetic_only',stop_reason='SYNTHETIC_fixture',
                   diagnostics_complete=False,issues=[])
        arms.append(arm)
    return dict(schema='breadth14_saved_coverage_v1',snapshot_usable=True,inputs_unchanged=True,arms=arms,
                input_sha256={},fixture_provenance=dict(kind='synthetic',purpose='display-only clock and layout checks; no pilot results'))


class ClockDisplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((h.ROOT/'bundle/MANIFEST.json').read_text())

    def test_clock_copy_preserves_candidates_values_and_source(self):
        data=fixture(self.manifest);cells,_=h.select_groups(data,self.manifest)
        for q,cell in cells.items():
            original=copy.deepcopy(cell);shown=h.display_cell(cell,q)
            self.assertEqual(cell,original)
            for before,after in zip(cell,shown):
                restored=copy.deepcopy(after)
                for field in ('history','checkpoints'):
                    for a,b in zip(before[field],restored[field]):
                        self.assertEqual(b['time'],q*a['time']);b['time']=a['time']
                for r,pref in before['prefixes'].items():
                    self.assertEqual(after['prefixes'][r]['end_time'],q*pref['end_time'])
                    restored['prefixes'][r]['end_time']=pref['end_time']
                self.assertEqual(restored,before)
            self.assertEqual(h.counts(cell),h.counts(shown))

    def test_aggregate_copy_changes_only_abscissa_and_shared_support(self):
        cells,_=h.select_groups(fixture(self.manifest),self.manifest)
        for q,cell in cells.items():
            agg=h.p.aggregate(cell);before=copy.deepcopy(agg);shown=h.display_aggregate(agg,q)
            np.testing.assert_array_equal(shown['x'],agg['x']*q)
            np.testing.assert_allclose(shown['support'],[0.,30.],rtol=0,atol=1e-14)
            np.testing.assert_array_equal(agg['x'],before['x'])
            for metric,info in agg['metrics'].items():
                for key,value in info.items():np.testing.assert_array_equal(shown['metrics'][metric][key],value)
            gap=(shown['x']>10)&(shown['x']<30)
            self.assertTrue(np.isnan(shown['metrics']['A_min']['range_median'][:,gap]).all())
            end=shown['x']==30
            self.assertTrue(np.isfinite(shown['metrics']['A_min']['range_median'][:,end]).all())
            self.assertFalse(shown['metrics']['A_min']['flags'][:,end].all())

    def test_no_one_seed_support_after_scaling_and_loss_only_zoom(self):
        cells,_=h.select_groups(fixture(self.manifest),self.manifest)
        cell=cells[.3];shown=h.display_cell(cell,.3)
        self.assertAlmostEqual(h.rank.zoom_end(shown,1.01),20)
        self.assertAlmostEqual(h.rank.zoom_end(shown,1.05),30)
        for arm in shown:
            for point in arm['checkpoints']:point.update(A_min=999,refit_raw=-999)
        self.assertAlmostEqual(h.rank.zoom_end(shown,1.01),20)
        cell[1]['history']=[]
        self.assertIsNone(h.p.aggregate(cell));self.assertIsNone(h.display_aggregate(None,.3))
        self.assertIsNone(h.rank.zoom_end(h.display_cell(cell,.3),1.01))

    def test_all_six_outcomes_and_exact_q_recipes_required(self):
        data=fixture(self.manifest)
        for broken in ['dropped','duplicate','wrong_recipe']:
            changed=copy.deepcopy(data)
            if broken=='dropped':changed['arms'].pop()
            if broken=='duplicate':changed['arms'][0]=copy.deepcopy(changed['arms'][1])
            if broken=='wrong_recipe':changed['arms'][0]['config'][0]['args']['head_lr']=99
            with self.subTest(broken=broken),self.assertRaises(ValueError):h.select_groups(changed,self.manifest)

    def test_invalid_display_scale_rejected(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):h.display_cell([],value)
            with self.subTest(value=value),self.assertRaises(ValueError):h.display_aggregate(None,value)

    def test_outer_config_identity_disagreement_rejected(self):
        for key,value in [('seed',9351),('link','h4')]:
            manifest=copy.deepcopy(self.manifest);manifest['configs'][0]['config'][0]['args'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.select_groups(fixture(manifest),manifest)
        manifest=copy.deepcopy(self.manifest);manifest['configs'][0]['config'][0]['rank']=16
        with self.assertRaises(ValueError):h.select_groups(fixture(manifest),manifest)


if __name__=='__main__':unittest.main()
