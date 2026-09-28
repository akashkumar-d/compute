"""Meaningful saved-record edge fixtures and immutable v6 regression checks."""
from pathlib import Path
import copy,json,tempfile,unittest,sys,hashlib
from unittest import mock
import summarize_coverage as s

HERE=Path(__file__).resolve().parent
W=HERE.parents[2]

def history(losses):return [dict(step=i,time=float(i),loss_raw=x,loss_over_target_variance=x) for i,x in enumerate(losses)]
def point(step,amin,risk,loss=1.,**kw):
    p=s.empty_point(step,float(step));p.update(loss_raw=loss,A_min=amin,A_mean=.99,A_top=1.,refit_raw=risk,refit_low_raw=risk,refit_high_raw=risk,
        agop_screen=True,refit_screen=True,raw_refit_nonnegative_screen=True,diagnostic_status='ok');p.update(kw);return p

def arm(losses,points,var=1.,student='relu'):
    h=history(losses)
    return dict(student=student,target_variance=var,history=h,checkpoints=points,configuration_matches=True,issues=[],
        prefixes={str(r):s.exact_prefix(h,r) for r in s.RATIOS})

class EdgeCases(unittest.TestCase):
    def test_max_min_uses_all_updates_not_first_last(self):
        p=s.exact_prefix(history([1.,1.009,.995,.994]),1.01)
        self.assertEqual(p['end_step'],1);self.assertEqual(p['first_exit_step'],2)
        self.assertFalse(p['right_censored']);self.assertAlmostEqual(p['actual_ratio'],1.009)
    def test_exact_boundary_inclusive_next_crossing_excluded(self):
        a=arm([1.,.995,.95,.7],[point(0,0.,.9),point(1,.6,.7,.995),point(2,.8,.3,.95)])
        s.evaluate_arm(a);self.assertEqual(a['prefixes']['1.05']['candidate_steps'],[1])
        self.assertEqual(a['prefixes']['1.05']['first_complete_sequence']['step'],1)
    def test_cannot_splice_alignment_and_refit_checkpoints(self):
        a=arm([1.,.999,.998,.94],[point(0,0.,.9),point(1,.7,.85,.999),point(2,.2,.6,.998)])
        s.evaluate_arm(a);self.assertFalse(a['prefixes']['1.05']['first_material_candidate'])
    def test_mean_and_top1_never_replace_minimum(self):
        a=arm([1.,.999,.94],[point(0,0.,.9),point(1,.02,.1,.999)])
        s.evaluate_arm(a);self.assertEqual(a['prefixes']['1.05']['candidate_count'],0)
    def test_open_window_is_not_negative(self):
        a=arm([1.,.999],[point(0,0.,.9),point(1,.1,.8,.999)]);s.evaluate_arm(a)
        self.assertEqual(a['prefixes']['1.05']['assessment'],'window_right_censored')
    def test_missing_required_diagnostic_is_not_negative(self):
        a=arm([1.,.999,.94],[point(0,0.,.9),s.empty_point(1,1.)]);s.evaluate_arm(a)
        self.assertEqual(a['prefixes']['1.05']['assessment'],'diagnostics_missing');self.assertEqual(a['prefixes']['1.05']['missing_diagnostic_steps'],[1])
    def test_missing_original_guard_is_unresolved(self):
        a=arm([1.,.999,.94],[point(0,0.,.9),point(1,.9,.1,.999,agop_screen=None)]);s.evaluate_arm(a)
        self.assertEqual(a['prefixes']['1.05']['assessment'],'numerically_unresolved')
    def test_additional_refit_qualification_does_not_overwrite_original(self):
        a=arm([1.,.999,.94],[point(0,0.,.9),point(1,.7,.6,.999,refit_screen=False)]);s.evaluate_arm(a)
        p=a['prefixes']['1.05'];self.assertEqual(p['candidate_count'],1);self.assertEqual(p['qualified_candidate_count'],0)
        self.assertEqual(p['numerical_qualification_assessment'],'original_candidate_numerically_unqualified')
    def test_low_initial_refit_retained_as_ineligible(self):
        a=arm([1.,.999,.94],[point(0,0.,.05),point(1,.7,0.,.999)]);s.evaluate_arm(a)
        self.assertEqual(a['effect_size_eligibility'],'initial_refit_already_below_material_threshold');self.assertFalse(a['prefixes']['1.05']['first_material_candidate'])
    def test_negative_roundoff_cannot_manufacture_material_gain(self):
        a=arm([1.,.999,.94],[point(0,0.,.1-5e-12),point(1,.7,-1e-11,.999)]);s.evaluate_arm(a)
        self.assertFalse(a['prefixes']['1.05']['first_material_candidate']);self.assertLess(a['checkpoints'][1]['refit_gain_lower_raw'],.1)
        self.assertEqual(a['checkpoints'][1]['refit_raw'],-1e-11)
    def test_later_loss_is_from_candidate_and_requires_later_step(self):
        a=arm([1.,.999,.995],[point(0,0.,.9),point(1,.7,.6,.999),point(2,.8,.5,.995)]);s.evaluate_arm(a)
        self.assertFalse(a['prefixes']['1.05']['first_complete_sequence']);self.assertIsNone(a['checkpoints'][-1]['later_loss_min_raw'])
    def test_source_config_mismatch_blocks_success(self):
        a=arm([1.,.999,.7],[point(0,0.,.9),point(1,.7,.6,.999)]);a['configuration_matches']=False;s.evaluate_arm(a)
        self.assertEqual(a['prefixes']['1.05']['assessment'],'input_inconsistent')
    def test_swiglu_variance_applied_exactly_once_and_negative_rejected(self):
        row=dict(status='ok',agop_Amin=.7,agop_A=.9,agop_full_rank_resolved=True,agop_psd_resolved=True,agop_resolution_guard=1e-9,
            agop_rank_r_eigenvalue_relative=.1,agop_rank_r_relative_gap=.01,equilibrated_lambda_ratio=.1,
            pinv={k:dict(actual_mse=.5,relative_normal_residual=1e-12) for k in ['1e-08','1e-10','1e-12','1e-14']})
        p=s.swiglu_point(dict(step=1,t=2.,L=.99),row,.2)
        self.assertAlmostEqual(p['loss_raw'],.198);self.assertAlmostEqual(p['refit_raw'],.1);self.assertTrue(p['refit_screen'])
        row['pinv']['1e-14']['actual_mse']=-.1;p=s.swiglu_point(dict(step=1,t=2.,L=.99),row,.2)
        self.assertFalse(p['raw_refit_nonnegative_screen']);self.assertFalse(p['refit_screen'])
    def test_missing_swi_cutoff_never_gives_valid_envelope(self):
        row=dict(status='ok',pinv={'1e-12':dict(actual_mse=.5,relative_normal_residual=1e-12)})
        p=s.swiglu_point(dict(step=1,t=2.,L=.99),row,.2);self.assertIsNone(p['refit_low_raw']);self.assertFalse(p['refit_screen'])
    def test_invalid_loss_does_not_create_completed_window(self):
        p=s.exact_prefix(history([1.,float('nan'),.5]),1.05)
        self.assertTrue(p['invalid_loss']);self.assertTrue(p['right_censored']);self.assertFalse(p['exit_observed'])
    @unittest.skipUnless((HERE.parent/'bundle/MANIFEST.json').is_file(), 'Local frozen 60-arm manifest unavailable; portable edge fixtures still run')
    def test_all_sixty_unstarted_retained_without_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=s.summarize(HERE.parent/'bundle/MANIFEST.json',Path(tmp)/'execution')
        self.assertEqual(x['planned_arms'],60);self.assertEqual(x['canonical_arms'],56);self.assertEqual(x['supplemental_arms'],4);self.assertEqual(len(x['cells']),28)
        self.assertTrue(all(a['process_state']=='not_started' for a in x['arms']))
        self.assertTrue(all(a['prefixes']['1.05']['assessment']=='not_started' for a in x['arms']))
        self.assertTrue(x['inputs_unchanged'])
    def test_completed_receipt_uses_returncode_and_does_not_infer_scientific_success(self):
        label,r=s.process_state('a',{'completed':[dict(id='a',returncode=0)]},{},True);self.assertEqual(label,'exit_zero')
        label,r=s.process_state('a',{'completed':[dict(id='a',returncode=1)]},{},True);self.assertEqual(label,'failed_or_terminated')

    def test_invalid_bound_order_blocks_original_not_only_qualification(self):
        a=arm([1.,.999,.94],[point(0,0.,.05,refit_low_raw=.9,refit_screen=False),point(1,.7,.2,.999)])
        s.evaluate_arm(a);self.assertIsNone(a['prefixes']['1.05']['first_material_candidate'])
        self.assertIn('refit_interval_order_invalid',a['checkpoints'][0]['issues'])
    def test_unknown_required_grid_never_resolves_negative(self):
        a=arm([1.,.999,.94],[point(0,0.,.9),point(1,.1,.8,.999)]);a['required_grid_provenance']='observed_rows_only';s.evaluate_arm(a)
        self.assertEqual(a['prefixes']['1.05']['assessment'],'required_diagnostic_grid_unknown')
    def test_corrupt_or_missing_snapshot_hash_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'state.npz').write_bytes(b'corrupt');r=s.Reader()
            issues=r.verify_hashes(root,{'state.npz':'0'*64,'absent.npz':'f'*64})
            self.assertIn('recorded_file_hash_mismatch:state.npz',issues);self.assertIn('recorded_file_missing:absent.npz',issues)
            self.assertIsNone(r.snapshot_metadata(root/'state.npz'));self.assertTrue(r.errors)
    def test_snapshot_shapes_and_times_loaded_without_model_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'s.npz';s.np.savez(p,t=s.np.array([0.,1.]),P=s.np.zeros((2,3,4)),V=s.np.zeros((2,3,4)),a=s.np.zeros((2,4)))
            sm=s.Reader().snapshot_metadata(p);self.assertEqual(sm['shapes']['P'],[2,3,4]);self.assertEqual(sm['times'],[0.,1.])
    def test_reader_hashes_preserve_original_on_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.json';p.write_text('{}');r=s.Reader();r.read(p);before=dict(r.hashes);p.write_text('{"changed":true}')
            self.assertFalse(r.unchanged());self.assertEqual(r.hashes,before);self.assertEqual(r.changed_paths,[str(p.resolve())])
    @unittest.skipUnless((HERE.parent/'bundle/MANIFEST.json').is_file(), 'Local frozen 60-arm manifest unavailable; Reader mutation fixtures still run')
    def test_mutated_snapshot_keeps_cells_and_clears_all_positive_counts(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(s.Reader,'unchanged',return_value=False):
            x=s.summarize(HERE.parent/'bundle/MANIFEST.json',Path(tmp)/'execution')
        self.assertFalse(x['snapshot_usable']);self.assertEqual(len(x['cells']),28)
        self.assertTrue(all(a['prefixes']['1.05']['assessment']=='input_snapshot_changed' for a in x['arms']))
        self.assertTrue(all(c['windows']['1.05']['material_candidates']==0 and c['windows']['1.05']['numerically_qualified_candidates']==0 for c in x['cells']))

    def test_clipped_nonnegative_lower_bound_keeps_conservative_gap(self):
        p=s.relu_point(dict(step=0,refit=1e-9,refit_lower=0.,refit_gap=1e-6,refit_info={'numerical_lower_bound':0.,'objective_gap_bound':1e-6}))
        self.assertNotIn('refit_interval_gap_inconsistent',p['issues'])
    def test_relu_horizon_and_signal_stops_use_saved_training_flags(self):
        c=dict(id='case',student='relu',teacher='h3',r=8,d=64,m=256,seed=1,scale=.01,h=.5,steps=2,profile_intercept=True,force_time_max=1.)
        job=dict(id='case',engine='relu',config=c)
        for reason,censored,reached in [(None,False,True),('force_time_horizon',False,True),('signal_15',True,False)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'meta.json').write_text(json.dumps({'config':c}));(root/'TRAIN_RESULT.json').write_text(json.dumps({'training_censored':censored,'complete':not censored,'declared_horizon_or_loss_target_reached':reached,'stop_reason':reason,'observed_steps':1 if censored else 2}))
                n=2 if censored else 3;s.np.save(root/'loss_every_step.npy',s.np.array([1.,.999,.998][:n]));s.np.save(root/'accepted_h.npy',s.np.array([.5]*(n-1)))
                a=s.load_arm(job,root,s.Reader(),{})
                self.assertEqual(a['training_wall_censored'],censored);self.assertEqual(a['declared_horizon_reached'],reached)

    def test_changed_then_restored_reread_does_not_reseal_mixed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';p.write_text('{}');r=s.Reader();r.read(p);first=dict(r.hashes)
            p.write_text('{"new":true}');r.read(p);r.file_hash(p);p.write_text('{}')
            self.assertEqual(r.hashes,first);self.assertFalse(r.unchanged());self.assertEqual(r.changed_paths,[str(p.resolve())])
    def test_missing_snapshot_verification_blocks_only_extra_qualification(self):
        a=arm([1.,.999,.94],[point(0,0.,.9),point(1,.7,.6,.999)]);a['snapshot_integrity_verified']=False;s.evaluate_arm(a)
        self.assertIsNotNone(a['prefixes']['1.05']['first_material_candidate']);self.assertIsNone(a['prefixes']['1.05']['first_numerically_qualified_candidate'])

class SavedV6Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required=[W/'goal_followup_v6'/x for x in [
            'relu_confirmation/design/MANIFEST.json','relu_confirmation/execution',
            'relu_confirmation/analysis/GOAL_METRICS.json','swiglu_head_rate_v1/design/MANIFEST.json',
            'swiglu_head_rate_v1/execution','swiglu_head_rate_v1/analysis/SUMMARY.json',
            'lightning_sine_confirmation/MANIFEST.json','lightning_sine_confirmation/remote_results/execution',
            'lightning_sine_analysis/analysis/GOAL_METRICS.json']]
        missing=[str(p) for p in required if not p.exists()]
        if missing:raise unittest.SkipTest('Local historical v6 data unavailable; portable edge fixtures still run. Missing: '+', '.join(missing))
        (HERE/'validation').mkdir(exist_ok=True)
        cls.reports={}
        for name,folder,manifest,execution in [('v6_relu','relu_confirmation','design/MANIFEST.json','execution'),('v6_swiglu','swiglu_head_rate_v1','design/MANIFEST.json','execution'),('v6_h3_sine','lightning_sine_confirmation','MANIFEST.json','remote_results/execution')]:
            root=W/'goal_followup_v6'/folder;d=s.summarize(root/manifest,root/execution);cls.reports[name]=d
            (HERE/'validation'/f'{name}_summary.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
            (HERE/'validation'/f'{name}_summary.md').write_text(s.markdown(d))
    def test_relu_original_prefixes_and_first_candidates_match_prior_saved_analysis(self):
        for label,path in [('v6_relu','relu_confirmation/analysis/GOAL_METRICS.json'),('v6_h3_sine','lightning_sine_analysis/analysis/GOAL_METRICS.json')]:
            old={a['id']:a for a in json.loads((W/'goal_followup_v6'/path).read_text())['runs']}
            for a in self.reports[label]['arms']:
                self.assertFalse(a['issues'],a['id'])
                for k,p in a['prefixes'].items():
                    o=old[a['id']]['windows'][k]
                    self.assertEqual(p['end_step'],o['endpoint']);self.assertEqual(p['right_censored'],o['right_censored'])
                    self.assertEqual(p['candidate_count'],o['candidate_count'])
                    if o['first_candidate']:
                        self.assertEqual(p['first_material_candidate']['step'],o['first_candidate']['step'])
                        self.assertAlmostEqual(p['first_material_candidate']['refit_gain_lower_raw'],o['first_candidate']['refit_gain_lower_or_sensitivity_raw'])
    def test_swiglu_same_windows_candidates_and_censored_cubic_match(self):
        old={a['id']:a for a in json.loads((W/'goal_followup_v6/swiglu_head_rate_v1/analysis/SUMMARY.json').read_text())['arms']}
        for a in self.reports['v6_swiglu']['arms']:
            self.assertFalse(a['issues'])
            for key,ok in [('1.01','ratio_1pct'),('1.05','ratio_5pct')]:
                p=a['prefixes'][key];o=old[a['id']]
                self.assertEqual(p['end_step'],o['prefixes'][ok]['end_step']);self.assertEqual(p['right_censored'],o['prefixes'][ok]['right_censored'])
                c=o['candidates'][ok]['first_material_candidate'];n=p['first_material_candidate']
                self.assertEqual(None if n is None else n['step'],None if c is None else c['step'])
                if n:self.assertAlmostEqual(n['cutoff_envelope_gain_raw'],c['refit_gain_envelope_raw'])
    def test_old_small_scale_original_positives_and_additional_gap_warning_separate(self):
        arms=[a for a in self.reports['v6_relu']['arms'] if 'smaller_scale' in a['id']]
        self.assertEqual(len(arms),2)
        for a in arms:
            p=a['prefixes']['1.05'];self.assertIsNotNone(p['first_material_candidate']);self.assertIsNone(p['first_numerically_qualified_candidate'])
            self.assertGreater(p['first_material_candidate']['refit_gap_raw'],1e-7)
    def test_no_input_changes_or_model_errors(self):
        for d in self.reports.values():self.assertTrue(d['inputs_unchanged']);self.assertFalse(d['reader_errors'])
    def test_empty_quadrature_checks_not_claimed_passed(self):
        for a in self.reports['v6_swiglu']['arms']:
            for p in a['checkpoints']:
                if not p['quadrature_double_checked']:self.assertEqual(p['quadrature_followup_audit_status'],'not_recomputed_at_this_checkpoint')

if __name__=='__main__':unittest.main()
