"""Synthetic sidecar provenance/overlay tests; no model code."""
import copy
from pathlib import Path
import types
import unittest
import complete_with_sidecar as overlay


def fixture():
    cfg=dict(d=2,m=2,seed=7)
    rows=[dict(step=i,t=float(i)) for i in range(3)]
    old=[dict(index=i,step=i,t=float(i),status='ok' if i!=1 else 'not_evaluated_runtime_budget',value=-0.0 if i==0 else float(i)) for i in range(3)]
    new=copy.deepcopy(old);new[1].update(status='ok',value=3.)
    original=dict(cfg=cfg,diagnostics=old,termination={'step':2})
    raw=dict(rows=rows,termination={'step':2});spec=dict(tag='mock',cfg=cfg,files={'pin':'same'},expected_checkpoint_count=3,expected_success_indices=[0,2])
    plan=dict(full_compute_order=[1])
    sidecar=dict(completion_pass_finished=True,completion_pass_complete=True,input_files_unchanged_after=True,new_training_updates=0,
        original_process_diagnostics_complete=False,original_result_mutated=False,independent_seed=False,quadrature_multiplier=1,
        tag='mock',cfg=cfg,input_files_before=spec['files'],input_files_after=spec['files'],plan=plan,attempted_indices=[1],
        original_training_termination=original['termination'],diagnostics=new,diagnostic_success_count=3,missing_indices=[],
        diagnostic_row_origins={'0':'original_interrupted_diagnostic_process','1':'diagnostics_only_completion_pass','2':'original_interrupted_diagnostic_process'})
    metadata=dict(shapes={'t':[3],'P':[3,3,2],'V':[3,3,2],'a':[3,2]},times=[0.,1.,2.])
    return original,raw,sidecar,spec,plan,metadata


class SidecarTests(unittest.TestCase):
    def test_exact_record_and_complete_grid_check(self):
        values=fixture();before=copy.deepcopy(values)
        receipt=overlay.verify_sidecar(*values)
        self.assertEqual(receipt['preserved_indices'],[0,2]);self.assertEqual(receipt['newly_computed_indices'],[1])
        self.assertEqual(values,before)
        self.assertTrue(receipt['original_successful_rows_exact_typed_float_bits'])

    def test_negative_zero_and_numeric_type_changes_detected(self):
        self.assertFalse(overlay.exact_values(-0.,0.))
        self.assertFalse(overlay.exact_values(1,1.))
        self.assertTrue(overlay.exact_values({'a':[1,1.,-0.]},{'a':[1,1.,-0.]}))
        for value in (0.,0):
            args=list(fixture());args[2]['diagnostics'][0]['value']=value
            with self.assertRaisesRegex(ValueError,'value changed'):overlay.verify_sidecar(*args)

    def test_incomplete_grid_metadata_partition_or_training_change_rejected(self):
        for case in ('train','pin','order','shape','time','identity','status','origin','complete'):
            args=list(fixture());sidecar=args[2]
            if case=='train':sidecar['new_training_updates']=1
            elif case=='pin':sidecar['input_files_after']={'changed':'x'}
            elif case=='order':sidecar['attempted_indices']=[2]
            elif case=='shape':args[5]['shapes']['P']=[3,2,3]
            elif case=='time':args[5]['times'][1]=1.1
            elif case=='identity':sidecar['diagnostics'][1]['step']=2
            elif case=='status':sidecar['diagnostics'][1]['status']='failed'
            elif case=='origin':sidecar['diagnostic_row_origins']['1']='original_interrupted_diagnostic_process'
            else:sidecar['completion_pass_complete']=False
            with self.assertRaises(ValueError,msg=case):overlay.verify_sidecar(*args)

    def test_overlay_changes_only_diagnostic_view_and_does_not_write(self):
        original,_,sidecar,_,_,_=fixture();original.update(completed=True,diagnostic_success_count=2)
        before=copy.deepcopy(original)
        class Reader:
            def read(self,path,kind='json',default=None):return copy.deepcopy(original)
        cls=overlay.overlay_reader_class(types.SimpleNamespace(Reader=Reader),Path('/immutable/result.json'),sidecar)
        updated=cls().read('/immutable/result.json')
        self.assertEqual(updated['diagnostics'],sidecar['diagnostics'])
        self.assertEqual({k:v for k,v in updated.items() if k!='diagnostics'},{k:v for k,v in before.items() if k!='diagnostics'})
        self.assertEqual(cls().read('/another/result.json'),original)
        self.assertEqual(original,before)
        self.assertEqual(updated['diagnostic_success_count'],2)

if __name__=='__main__':unittest.main()
