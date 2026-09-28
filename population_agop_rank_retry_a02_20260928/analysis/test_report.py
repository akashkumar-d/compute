"""Fixed-attempt record-selection tests; no scientific kernels or real outcome evaluation."""
import copy
import json
from pathlib import Path
import unittest
import report


def fixture():
    original_manifest=report.read(report.ORIGINAL_MANIFEST)
    retry_manifest=report.read(report.RETRY_MANIFEST)
    selection=report.read(report.SELECTION)
    def record(job,attempt):
        return dict(id=job['id'],config=copy.deepcopy(job['config']),execution_directory=f'/artificial/a0{attempt}',
            process_state='fixture',history=[{'artificial_loss':attempt}],checkpoints=[{'artificial_gain':100-attempt}],
            original_outcome='positive' if attempt==1 else 'negative')
    common=dict(criteria={'fixed':True},script_sha256='same',snapshot_usable=True,inputs_unchanged=True,reader_errors=[])
    original=dict(common,arms=[record(j,1) for j in original_manifest['configs']])
    retry=dict(common,arms=[record(j,2) for j in retry_manifest['configs']])
    return original,retry,original_manifest,retry_manifest,selection


class SelectionTests(unittest.TestCase):
    def test_exact20_12_rule_ignores_outcomes_and_preserves_records(self):
        values=fixture();before=copy.deepcopy(values)
        selected,mapping=report.select_records(*values)
        self.assertEqual(values,before)
        self.assertEqual(len(selected),32)
        self.assertEqual(sum(r['attempt']==1 for r in mapping),20)
        self.assertEqual(sum(r['attempt']==2 for r in mapping),12)
        old={r['id']:r for r in values[0]['arms']};new={r['id']:r for r in values[1]['arms']}
        for record,provenance in zip(selected,mapping):
            source=new[record['id']] if provenance['attempt']==2 else old[record['id']]
            self.assertEqual(record,source)
            self.assertEqual(report.record_sha(record),provenance['selected_record_sha256'])
            if provenance['attempt']==2:self.assertEqual(record['original_outcome'],'negative')
        selected[0]['history'].append('changed derived copy')
        self.assertEqual(values,before)

    def test_missing_duplicate_extra_or_swapped_ids_rejected(self):
        for case in ('missing','duplicate','extra','selection','old_missing'):
            values=list(fixture())
            if case=='missing':values[1]['arms'].pop()
            elif case=='duplicate':values[1]['arms'][1]=copy.deepcopy(values[1]['arms'][0])
            elif case=='extra':values[1]['arms'].append(dict(values[1]['arms'][0],id='unplanned'))
            elif case=='selection':
                values[4]['retry_ids'][0],values[4]['retained_completion_ids'][0]=values[4]['retained_completion_ids'][0],values[4]['retry_ids'][0]
            else:values[0]['arms'].pop()
            with self.assertRaises(ValueError,msg=case):report.select_records(*values)

    def test_criteria_configuration_source_or_integrity_changes_rejected(self):
        for case in ('criteria','config','summary_config','source','unstable','errors'):
            values=list(fixture())
            if case=='criteria':values[1]['criteria']={'changed':True}
            elif case=='config':values[3]['configs'][0]['config'][0]['args']['seed']=999
            elif case=='summary_config':values[1]['arms'][0]['config'][0]['args']['h']=.9
            elif case=='source':values[1]['script_sha256']='changed'
            elif case=='unstable':values[1]['snapshot_usable']=False
            else:values[1]['reader_errors']=['unresolved input']
            with self.assertRaises(ValueError,msg=case):report.select_records(*values)

    def test_frozen_plan_matches_selection_and_sources(self):
        plan=report.read(Path(report.__file__).parent/'PLAN.json')
        report.check_hashes(plan['pinned_inputs'])
        selection=report.read(report.SELECTION)
        self.assertEqual({r['id'] for r in plan['attempt_map'] if r['attempt']==2},set(selection['retry_ids']))
        self.assertEqual({r['id'] for r in plan['attempt_map'] if r['attempt']==1},set(selection['retained_completion_ids']))
        self.assertFalse(plan['best_attempt_selection']);self.assertEqual(plan['new_independent_seeds'],0)

if __name__=='__main__':unittest.main()
