"""Prepare a frozen four-seed confirmation; standard-library file work only."""
from pathlib import Path
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
W=HERE.parent.parent
OLD=W/'goal_followup_v10/rank_sweep/bundle'
BUNDLE=HERE/'bundle'
ORIGINAL_HASH='e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01'
SEEDS=[9351,9352,9353,9354]
RUNTIME=dict(workers=4,reserved_cpus=28,min_available_memory_gib=16,min_free_disk_gib=5,
             global_seconds=2150,per_arm_seconds=1990,diagnostic_reserve_seconds=180)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def copyfile(path,name):
    dest=BUNDLE/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest)


GATE=r'''def validate_protocol(manifest, root=ROOT):
    """Only the fixed four unused seeds and the exact development recipe."""
    expected_runtime=dict(workers=4,reserved_cpus=28,min_available_memory_gib=16,
                          min_free_disk_gib=5,global_seconds=2150,per_arm_seconds=1990,
                          diagnostic_reserve_seconds=180)
    if manifest['runtime'] != expected_runtime or manifest.get('dependency') is not None:
        raise ValueError('Runtime must equal the frozen four-seed confirmation bounds')
    original_path=relative_file(root,'reference/rank_MANIFEST.json')
    original_hash='e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01'
    if sha256(original_path) != original_hash or manifest.get('original_manifest_sha256') != original_hash:
        raise ValueError('Original development manifest changed')
    original=load_json(original_path)
    design_path=relative_file(root,'DESIGN.json')
    if sha256(design_path) != manifest['design_sha256']:
        raise ValueError('Confirmation design changed')
    design=load_json(design_path)
    audit_path=relative_file(root,'FRESH_SEED_AUDIT.json')
    if sha256(audit_path) != '__AUDIT_SHA__' or manifest.get('fresh_seed_audit_sha256') != '__AUDIT_SHA__':
        raise ValueError('Precreation seed audit changed')
    audit=load_json(audit_path)
    seeds=[9351,9352,9353,9354]
    if audit['seeds'] != seeds or audit['collisions'] or audit['read_errors']:
        raise ValueError('Fresh-seed audit does not establish the fixed unused range')
    if (manifest.get('total_arms') != 4 or manifest.get('confirmation_denominator') != 4 or
            manifest.get('independent_seed_count') != 4 or manifest.get('paired_reused_seeds') != [] or
            manifest.get('development_seeds_not_pooled') != [641,642] or
            design.get('seeds') != seeds or design.get('keep_all_outcomes') is not True):
        raise ValueError('Keep all four confirmation outcomes separate from development seeds')
    bases=[e for e in original['configs'] if e['engine']=='relu' and e['teacher']=='h2' and e['rank']==16]
    if len(bases)!=2 or sorted(e['seed'] for e in bases)!=[641,642]:
        raise ValueError('Expected the two frozen development recipe references')
    base=next(e for e in bases if e['seed']==641)
    science=lambda c:{k:v for k,v in c.items() if k not in ('id','seed')}
    if science(bases[0]['config']) != science(bases[1]['config']):
        raise ValueError('Development recipes differ beyond ID and seed')
    if len(manifest['configs'])!=4 or len(design['jobs'])!=4:
        raise ValueError('Exactly four confirmation arms are required')
    for actual,planned,seed in zip(manifest['configs'],design['jobs'],seeds):
        cid=f'v10_confirm_relu_h2_r16_d64_m256_seed{seed}'
        cfg=dict(base['config'],id=cid,seed=seed)
        if (actual['id']!=cid or actual['seed']!=seed or actual['config']!=cfg or
                actual!=planned or actual['config_path']!=f'configs/{cid}.json' or
                actual['engine']!='relu' or actual['student']!='relu' or actual['teacher']!='h2' or
                actual['rank']!=16 or actual['width']!=256 or actual['cell']!='v10_confirm_relu_h2_r16'):
            raise ValueError('Only ID and fresh seed may differ from the exact development recipe')
        path=relative_file(root,actual['config_path'])
        if sha256(path)!=actual['configuration_sha256'] or load_json(path)!=cfg:
            raise ValueError('Confirmation configuration file differs from its frozen object/hash')
        changes={k:{'before':base['config'].get(k),'after':v} for k,v in cfg.items() if base['config'].get(k)!=v}
        if set(changes)!={'id','seed'} or actual['changes_from_development']!=changes:
            raise ValueError('Confirmation changes must be exactly ID and seed')
    for name,digest in original['source_sha256'].items():
        if name.startswith(('code/','diagnostics/','swiglu/')) or name in ('TEACHERS.json','requirements.txt'):
            if manifest['source_sha256'].get(name)!=digest or sha256(relative_file(root,name))!=digest:
                raise ValueError('Scientific source changed from the development bundle')
    if manifest['criterion']!=original['criterion'] or design['criterion']!=original['criterion']:
        raise ValueError('Original 1percent/5percent same-state success criteria must remain unchanged')


'''


EXTRA_TESTS=r'''
    def test_only_id_and_seed_change_from_development(self):
        root=Path(__file__).resolve().parent.parent
        manifest=launch.load_json(root/'MANIFEST.json')
        original=launch.load_json(root/'reference/rank_MANIFEST.json')
        base=next(e['config'] for e in original['configs'] if e['id']=='v10_rank_relu_h2_r16_d64_m256_seed641')
        for entry in manifest['configs']:
            changed={k for k in entry['config'] if entry['config'][k]!=base[k]}
            self.assertEqual(changed,{'id','seed'})
        for field in ('h','steps','force_time_max','loss_stop','scale','diagnostic_stride'):
            changed=json.loads(json.dumps(manifest))
            changed['configs'][0]['config'][field]*=2
            with self.assertRaises(ValueError,msg=field):launch.validate_protocol(changed,root)

    def test_all_four_outcomes_separate_from_development(self):
        root=Path(__file__).resolve().parent.parent
        manifest=launch.load_json(root/'MANIFEST.json')
        for kind in ('drop','pool','reuse','freshcount','label'):
            changed=json.loads(json.dumps(manifest))
            if kind=='drop':changed['configs'].pop()
            elif kind=='pool':changed['confirmation_denominator']=6
            elif kind=='reuse':changed['paired_reused_seeds']=[641,642]
            elif kind=='freshcount':changed['independent_seed_count']=6
            else:changed['configs'][0]['cell']='v10_rank_relu_h2_r16'
            with self.assertRaises(ValueError,msg=kind):launch.validate_protocol(changed,root)

    def test_fresh_seed_audit_frozen_before_creation(self):
        root=Path(__file__).resolve().parent.parent
        audit=launch.load_json(root/'FRESH_SEED_AUDIT.json')
        self.assertEqual(audit['seeds'],[9351,9352,9353,9354])
        self.assertFalse(audit['collisions'])
        self.assertFalse(audit['read_errors'])
        self.assertGreater(audit['files_checked'],1000)
        manifest=launch.load_json(root/'MANIFEST.json')
        manifest['fresh_seed_audit_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'seed audit changed'):launch.validate_protocol(manifest,root)

'''


PROTOCOL='''# Fresh-seed confirmation: ReLU h2 at rank 16

This preregistered four-arm cohort tests the selected development recipe on seeds 9351–9354. A precreation scan of 1,827 existing manifest/config files found no collision or read error; its hashed receipt is frozen in `FRESH_SEED_AUDIT.json`. All four seeds are fixed before these outcomes. Record every completed, failed, pending, capped or numerically unresolved arm. Do not replace a seed after seeing its result. Development seeds 641/642 remain separate; the confirmation denominator is 4, never six fresh seeds.

The source recipe is the reviewed original rank 16 ReLU h2 configuration. Only scientific configuration ID and seed change: d = 64, width 256, raw Gaussian scale 0.01, h = 0.5 Armijo population GD, 20,000-step cap, force-time horizon 1500, raw loss stop 0.01, profiled intercept, stride 25 and diagnostic priority fractions 0.25/0.5/0.75 all remain unchanged. h2 is the raw normalized Hermite teacher with E[Y²]=Var(Y)=1; no teacher or normalization change is introduced. The source retains l2<=32 and l1<=64sqrt(16)=256 refit budgets, original stable AGOP calculations, numerical tolerances and analytic teacher/student kernels.

Apply both original initial whole-update max/min loss windows 1.01 and 1.05. At the same saved state require minimum AGOP alignment gain>=0.5 and numerical refit-improvement lower difference>=0.1Var(Y), then a subsequent raw loss decrease>=0.1Var(Y), with original initial/current numerical screens. Use the canonical summary criterion, not the native ReLU legacy raw-refit>1e-5 flag. No threshold, starting reference, window, readout class, checkpoint grid or plotting normalization is changed after outcomes. This is confirmation of a selected recipe, not an architecture-wide claim or a formal population certificate. Preserve numerical flags and incomplete observations.

The isolated bundle has four SERVER-only single-thread workers plus 28 reserved CPU slots. Admission: at least 32 effective CPUs, 16 GiB available memory, 5 GiB free disk, verified inherited nice>=10. Per arm: 1990 s = 1800 s soft training + 180 s soft diagnostics + 10 s cleanup; global 2150 s. Existing scheduler safeguards, bounded TERM/KILL/reaping, source revalidation, refused existing outputs and persistent restart journal remain in force. Soft phases may be shortened by evaluation/runtime overhead. Reservations do not inspect other bundles: the parent checks at most 26 other task-owned workers immediately before dispatch so combined usage stays <= 30 with 2 CPUs free. The 13 reported current workers do not waive that fresh check.

No automatic dispatch is configured. The parent independently reviews this exact manifest, verifies live capacity, publishes the new shared leaf and manually dispatches. `REVIEW.json` is pending. No model evaluation, training, publication, network call or manuscript change was performed to prepare this design.

Use a distinct publication leaf `population_agop_relu_rank16_confirmation_20260928`, remote job `agop-relu-r16-confirm-20260928-a01`, and output `execution_confirmation_a01`. Preserve prior development and retry artifacts. The exact commands and interpretation requirements are in `README.md`.
'''


def main():
    if BUNDLE.exists():raise FileExistsError('Preserve existing confirmation bundle')
    assert sha(OLD/'MANIFEST.json')==ORIGINAL_HASH
    old=load(OLD/'MANIFEST.json');audit=load(HERE/'FRESH_SEED_AUDIT.json')
    assert audit['seeds']==SEEDS and not audit['collisions'] and not audit['read_errors']
    assert all(sha(OLD/p)==h for p,h in old['source_sha256'].items())
    bases=[e for e in old['configs'] if e['engine']=='relu' and e['teacher']=='h2' and e['rank']==16]
    base=next(e for e in bases if e['seed']==641)
    assert {k:v for k,v in bases[0]['config'].items() if k not in ('id','seed')}=={k:v for k,v in bases[1]['config'].items() if k not in ('id','seed')}
    for name in old['source_sha256']:
        if name.startswith(('code/','diagnostics/','swiglu/')) or name in ('TEACHERS.json','requirements.txt','SOURCE_REVIEW.json'):
            copyfile(OLD/name,name)
    for source,name in [(OLD/'MANIFEST.json','rank_MANIFEST.json'),(OLD/'REVIEW.json','rank_REVIEW.json'),
                        (OLD/'launcher/launch.py','rank_launch.py'),(OLD/'launcher/test_launch.py','rank_test_launch.py'),
                        (OLD/'SOURCE_LINEAGE.json','rank_SOURCE_LINEAGE.json')]:copyfile(source,'reference/'+name)
    for e in bases:copyfile(OLD/e['config_path'],'reference/'+Path(e['config_path']).name)
    copyfile(HERE/'FRESH_SEED_AUDIT.json','FRESH_SEED_AUDIT.json')
    configs=[]
    for seed in SEEDS:
        cid=f'v10_confirm_relu_h2_r16_d64_m256_seed{seed}'
        cfg=dict(base['config'],id=cid,seed=seed)
        path=f'configs/{cid}.json';save(BUNDLE/path,cfg)
        configs.append(dict(id=cid,tag=cid,cell='v10_confirm_relu_h2_r16',engine='relu',student='relu',teacher='h2',
                            rank=16,seed=seed,width=256,supplemental=False,recipe='exact_development_rank16_h2_fresh_confirmation',
                            parent_run_id=base['id'],development_reference_ids=[e['id'] for e in bases],
                            config_path=path,configuration_sha256=sha(BUNDLE/path),config=cfg,
                            changes_from_development={k:dict(before=base['config'][k],after=cfg[k]) for k in ('id','seed')},
                            target_variance=1.,normalized_loss_stop=.01,seed_status='unused_manifest_config_seed_at_preregistration'))
    design=dict(schema='relu_rank16_h2_four_fresh_seeds_v1',status='preregistered_unlaunched',seeds=SEEDS,
                keep_all_outcomes=True,confirmation_denominator=4,development_seeds_separate=[641,642],
                original_manifest_sha256=ORIGINAL_HASH,fresh_seed_audit_sha256=sha(BUNDLE/'FRESH_SEED_AUDIT.json'),
                runtime=RUNTIME,criterion=old['criterion'],jobs=configs)
    save(BUNDLE/'DESIGN.json',design)
    (BUNDLE/'PROTOCOL.md').write_text(PROTOCOL)
    script=(OLD/'launcher/launch.py').read_text()
    script=script.replace('Bounded v10 rank-sweep SERVER-only wrapper','Bounded fresh-seed ReLU rank16 confirmation SERVER-only wrapper')
    script=script.replace('not 1 <= workers <= 28','not 1 <= workers <= 4').replace('integer from 1 to 28','integer from 1 to 4')
    script=script.replace('total > 3600','total > 2150').replace('global <= 3600 seconds','global <= 2150 seconds')
    script=script.replace('Rank sweep has no local execution mode.','Fresh-seed confirmation has no local execution mode.')
    start,end=script.index('def validate_protocol('),script.index('def require_review(')
    script=script[:start]+GATE.replace('__AUDIT_SHA__',sha(BUNDLE/'FRESH_SEED_AUDIT.json'))+script[end:]
    script=script.replace("'.rank_sweep.lock'","'.relu_rank16_confirmation.lock'")
    script=script.replace('default="execution",','default="execution_confirmation_a01",')
    (BUNDLE/'launcher').mkdir();(BUNDLE/'launcher/launch.py').write_text(script)
    tests=(OLD/'launcher/test_launch.py').read_text()
    tests=tests.replace('test_28_rank_workers_leave_four_cpus_available','test_four_confirmation_workers_reserve_twenty_eight_cpus')
    tests=tests.replace('workers=28,reserved_cpus=4,min_available_memory_gib=24,min_free_disk_gib=10','workers=4,reserved_cpus=28,min_available_memory_gib=16,min_free_disk_gib=5')
    tests=tests.replace("['workers'],28)","['workers'],4)").replace('(4,28,30,31.9)','(4,16,30,31.9)')
    tests=tests.replace('available_memory_gib=23.9','available_memory_gib=15.9').replace('free_disk_gib=9.9','free_disk_gib=4.9')
    tests=tests.replace('["global_seconds"] = 3601','["global_seconds"] = 2151').replace('global <= 3600','global <= 2150')
    tests=tests.replace('["global_seconds"] = 3600','["global_seconds"] = 2150').replace('["workers"] = 29','["workers"] = 5').replace('from 1 to 28','from 1 to 4')
    tests=tests.replace('test_rank_queue_32_arms_never_exceeds_28_workers','test_four_confirmation_arms_share_one_bounded_wave')
    tests=tests.replace("['workers'] = 28","['workers'] = 4").replace('range(1,32)','range(1,4)')
    tests=tests.replace("len(status['completed']),32","len(status['completed']),4").replace('started.count(1000.),28','started.count(1000.),4')
    tests=tests.replace('self.assertEqual(started.count(1005.),4)','self.assertEqual(len(set(started)),1)').replace('clock.now,1010.','clock.now,1005.')
    tests=tests.replace('class V10GatesTests','class ConfirmationGatesTests').replace('test_frozen_rank_design_criteria_order_and_runtime','test_frozen_confirmation_design_criteria_order_and_runtime')
    tests=tests.replace("len(manifest['configs']),32","len(manifest['configs']),4").replace("['global_seconds']=4200","['global_seconds']=2151")
    tests=tests.replace("entry = dict(engine='swiglu',config_path='configs/a.json',id='a')","entry = dict(engine='relu',config_path='configs/a.json',id='a')")
    tests=tests.replace("self.assertEqual(cmd[cmd.index('--wall-seconds')+1],'1800.0')","self.assertEqual(cmd[cmd.index('--max-seconds')+1],'1980.0')")
    tests=tests.replace("self.assertEqual(cmd[cmd.index('--diagnostic-seconds')+1],'180.0')","self.assertEqual(cmd[cmd.index('--diagnostic-reserve')+1],'180.0')")
    tests=tests.replace('with self.assertRaises(FileExistsError):\n            launch.main(["--dry-run"], self.root)',
                        'with self.assertRaises(FileExistsError):\n            launch.main(["--dry-run", "--execution-dir", "execution"], self.root)')
    insert=tests.index('\n\nif __name__ == "__main__":');tests=tests[:insert]+EXTRA_TESTS+tests[insert:]
    (BUNDLE/'launcher/test_launch.py').write_text(tests)
    (BUNDLE/'LAUNCHER.diff').write_text(''.join(difflib.unified_diff((OLD/'launcher/launch.py').read_text().splitlines(True),script.splitlines(True),fromfile='reference/rank_launch.py',tofile='launcher/launch.py')))
    (BUNDLE/'README.md').write_text(PROTOCOL+'''\nStatic checks (no scientific import or execution):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_confirmation_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

After independent exact-manifest review, source publication and a fresh resource check, the parent uses its verified server interpreter:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_confirmation_a01
```

The bundle lock protects this bundle only. Do not bypass it by copying an active bundle, invoking the scientific runner directly, clearing the journal or reusing an output directory. Preserve raw loss histories, all snapshots, logs and incomplete diagnostics. Process exit 0 is not scientific success. Analysis must keep the four fresh-seed outcomes together and the two development outcomes separately labeled.
''')
    hashes={str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
    manifest=dict(protocol_version='v10_relu_rank16_h2_fresh_confirmation',created_utc=datetime.now(timezone.utc).isoformat(),
                  created_before_outcomes=True,phase='four_fresh_seed_confirmation_of_selected_development_recipe',
                  total_arms=4,confirmation_denominator=4,teacher_student_cells=1,canonical_teachers=['h2'],
                  paired_reused_seeds=[],independent_seed_count=4,development_seeds_not_pooled=[641,642],
                  ranks=[16],dimension=64,runtime=RUNTIME,dependency=None,original_manifest_sha256=ORIGINAL_HASH,
                  design_sha256=sha(BUNDLE/'DESIGN.json'),fresh_seed_audit_sha256=sha(BUNDLE/'FRESH_SEED_AUDIT.json'),
                  criterion=old['criterion'],configs=configs,arms=[{k:v for k,v in e.items() if k!='config'} for e in configs],
                  source_sha256=hashes,pinned_sources=hashes.copy(),queue_order='Fixed9351,9352,9353,9354; one four-worker wave; retain all outcomes',
                  publication_leaf='population_agop_relu_rank16_confirmation_20260928',remote_run_name='agop-relu-r16-confirm-20260928-a01',
                  default_execution_directory='execution_confirmation_a01',analysis_note='Four fresh outcomes only; development641/642 separate. No six-fresh-seed pooling, threshold changes or seed replacement.')
    save(BUNDLE/'MANIFEST.json',manifest)
    save(BUNDLE/'REVIEW.json',dict(status='pending_independent_runtime_review',execution_launched=False,
         dispatch_prerequisites=['Independent exact source/config/design/runtime review','Fresh live-capacity check and aggregate<=30 task workers','Parent shared-code publication and manual dispatch']))
    assert all(sha(OLD/p)==h for p,h in old['source_sha256'].items())
    print(json.dumps(dict(manifest_sha256=sha(BUNDLE/'MANIFEST.json'),frozen_files=len(hashes),config_count=4,seeds=SEEDS,
                         source_files_unchanged=True,review_pending=True)))


if __name__=='__main__':main()
