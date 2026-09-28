"""Assemble a retry-only artifact from frozen local files; never import models."""
from pathlib import Path
import ast
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
W = HERE.parent.parent
OLD = W / 'goal_followup_v10/rank_sweep/bundle'
INVENTORY = OLD.parent / 'REMOTE_INVENTORY_AFTER_STOP.json'
BUNDLE = HERE / 'bundle'
ORIGINAL_HASH = 'e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01'
INVENTORY_HASH = '1307b4d1c51cbb757e765db10bb29d366a8a20b03fc3a7a5df40c29e456177eb'
RUNTIME = dict(workers=12, reserved_cpus=20, min_available_memory_gib=32,
               min_free_disk_gib=10, global_seconds=2150, per_arm_seconds=1990,
               diagnostic_reserve_seconds=180)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy_file(source, name):
    target = BUNDLE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


PROTOCOL_FUNCTION = r'''def validate_protocol(manifest, root=ROOT):
    """Admit exactly the unfinished twelve original arms, from fixed evidence."""
    expected_runtime = dict(workers=12, reserved_cpus=20, min_available_memory_gib=32,
                            min_free_disk_gib=10, global_seconds=2150, per_arm_seconds=1990,
                            diagnostic_reserve_seconds=180)
    if manifest['runtime'] != expected_runtime or manifest.get('dependency') is not None:
        raise ValueError('Runtime must equal the frozen twelve-arm retry bounds')
    original_path = relative_file(root, 'reference/rank_MANIFEST.json')
    inventory_path = relative_file(root, 'reference/REMOTE_INVENTORY_AFTER_STOP.json')
    original_hash = 'e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01'
    inventory_hash = '1307b4d1c51cbb757e765db10bb29d366a8a20b03fc3a7a5df40c29e456177eb'
    if sha256(original_path) != original_hash or sha256(inventory_path) != inventory_hash:
        raise ValueError('Original manifest or stopped-run inventory changed')
    if (manifest.get('original_manifest_sha256') != original_hash or
            manifest.get('stop_inventory_sha256') != inventory_hash):
        raise ValueError('Retry reference hash declaration changed')
    original, inventory = load_json(original_path), load_json(inventory_path)
    if inventory['files']['MANIFEST.snapshot.json']['sha256'] != original_hash:
        raise ValueError('Inventory belongs to a different original manifest')
    design = load_json(root / 'DESIGN.snapshot.json')
    if (sha256(root / 'DESIGN.snapshot.json') != original['design_sha256'] or
            manifest['design_sha256'] != original['design_sha256']):
        raise ValueError('Original scientific design changed')
    if (len(original['configs']) != 32 or original['total_arms'] != 32 or
            manifest.get('original_study_total_arms') != 32 or manifest.get('study_denominator') != 32 or
            manifest.get('total_arms') != 12 or manifest.get('attempt_number') != 2 or
            manifest.get('new_independent_seeds') != 0):
        raise ValueError('Keep twelve attempt-two retries inside the original thirty-two-arm study')
    completed, missing = [], []
    for entry in original['configs']:
        indicator = 'DONE.json' if entry['engine'] == 'relu' else 'result.json'
        marker = f"data/{entry['id']}/{indicator}"
        if marker in inventory['files']:
            record = inventory['files'][marker]
            if (record['size'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
                raise ValueError('Invalid completion marker in frozen inventory')
            completed.append(entry)
        else:
            missing.append(entry)
    expected_cells = {(r, teacher, seed) for r in (16,8,4)
                      for teacher in ('h2','relu') for seed in (641,642)}
    if (len(completed) != 20 or len(missing) != 12 or
            sum(e['engine'] == 'relu' for e in completed) != 16 or
            {(e['rank'],e['teacher'],e['seed']) for e in completed if e['engine']=='swiglu'} !=
              {(2,t,s) for t in ('h2','relu') for s in (641,642)} or
            any(e['engine'] != 'swiglu' for e in missing) or
            {(e['rank'],e['teacher'],e['seed']) for e in missing} != expected_cells):
        raise ValueError('Frozen inventory does not identify exactly the twelve unfinished SwiGLU arms')
    if len(manifest['configs']) != len(missing):
        raise ValueError('Retry set must contain exactly twelve configs')
    selection = load_json(root / 'RETRY_SELECTION.json')
    if (selection['retry_ids'] != [e['id'] for e in missing] or
            selection['retained_completion_ids'] != [e['id'] for e in completed] or
            selection['original_manifest_sha256'] != original_hash or
            selection['stop_inventory_sha256'] != inventory_hash or
            selection['study_denominator'] != 32):
        raise ValueError('Retry selection record differs from frozen completion evidence')
    provenance = dict(attempt_number=2, restart_mode='from_same_initialization',
                      original_study_total_arms=32, original_manifest_sha256=original_hash,
                      stop_inventory_sha256=inventory_hash,
                      original_execution='agop-ranks-2to16-20260928-a01/execution_remote_a01')
    for actual, prior in zip(manifest['configs'], missing):
        expected = dict(prior, retry_provenance=dict(provenance, original_id=prior['id']))
        if actual != expected:
            raise ValueError('Retry identity, config, order or attempt provenance differs from original')
        path = relative_file(root, actual['config_path'])
        original_config_hash = original['source_sha256'][actual['config_path']]
        if (sha256(path) != original_config_hash or actual['configuration_sha256'] != original_config_hash or
                load_json(path) != prior['config']):
            raise ValueError('Retry config must remain byte-identical to the original')
    for name, digest in original['source_sha256'].items():
        if name.startswith(('code/','diagnostics/','swiglu/')) or name in ('TEACHERS.json','requirements.txt'):
            if (manifest['source_sha256'].get(name) != digest or
                    sha256(relative_file(root,name)) != digest):
                raise ValueError('Scientific source differs from the original rank bundle')
    if manifest['criterion'] != original['criterion'] or manifest['criterion'] != design['criterion']:
        raise ValueError('Success criteria must remain the original study criteria')


'''


ADDITIONAL_TESTS = r'''
    def test_completed_arms_cannot_be_reintroduced(self):
        root=Path(__file__).resolve().parent.parent
        manifest=launch.load_json(root/'MANIFEST.json')
        original=launch.load_json(root/'reference/rank_MANIFEST.json')
        for kind in ('relu','rank2'):
            changed=json.loads(json.dumps(manifest))
            completed=next(e for e in original['configs'] if
                (e['engine']=='relu' if kind=='relu' else e['engine']=='swiglu' and e['rank']==2))
            changed['configs'][0]=completed
            with self.assertRaisesRegex(ValueError,'Retry identity'):
                launch.validate_protocol(changed,root)

    def test_inventory_and_original_manifest_hash_gates(self):
        source=Path(__file__).resolve().parent.parent
        for filename in ('rank_MANIFEST.json','REMOTE_INVENTORY_AFTER_STOP.json'):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
                root=Path(td);(root/'reference').mkdir()
                for name in ('rank_MANIFEST.json','REMOTE_INVENTORY_AFTER_STOP.json'):
                    (root/'reference'/name).write_bytes((source/'reference'/name).read_bytes())
                with (root/'reference'/filename).open('a') as stream:stream.write('\n')
                with self.assertRaisesRegex(ValueError,'inventory changed'):
                    launch.validate_protocol(launch.load_json(source/'MANIFEST.json'),root)

    def test_retry_denominator_and_attempt_provenance(self):
        root=Path(__file__).resolve().parent.parent
        manifest=launch.load_json(root/'MANIFEST.json')
        for kind in ('denominator','attempt','newseed','id','provenance'):
            changed=json.loads(json.dumps(manifest))
            if kind=='denominator':changed['study_denominator']=12
            elif kind=='attempt':changed['attempt_number']=1
            elif kind=='newseed':changed['new_independent_seeds']=2
            elif kind=='id':changed['configs'][0]['id']='new_seed_disguised_as_retry'
            else:changed['configs'][0]['retry_provenance']['original_id']='other'
            with self.assertRaises(ValueError,msg=kind):launch.validate_protocol(changed,root)

    def test_inventory_markers_override_stale_scheduler_count(self):
        root=Path(__file__).resolve().parent.parent
        selection=launch.load_json(root/'RETRY_SELECTION.json')
        self.assertEqual(len(selection['retained_completion_ids']),20)
        self.assertEqual(len(selection['retry_ids']),12)
        self.assertEqual(selection['study_denominator'],32)
        self.assertFalse(set(selection['retained_completion_ids']) & set(selection['retry_ids']))
        self.assertTrue(all('_swiglu_' in x and '_r2_' not in x for x in selection['retry_ids']))

'''


README = '''# Rank sweep attempt 2: unfinished-arm retry only

Prepared at the user's explicit request to rerun stopped jobs. This bundle reruns exactly twelve unfinished SwiGLU arms from their unchanged initialization: ranks16/8/4 × h2/raw ReLU × seeds641/642. It retains original IDs and config bytes. Each manifest entry adds attempt2 provenance; no seed, precision, horizon, update cap, teacher, loss stop, or success criterion changes. The original32-arm study remains the denominator. These retries add zero independent seeds.

The frozen remote file inventory, not the stale scheduler count, identifies20 completed markers: all16 ReLU DONE.json files and all4 rank2 SwiGLU result.json files. Those20 arms are excluded from execution. A marker denotes a completed artifact for retry selection, not scientific success. All old interrupted/partial artifacts remain in the original attempt. No partial local download was used to select jobs. Reference manifest and inventory hashes are enforced by the launcher, as are the exact original config bytes and scientific sources.

Runtime: Linux SERVER only;12 single-thread workers,20 reserved slots;32 effective CPUs,32GiB available memory,10GiB free disk. Per arm1990s=1800s soft training+180s soft diagnostics+10s cleanup, global2150s. The existing priority, deadline, TERM/KILL/reaping, source revalidation and persistent restart-journal rules are retained. A current evaluation may overrun a soft budget; pending, capped and incomplete-diagnostic outcomes remain explicit. The reservation does not inspect other jobs: the parent must verify that the combined task-owned worker count stays<=30, leaving2CPUs free. No GPU use, local training, auto-dispatch, network call or machine change is part of preparation.

Use a separate shared-repository leaf `population_agop_rank_retry_a02_20260928`, a new remote job `agop-ranks-retry-20260928-a02`, and the new output `execution_remote_a02`; the parent owns publication and dispatch. Do not write into or resume the old execution directory, and do not rename the original scientific IDs. `MANIFEST.snapshot.json` plus `PROVENANCE.json` links all original-ID result/status records to this attempt2 bundle. Preserve both attempts; do not count retries as new study arms or splice their clocks into one trajectory.

`REVIEW.json` is deliberately pending. The parent independently reviews this exact manifest, verifies old jobs are stopped and current capacity is sufficient, then records approval and publishes before manually dispatching. The previous rank bundle's review is historical lineage, not current approval.

Static and mocked checks, which never import scientific runners:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_remote_a02
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

After review and staging, the parent invokes its verified server interpreter on:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_remote_a02
```

This is a bounded attempt to complete interrupted observations. Common order32/96/48/24 remains provisional; candidate update/AGOP/refit precision checks and trajectory sensitivity review remain required. Apply the original canonical same-state .1Var criterion and initial1%/5% windows, retain all32 original IDs and both attempt records, and report which attempt supplies each displayed trajectory. Keep old unresolved observations and do not convert process completion into scientific success.
'''


def main():
    if BUNDLE.exists():
        raise FileExistsError('Preserve existing retry bundle; inspect explicitly instead of rebuilding')
    assert sha(OLD/'MANIFEST.json') == ORIGINAL_HASH
    assert sha(INVENTORY) == INVENTORY_HASH
    original, inventory = load(OLD/'MANIFEST.json'), load(INVENTORY)
    assert all(sha(OLD/p)==h for p,h in original['source_sha256'].items())
    assert inventory['files']['MANIFEST.snapshot.json']['sha256'] == ORIGINAL_HASH
    completed, missing, markers = [], [], []
    for entry in original['configs']:
        marker = f"data/{entry['id']}/" + ('DONE.json' if entry['engine']=='relu' else 'result.json')
        if marker in inventory['files']:
            completed.append(entry)
            markers.append(dict(original_id=entry['id'], marker=marker, **inventory['files'][marker]))
        else:
            missing.append(entry)
    assert len(completed)==20 and len(missing)==12
    assert sum(e['engine']=='relu' for e in completed)==16
    assert all(e['engine']=='swiglu' and e['rank'] in (4,8,16) for e in missing)
    selected_paths = {e['config_path'] for e in missing}
    for name in original['source_sha256']:
        if (name.startswith(('code/','diagnostics/','swiglu/')) or name in
                ('TEACHERS.json','requirements.txt','SOURCE_REVIEW.json','DESIGN.snapshot.json','PROTOCOL.snapshot.md') or
                name in selected_paths):
            copy_file(OLD/name,name)
    for origin,name in [(OLD/'MANIFEST.json','rank_MANIFEST.json'),(INVENTORY,'REMOTE_INVENTORY_AFTER_STOP.json'),
                         (OLD/'REVIEW.json','rank_REVIEW.json'),(OLD/'launcher/launch.py','rank_launch.py'),
                         (OLD/'launcher/test_launch.py','rank_test_launch.py'),(OLD/'SOURCE_LINEAGE.json','rank_SOURCE_LINEAGE.json')]:
        copy_file(origin,'reference/'+name)
    script=(OLD/'launcher/launch.py').read_text()
    script=script.replace('Bounded v10 rank-sweep SERVER-only wrapper','Bounded v10 rank retry attempt2 SERVER-only wrapper')
    script=script.replace('not 1 <= workers <= 28','not 1 <= workers <= 12').replace('integer from 1 to 28','integer from 1 to 12')
    script=script.replace('total > 3600','total > 2150').replace('global <= 3600 seconds','global <= 2150 seconds')
    script=script.replace('Rank sweep has no local execution mode.','Rank retry has no local execution mode.')
    start,end=script.index('def validate_protocol('),script.index('def require_review(')
    script=script[:start]+PROTOCOL_FUNCTION+script[end:]
    script=script.replace("'.rank_sweep.lock'","'.rank_retry_a02.lock'")
    script=script.replace('default="execution",','default="execution_remote_a02",')
    (BUNDLE/'launcher').mkdir()
    (BUNDLE/'launcher/launch.py').write_text(script)
    tests=(OLD/'launcher/test_launch.py').read_text()
    tests=tests.replace('test_28_rank_workers_leave_four_cpus_available','test_12_retry_workers_reserve_twenty_cpus')
    tests=tests.replace('workers=28,reserved_cpus=4,min_available_memory_gib=24','workers=12,reserved_cpus=20,min_available_memory_gib=32')
    tests=tests.replace('available_memory_gib=30,free_disk_gib=100','available_memory_gib=40,free_disk_gib=100')
    tests=tests.replace("['workers'],28)","['workers'],12)").replace('(4,28,30,31.9)','(4,12,30,31.9)').replace('available_memory_gib=23.9','available_memory_gib=31.9')
    tests=tests.replace('["global_seconds"] = 3601','["global_seconds"] = 2151').replace('global <= 3600','global <= 2150')
    tests=tests.replace('["global_seconds"] = 3600','["global_seconds"] = 2150')
    tests=tests.replace('["workers"] = 29','["workers"] = 13').replace('from 1 to 28','from 1 to 12')
    tests=tests.replace('test_rank_queue_32_arms_never_exceeds_28_workers','test_twelve_retry_arms_share_one_bounded_wave')
    tests=tests.replace("['workers'] = 28","['workers'] = 12").replace('range(1,32)','range(1,12)')
    tests=tests.replace("len(status['completed']),32","len(status['completed']),12").replace('started.count(1000.),28','started.count(1000.),12')
    tests=tests.replace('self.assertEqual(started.count(1005.),4)','self.assertEqual(len(set(started)),1)').replace('clock.now,1010.','clock.now,1005.')
    tests=tests.replace('class V10GatesTests','class RetryGatesTests').replace('test_frozen_rank_design_criteria_order_and_runtime','test_frozen_retry_subset_criteria_order_and_runtime')
    tests=tests.replace("len(manifest['configs']),32","len(manifest['configs']),12")
    tests=tests.replace("['global_seconds']=4200","['global_seconds']=2151")
    tests=tests.replace('with self.assertRaises(FileExistsError):\n            launch.main(["--dry-run"], self.root)',
                        'with self.assertRaises(FileExistsError):\n            launch.main(["--dry-run", "--execution-dir", "execution"], self.root)')
    insert=tests.index('\n\nif __name__ == "__main__":')
    tests=tests[:insert]+ADDITIONAL_TESTS+tests[insert:]
    (BUNDLE/'launcher/test_launch.py').write_text(tests)
    (BUNDLE/'LAUNCHER.diff').write_text(''.join(difflib.unified_diff((OLD/'launcher/launch.py').read_text().splitlines(True),script.splitlines(True),fromfile='reference/rank_launch.py',tofile='launcher/launch.py')))
    (BUNDLE/'README.md').write_text(README)
    selection=dict(schema='rank_retry_selection_v1', original_manifest_sha256=ORIGINAL_HASH,
                   stop_inventory_sha256=INVENTORY_HASH, original_inventory_files=inventory['count'],
                   study_denominator=32, attempt_number=2, new_independent_seeds=0,
                   retry_ids=[e['id'] for e in missing], retained_completion_ids=[e['id'] for e in completed],
                   retained_completion_markers=markers,
                   inclusion_rule='Absence of the engine completion indicator in the frozen remote file inventory; not stale scheduler status or scientific outcome.',
                   original_artifacts='Preserved under rank_sweep and the original a01 remote job; no old output is modified.')
    write_json(BUNDLE/'RETRY_SELECTION.json',selection)
    provenance=dict(attempt_number=2,restart_mode='from_same_initialization',original_study_total_arms=32,
                    original_manifest_sha256=ORIGINAL_HASH,stop_inventory_sha256=INVENTORY_HASH,
                    original_execution='agop-ranks-2to16-20260928-a01/execution_remote_a01')
    entries=[dict(copy.deepcopy(e),retry_provenance=dict(provenance,original_id=e['id'])) for e in missing]
    manifest=dict(protocol_version='v10_rank_retry_attempt2',created_utc=datetime.now(timezone.utc).isoformat(),
                  created_before_outcomes=False,phase='retry_interrupted_original_arms_not_new_experiments',
                  total_arms=12,original_study_total_arms=32,study_denominator=32,retained_completion_arms=20,
                  attempt_number=2,new_independent_seeds=0,paired_reused_seeds=[641,642],
                  independent_seed_count=2,ranks=[4,8,16],dimension=64,canonical_teachers=['h2','relu'],
                  runtime=RUNTIME,dependency=None,design_sha256=original['design_sha256'],
                  original_manifest_sha256=ORIGINAL_HASH,stop_inventory_sha256=INVENTORY_HASH,
                  criterion=original['criterion'],configs=entries,
                  arms=[{k:v for k,v in e.items() if k!='config'} for e in entries],
                  queue_order='Original relative order, excluding every completed marker: descending16,8,4; seed641/642; h2/rawReLU; all12 one wave.',
                  publication_leaf='population_agop_rank_retry_a02_20260928',remote_run_name='agop-ranks-retry-20260928-a02',
                  default_execution_directory='execution_remote_a02',
                  resource_note='12 single-thread workers+20reserved on32CPU; external capacity checked by parent;1990s arm/2150s global.',
                  analysis_note='Preserve original32-arm denominator and interrupted attempt1. Same IDs/configs/seeds, attempt2 provenance. No extra independent observations or clock splicing.')
    hashes={str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
    manifest['source_sha256']=hashes
    manifest['pinned_sources']=hashes.copy()
    write_json(BUNDLE/'MANIFEST.json',manifest)
    write_json(BUNDLE/'REVIEW.json',dict(status='pending_independent_runtime_review',execution_launched=False,
               requested_attempt_number=2,dispatch_prerequisites=['Independent review of exact source/config/subset/runtime manifest',
               'Parent verifies old jobs stopped and current combined worker/capacity admission',
               'Parent publishes shared code then manually dispatches the new leaf/job/output']))
    for p,digest in original['source_sha256'].items():assert sha(OLD/p)==digest
    assert sha(INVENTORY)==INVENTORY_HASH
    print(json.dumps(dict(status='assembled_pending_static_mock_checks',manifest_sha256=sha(BUNDLE/'MANIFEST.json'),
                         frozen_files=len(hashes),config_count=len(entries),completion_markers_retained=20,
                         scientific_python_files=sum(1 for p in hashes if p.endswith('.py') and p.startswith(('code/','diagnostics/','swiglu/'))),
                         tests_defined=sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(ast.parse(tests))))))


if __name__ == '__main__':
    main()
