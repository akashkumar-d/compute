"""Assemble two config-only follow-ups; stdlib reads/writes, no model imports."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import math
import shutil
import struct
import zipfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
V10 = HERE.parent
RANK = V10 / 'rank_sweep/bundle'
RETRY = V10 / 'rank_retry_a02'
SCHEDULER = V10 / 'headscale/bundle'
BUNDLE = HERE / 'bundle'
ORIGINAL_HASH = 'e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01'
INVENTORY_HASH = 'b61f4512d77005467d7100a91bbd59af5c1e3b841b22a608a9c5752937643f47'
RUNTIME = dict(workers=2, reserved_cpus=30, min_available_memory_gib=12,
               min_free_disk_gib=5, global_seconds=2150, per_arm_seconds=1990,
               diagnostic_reserve_seconds=180)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def cp(source, relative):
    dest = BUNDLE / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def first_array_bytes(path):
    """Hash saved first planes without NumPy, generating states, or model code."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for key in ('t', 'P', 'V', 'a'):
            with archive.open(key + '.npy') as stream:
                assert stream.read(6) == b'\x93NUMPY'
                version = stream.read(2)
                n = struct.unpack('<H' if version[0] == 1 else '<I', stream.read(2 if version[0] == 1 else 4))[0]
                hdr = ast.literal_eval(stream.read(n).decode('latin1'))
                assert hdr['descr'] == '<f8' and hdr['fortran_order'] is False
                size = 8 * math.prod(hdr['shape'][1:])
                data = stream.read(size)
                assert len(data) == size
                if key == 't': assert struct.unpack('<d', data)[0] == 0.0
                out[key] = dict(dtype=hdr['descr'], shape=list(hdr['shape'][1:]),
                                raw_c_order_bytes=size, sha256=hashlib.sha256(data).hexdigest())
    return out


GATE = '''def validate_protocol(manifest, root=ROOT):
    """Only two paired head-rate changes from the exact completed controls."""
    expected = dict(workers=2, reserved_cpus=30, min_available_memory_gib=12,
                    min_free_disk_gib=5, global_seconds=2150, per_arm_seconds=1990,
                    diagnostic_reserve_seconds=180)
    if manifest['runtime'] != expected or manifest.get('dependency') is not None:
        raise ValueError('Runtime must equal the frozen two-arm head-rate bounds')
    original_path = relative_file(root, 'reference/rank_MANIFEST.json')
    if sha256(original_path) != '__ORIGINAL_HASH__' or manifest['original_manifest_sha256'] != '__ORIGINAL_HASH__':
        raise ValueError('Original rank control manifest changed')
    original = load_json(original_path)
    if sha256(relative_file(root, 'DESIGN.json')) != manifest['design_sha256']:
        raise ValueError('Design hash changed')
    design = load_json(root / 'DESIGN.json')
    controls = load_json(root / 'CONTROL_PROVENANCE.json')
    if (controls['inventory_sha256'] != '__INVENTORY_HASH__' or
            sha256(relative_file(root, 'reference/retry_FINAL_INVENTORY.json')) != '__INVENTORY_HASH__'):
        raise ValueError('Completed control inventory changed')
    if (len(manifest['configs']) != 2 or len(design['arms']) != 2 or
            manifest['paired_reused_seeds'] != [641,642] or manifest['fresh_confirmation'] is not False or
            manifest['retain_all_outcomes'] is not True):
        raise ValueError('Retain both reused seeds; this is not fresh confirmation')
    for entry, planned, seed in zip(manifest['configs'], design['arms'], (641,642)):
        cid = f'v10_swiglu_relu_headlr001_r8_d64_m64_seed{seed}'
        control_id = f'v10_rank_swiglu_relu_r8_d64_m64_seed{seed}'
        base = next(e for e in original['configs'] if e['id'] == control_id)
        cfg = json.loads(json.dumps(base['config']))
        cfg[0]['tag'] = cid
        cfg[0]['args']['head_lr'] = 0.001
        if (entry != planned or entry['id'] != cid or entry['seed'] != seed or
                entry['engine'] != 'swiglu' or entry['student'] != 'swiglu' or
                entry['teacher'] != 'relu' or entry['rank'] != 8 or entry['width'] != 64 or
                entry['cell'] != base['cell'] or entry['control_id'] != control_id or entry['config'] != cfg):
            raise ValueError('Only tag and head_lr may change from the original paired control')
        path = relative_file(root, entry['config_path'])
        if load_json(path) != cfg or sha256(path) != entry['configuration_sha256']:
            raise ValueError('Config file must match frozen paired recipe')
        ref = controls['controls'][str(seed)]
        result_path = relative_file(root, ref['copied_result_path'])
        result = load_json(result_path)
        if (sha256(result_path) != ref['files']['result.json']['sha256'] or
                result['completed'] is not True or result['config_sha256'] != base['configuration_sha256'] or
                result['cfg'] != base['config'][0]['args'] or ref['process_receipt']['returncode'] != 0):
            raise ValueError('Control must be the completed matching attempt-2 run')
    for name, digest in original['source_sha256'].items():
        if name.startswith('swiglu/') or name == 'TEACHERS.json':
            if manifest['source_sha256'].get(name) != digest or sha256(relative_file(root, name)) != digest:
                raise ValueError('Scientific source differs from rank-study controls')
    if manifest['criterion'] != original['criterion'] or design['criterion'] != original['criterion']:
        raise ValueError('Original 1percent/5percent criteria must remain unchanged')


'''

PROTOCOL = '''# Raw ReLU teacher / SwiGLU: two paired head-rate follow-ups

The two fixed new runs use seeds 641 and 642 from the completed rank-8 controls. The only scientific change is `head_lr: 0.01 -> 0.001`; output tags are new. Both outcomes are retained, including caps, failures and numerical ambiguity. These are reused development seeds, not fresh confirmation. The original controls and every prior artifact remain intact.

Unchanged settings: raw ReLU teacher, eight unit coefficients, d = 64, m = 64, Gaussian scale 0.1, head_ratio = 1, alpha = 1, profiled intercept and biases; h = 0.01, dt_max = 50, t_max = 3000, max_steps = 20000, variance-normalized L_stop = 0.01; quadrature orders 32/96/48/24; cp_ratio = 1.06, cp_min = 0.5, dl_ratio = 1.5 and diagnostic priority fractions 0.25/0.5/0.75. All scientific source bytes match the rank study.

The controls are the exact completed `rank_retry_a02` attempts for `v10_rank_swiglu_relu_r8_d64_m64_seed641/642`, verified against the remote final inventory, execution snapshot, process receipts and original configuration hashes. They are completed processes with wall-censored trajectories (t = 1042.1841 / 951.0148), not horizon-complete experiments. `CONTROL_PROVENANCE.json` freezes their saved-data hashes and first P/V/a array hashes; no restart or generated control substitutes for these data. Initialization uses only the unchanged seed, dimensions, scale and head_ratio, so source inspection predicts identical initial arrays. Verify that equality from the new saved arrays after execution before interpreting paired comparisons. Compare trajectories on their common observed support and preserve each full per-run verdict.

Apply both original initial whole-update max/min loss windows 1.01 and 1.05; require same-checkpoint screened minimum-direction gain >= 0.5 and refit-improvement lower envelope difference >= 0.1 Var(Y), followed by a later same-run raw loss decrease >= 0.1 Var(Y). Keep the original initial reference, screens and tolerances. Here E[Y²] = 1 but Var(Y) = 0.21116926371833117; the raw 0.1-variance threshold is 0.021116926371833117. Refit cutoff envelopes are numerical sensitivity samples, not formal oracle bounds. A promising result still needs block-weighted-update, AGOP and refit order checks; no order-4 validation of these raw-link trajectories is claimed.

The profiled intercept fits the mean at every state. This test changes the head contribution to the continuous loss derivative by a factor of ten at a fixed state; it does not change initial features or remove the linear teacher component. The adaptive step also depends on the head-rate-weighted direction, so the new path is not a clock rescaling. Strong top-direction capture alone does not identify the aggregate linear direction or establish learning of every direction. Failure, concentration on a few directions, and a merely longer loss plateau remain plausible. This intervention is separate from the cubic coupled-head-scaling pilot.

Runtime: two single-thread SERVER workers, 30 reserved CPU slots, at least 32 effective CPUs, 12 GiB available memory and 5 GiB free disk, inherited nice >= 10. Each arm has 1800 s soft training + 180 s soft diagnostics + 10 s cleanup = 1990 s; the global cap is 2150 s, including bounded setup after queue admission. Both arms share one wave. Work near the global deadline is shortened, not extended. The reviewed scheduler's cleanup, reaping, source checks, output refusal and persistent restart guard are unchanged. Reservations do not discover other jobs; the parent performs a fresh combined live-process capacity check before dispatch.

The exact-manifest review gate remains pending. No automatic dispatch, training, model evaluation, publication, network action or manuscript edit is part of preparation. The parent owns independent review, publication and dispatch. Use a new leaf `population_agop_raw_relu_headslow_20260928`, job `agop-rawrelu-headslow-20260928-a01`, and output directory `execution_headslow_a01`.
'''


def main():
    if BUNDLE.exists(): raise FileExistsError('Preserve existing bundle')
    original = load(RANK / 'MANIFEST.json')
    scheduler = load(SCHEDULER / 'MANIFEST.json')
    inventory = load(RETRY / 'REMOTE_FINAL_INVENTORY.json')
    assert sha(RANK / 'MANIFEST.json') == ORIGINAL_HASH
    assert sha(RETRY / 'REMOTE_FINAL_INVENTORY.json') == INVENTORY_HASH
    assert all(sha(RANK / n) == h for n,h in original['source_sha256'].items())
    assert all(sha(SCHEDULER / n) == h for n,h in scheduler['source_sha256'].items())
    run = RETRY / 'remote_results' / inventory['root']
    retry_manifest = load(run / 'MANIFEST.snapshot.json')
    assert sha(run / 'MANIFEST.snapshot.json') == sha(RETRY / 'bundle/MANIFEST.json')
    for name in original['source_sha256']:
        if name.startswith('swiglu/') or name == 'TEACHERS.json': cp(RANK / name, name)
    for source,name in [(RANK/'MANIFEST.json','rank_MANIFEST.json'), (RANK/'REVIEW.json','rank_REVIEW.json'),
                        (SCHEDULER/'MANIFEST.json','scheduler_MANIFEST.json'), (SCHEDULER/'REVIEW.json','scheduler_REVIEW.json'),
                        (SCHEDULER/'launcher/launch.py','scheduler_launch.py'), (SCHEDULER/'launcher/test_launch.py','scheduler_test_launch.py'),
                        (run/'MANIFEST.snapshot.json','retry_MANIFEST.snapshot.json'), (run/'PROVENANCE.json','retry_PROVENANCE.json'),
                        (RETRY/'REMOTE_FINAL_INVENTORY.json','retry_FINAL_INVENTORY.json'), (RETRY/'TRANSFER_QA.json','retry_TRANSFER_QA.json')]:
        cp(source, 'reference/' + name)
    cp(V10/'weak_link_next/EVIDENCE_AND_NEXT_STEP.md','reference/EVIDENCE_AND_NEXT_STEP.md')
    controls = dict(inventory_sha256=INVENTORY_HASH, source_manifest_sha256=ORIGINAL_HASH,
                    execution_manifest_sha256=sha(run/'MANIFEST.snapshot.json'), controls={})
    configs = []
    for seed in (641,642):
        oid = f'v10_rank_swiglu_relu_r8_d64_m64_seed{seed}'
        cid = f'v10_swiglu_relu_headlr001_r8_d64_m64_seed{seed}'
        base = next(e for e in original['configs'] if e['id'] == oid)
        retry = next(e for e in retry_manifest['configs'] if e['id'] == oid)
        assert base['config'] == retry['config'] and base['configuration_sha256'] == retry['configuration_sha256']
        data = run/'data'/oid; result = load(data/'result.json'); launch = load(data/'launch.json')
        process = next(e for e in inventory['status']['completed'] if e['id'] == oid)
        assert process['returncode'] == 0 and process['reap_acknowledged'] and process['completion_indicator_exists']
        assert result['completed'] and result['cfg'] == base['config'][0]['args'] == launch['cfg']
        assert result['config_sha256'] == launch['config_sha256'] == base['configuration_sha256']
        assert sha(RANK / base['config_path']) == base['configuration_sha256']
        assert result['source_manifest_sha256'] == sha(BUNDLE/'swiglu/code/SOURCE_MANIFEST.json')
        for n,h in result['source_files_sha256'].items(): assert sha(BUNDLE/'swiglu/code'/n) == h
        files = {}
        for file in data.iterdir():
            if not file.is_file(): continue
            rel = str(file.relative_to(run)); expected = inventory['files'][rel]
            assert file.stat().st_size == expected['bytes'] and sha(file) == expected['sha256']
            files[file.name] = dict(path=str(file.relative_to(V10.parent)), **expected)
        for n,h in result['raw_files'].items(): assert files[n]['sha256'] == h
        raw = load(data/(oid+'.json'))
        assert {k:v for k,v in raw['cfg'].items() if k not in ('out','verbose')} == base['config'][0]['args']
        ref = f'reference/controls/{oid}'
        cp(data/'result.json',ref+'/result.json'); cp(data/'launch.json',ref+'/launch.json')
        cp(RANK/base['config_path'],ref+'/config.json')
        controls['controls'][str(seed)] = dict(id=oid, attempt=2, files=files,
            copied_result_path=ref+'/result.json', process_receipt=process, termination=result['termination'],
            initial_array_hashes=first_array_bytes(data/(oid+'_snaps.npz')), initial_saved_row=raw['rows'][0],
            config_sha256=base['configuration_sha256'], source_files_sha256=result['source_files_sha256'])
        cfg=copy.deepcopy(base['config']);cfg[0]['tag']=cid;cfg[0]['args']['head_lr']=0.001
        path=f'configs/{cid}.json';save(BUNDLE/path,cfg)
        configs.append(dict(id=cid,tag=cid,cell=base['cell'],engine='swiglu',student='swiglu',teacher='relu',rank=8,
                            seed=seed,width=64,control_id=oid,control_attempt=2,recipe='head_lr001_paired_followup',
                            supplemental=False,config_path=path,configuration_sha256=sha(BUNDLE/path),config=cfg,
                            target_variance=base['target_variance'],normalized_loss_stop=.01,
                            changes_from_control={'tag':{'before':oid,'after':cid},'args.head_lr':{'before':.01,'after':.001}}))
    save(BUNDLE/'CONTROL_PROVENANCE.json',controls)
    design=dict(planned_arms=2,paired_reused_seeds=[641,642],fresh_confirmation=False,retain_all_outcomes=True,
                controls_are_completed_attempt2=True,only_scientific_change={'head_lr':[.01,.001]},
                runtime=RUNTIME,criterion=original['criterion'],arms=configs)
    save(BUNDLE/'DESIGN.json',design)
    (BUNDLE/'PROTOCOL.md').write_text(PROTOCOL)
    (BUNDLE/'README.md').write_text(PROTOCOL+'''\nStatic checks (no model import or execution):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_headslow_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

After independent review and a fresh capacity check, the parent can use the verified server interpreter with `AGOP_EXECUTION_SITE=SERVER` and `launcher/launch.py --execution-dir execution_headslow_a01`. Keep the controls' head_lr=0.01 and follow-ups' head_lr=0.001 recipes separate in any report despite their unchanged scientific cell label. Verify the saved initial P/V/a hashes before pairing outcomes.
''')
    script=(SCHEDULER/'launcher/launch.py').read_text()
    script=script.replace('Bounded v10 cubic head-scaling SERVER-only wrapper','Bounded two-arm raw-ReLU head-rate SERVER-only wrapper')
    script=script.replace('not 1 <= workers <= 6','not 1 <= workers <= 2').replace('integer from 1 to 6','integer from 1 to 2')
    script=script.replace('Head-scaling pilot has no local execution mode.','Head-rate follow-up has no local execution mode.')
    start,end=script.index('def validate_protocol('),script.index('def require_review(')
    script=script[:start]+GATE.replace('__ORIGINAL_HASH__',ORIGINAL_HASH).replace('__INVENTORY_HASH__',INVENTORY_HASH)+script[end:]
    script=script.replace("'.headscale.lock'","'.raw_relu_headslow.lock'")
    (BUNDLE/'launcher').mkdir();(BUNDLE/'launcher/launch.py').write_text(script)
    tests=(SCHEDULER/'launcher/test_launch.py').read_text()
    edits={'test_6_workers_reserve_26_slots_on_32cpu':'test_two_workers_reserve_30_slots_on_32cpu',
           'workers=6,reserved_cpus=26,min_available_memory_gib=24,min_free_disk_gib=10':'workers=2,reserved_cpus=30,min_available_memory_gib=12,min_free_disk_gib=5',
           "['workers'],6)":"['workers'],2)",'(4,6,30,31.9)':'(2,6,30,31.9)',
           'available_memory_gib=23.9':'available_memory_gib=11.9','free_disk_gib=9.9':'free_disk_gib=4.9',
           '["workers"] = 7':'["workers"] = 3','from 1 to 6':'from 1 to 2',
           'test_six_headscale_arms_share_one_bounded_wave':'test_two_headrate_arms_share_one_bounded_wave',
           "['workers'] = 6":"['workers'] = 2",'range(1,6)':'range(1,2)',
           "len(status['completed']),6":"len(status['completed']),2",'started.count(1000.),6':'started.count(1000.),2',
           'HeadscaleGatesTests':'HeadrateGatesTests','test_frozen_headscale_design_criteria_order_and_runtime':'test_frozen_headrate_design_criteria_order_and_runtime',
           "len(manifest['configs']),6":"len(manifest['configs']),2","changed['configs'][0]['q']=.3":"changed['configs'][0]['teacher']='h3'"}
    for before,after in edits.items(): assert before in tests,before; tests=tests.replace(before,after)
    # Extend the existing strict-gate mutation test for the sole permitted delta.
    tests=tests.replace("('runtime','criteria','config','queue','labels')","('runtime','criteria','config','queue','labels','head_ratio','head_lr','orders','cohort')")
    tests=tests.replace("elif kind=='labels': changed['configs'][0]['teacher']='h3'", """elif kind=='labels': changed['configs'][0]['teacher']='h3'
            elif kind=='head_ratio': changed['configs'][0]['config'][0]['args']['head_ratio']=.1
            elif kind=='head_lr': changed['configs'][0]['config'][0]['args']['head_lr']=.01
            elif kind=='orders': changed['configs'][0]['config'][0]['args']['n_pair']=16
            elif kind=='cohort': changed['fresh_confirmation']=True""")
    (BUNDLE/'launcher/test_launch.py').write_text(tests)
    (BUNDLE/'LAUNCHER.diff').write_text(''.join(difflib.unified_diff((SCHEDULER/'launcher/launch.py').read_text().splitlines(True),script.splitlines(True),fromfile='reference/scheduler_launch.py',tofile='launcher/launch.py')))
    hashes={str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
    manifest=dict(protocol_version='v10_raw_relu_headrate_two_arm',created_utc=datetime.now(timezone.utc).isoformat(),
                  total_arms=2,paired_reused_seeds=[641,642],fresh_confirmation=False,retain_all_outcomes=True,
                  original_manifest_sha256=ORIGINAL_HASH,scheduler_manifest_sha256=sha(SCHEDULER/'MANIFEST.json'),
                  design_sha256=sha(BUNDLE/'DESIGN.json'),control_provenance_sha256=sha(BUNDLE/'CONTROL_PROVENANCE.json'),
                  runtime=RUNTIME,dependency=None,criterion=original['criterion'],configs=configs,
                  source_sha256=hashes,pinned_sources=hashes.copy(),
                  publication_leaf='population_agop_raw_relu_headslow_20260928',remote_run_name='agop-rawrelu-headslow-20260928-a01',
                  recommended_execution_directory='execution_headslow_a01')
    save(BUNDLE/'MANIFEST.json',manifest)
    save(BUNDLE/'REVIEW.json',dict(status='pending_independent_runtime_review',execution_launched=False,
         dispatch_prerequisites=['Independent exact-manifest review','Fresh combined live-process capacity check','Parent publication and manual dispatch']))
    assert all(sha(RANK/n)==h for n,h in original['source_sha256'].items())
    assert all(sha(SCHEDULER/n)==h for n,h in scheduler['source_sha256'].items())
    print(json.dumps(dict(manifest_sha256=sha(BUNDLE/'MANIFEST.json'),frozen_files=len(hashes),arms=2,review_pending=True)))


if __name__ == '__main__': main()
