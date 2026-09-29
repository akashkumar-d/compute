"""Prepare the preregistered ten-seed high-rank cohort as three chained bundles.

Standard-library file work only: no scientific import, no model evaluation, no
network. Run from a checkout of the shared compute repository:

    python3 population_agop_highrank_seeds10_20260929/prepare.py

Every scientific source is copied byte-for-byte from the reviewed v10 rank-sweep
bundle. Every configuration is a reviewed template (v10 rank sweep or v7 CPU32
breadth screen) with only the declared fields changed: id/tag/cell and a fresh
seed from 9351-9360, plus the two declared SwiGLU recipe arms (link abs/RBF on
the v10 rank-8 recipe; head_lr 0.01 -> 0.001 on the v10 rank-8/16 recipe).
"""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEAF = HERE.name
OLD = REPO / 'population_agop_rank_sweep_20260928/bundle'
BREADTH = REPO / 'population_agop_breadth14_cpu32_20260928/bundle'
OLD_HASH = 'e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01'
SEEDS = list(range(9351, 9361))
DEV_SEEDS = [641, 642]
REUSED = {'v11_hr_relu_h2_r16': dict(
    seeds=[9351, 9352, 9353, 9354],
    source='population_agop_relu_r16_confirmation_20260928 (remote run agop-relu-r16-confirm-20260928-a01)',
    note='Identical recipe (v10 rank-16 h2 ReLU config with only id/seed changed); '
         'completed, verified outcomes are reused and not rerun.')}
RUNTIME = dict(workers=28, reserved_cpus=4, min_available_memory_gib=32, min_free_disk_gib=10,
               global_seconds=3600, per_arm_seconds=1990, diagnostic_reserve_seconds=180)
RUN = 'agop-highrank-s10-20260929-{}'
BUNDLES = {
    'a01': dict(run_name=RUN.format('a01'), execution_dir='execution_a01', dependency=None),
    'a02': dict(run_name=RUN.format('a02'), execution_dir='execution_a02',
                dependency=dict(run=RUN.format('a01'), max_wait_seconds=4500)),
    'a03': dict(run_name=RUN.format('a03'), execution_dir='execution_a03',
                dependency=dict(run=RUN.format('a02'), max_wait_seconds=9000)),
}
TEMPLATES = {
    'relu_h2_r2': ('rank', 'configs/v10_rank_relu_h2_r2_d64_m256_seed641.json'),
    'relu_h2_r4': ('rank', 'configs/v10_rank_relu_h2_r4_d64_m256_seed641.json'),
    'relu_h2_r8': ('rank', 'configs/v10_rank_relu_h2_r8_d64_m256_seed641.json'),
    'relu_h2_r16': ('rank', 'configs/v10_rank_relu_h2_r16_d64_m256_seed641.json'),
    'relu_abs_r8': ('breadth', 'configs/relu_abs_r8_d64_m256_seed641.json'),
    'relu_gaussian_rbf_r8': ('breadth', 'configs/relu_gaussian_rbf_r8_d64_m256_seed641.json'),
    'swiglu_h2_r2': ('rank', 'configs/v10_rank_swiglu_h2_r2_d64_m64_seed641.json'),
    'swiglu_h2_r4': ('rank', 'configs/v10_rank_swiglu_h2_r4_d64_m64_seed641.json'),
    'swiglu_h2_r8': ('rank', 'configs/v10_rank_swiglu_h2_r8_d64_m64_seed641.json'),
    'swiglu_h2_r16': ('rank', 'configs/v10_rank_swiglu_h2_r16_d64_m64_seed641.json'),
}
# cell, engine, teacher, rank, width, template, recipe changes (SwiGLU args only)
CELLS = [
    ('v11_hr_relu_h2_r2', 'relu', 'h2', 2, 256, 'relu_h2_r2', {}),
    ('v11_hr_relu_h2_r4', 'relu', 'h2', 4, 256, 'relu_h2_r4', {}),
    ('v11_hr_relu_h2_r8', 'relu', 'h2', 8, 256, 'relu_h2_r8', {}),
    ('v11_hr_relu_h2_r16', 'relu', 'h2', 16, 256, 'relu_h2_r16', {}),
    ('v11_hr_relu_abs_r8', 'relu', 'abs', 8, 256, 'relu_abs_r8', {}),
    ('v11_hr_relu_gaussian_rbf_r8', 'relu', 'gaussian_rbf', 8, 256, 'relu_gaussian_rbf_r8', {}),
    ('v11_hr_swiglu_h2_r2', 'swiglu', 'h2', 2, 64, 'swiglu_h2_r2', {}),
    ('v11_hr_swiglu_h2_r4', 'swiglu', 'h2', 4, 64, 'swiglu_h2_r4', {}),
    ('v11_hr_swiglu_h2_r8', 'swiglu', 'h2', 8, 64, 'swiglu_h2_r8', {}),
    ('v11_hr_swiglu_h2_r16', 'swiglu', 'h2', 16, 64, 'swiglu_h2_r16', {}),
    ('v11_hr_swiglu_abs_r8', 'swiglu', 'abs', 8, 64, 'swiglu_h2_r8', {'link': 'abs'}),
    ('v11_hr_swiglu_gaussian_rbf_r8', 'swiglu', 'gaussian_rbf', 8, 64, 'swiglu_h2_r8', {'link': 'gaussian_rbf'}),
    ('v11_hr_swiglu_h2_headslow_r8', 'swiglu', 'h2', 8, 64, 'swiglu_h2_r8', {'head_lr': 0.001}),
    ('v11_hr_swiglu_h2_headslow_r16', 'swiglu', 'h2', 16, 64, 'swiglu_h2_r16', {'head_lr': 0.001}),
]
S = SEEDS
QUEUE = {  # (cell, seeds) in launch order; longest arms first inside each SwiGLU slice
    'a01': [('v11_hr_relu_h2_r16', S[4:]), ('v11_hr_relu_h2_r8', S), ('v11_hr_relu_gaussian_rbf_r8', S),
            ('v11_hr_relu_abs_r8', S), ('v11_hr_relu_h2_r4', S), ('v11_hr_relu_h2_r2', S),
            ('v11_hr_swiglu_h2_r16', S), ('v11_hr_swiglu_h2_headslow_r16', S), ('v11_hr_swiglu_h2_r8', S[:8])],
    'a02': [('v11_hr_swiglu_h2_r8', S[8:]), ('v11_hr_swiglu_h2_headslow_r8', S),
            ('v11_hr_swiglu_abs_r8', S), ('v11_hr_swiglu_gaussian_rbf_r8', S[:6])],
    'a03': [('v11_hr_swiglu_gaussian_rbf_r8', S[6:]), ('v11_hr_swiglu_h2_r4', S), ('v11_hr_swiglu_h2_r2', S)],
}

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def job_id(cell, width, seed):
    return f'{cell}_d64_m{width}_seed{seed}'


def build_config(engine, template, cid, cell, seed, changes):
    """The only construction rule; duplicated verbatim inside the launcher gate."""
    if engine == 'relu':
        if changes:
            raise ValueError('ReLU configs admit no recipe change')
        return dict(template, id=cid, seed=seed)
    (outer,) = template
    args = dict(outer['args'], seed=seed)
    for key, value in changes.items():
        if key not in ('link', 'head_lr') or key not in args:
            raise ValueError('Only link/head_lr may change in a SwiGLU recipe')
        args[key] = value
    return [dict(outer, tag=cid, cell=cell, args=args)]


GATE = r'''def build_config(engine, template, cid, cell, seed, changes):
    """The only construction rule; duplicated verbatim from prepare.py."""
    if engine == 'relu':
        if changes:
            raise ValueError('ReLU configs admit no recipe change')
        return dict(template, id=cid, seed=seed)
    (outer,) = template
    args = dict(outer['args'], seed=seed)
    for key, value in changes.items():
        if key not in ('link', 'head_lr') or key not in args:
            raise ValueError('Only link/head_lr may change in a SwiGLU recipe')
        args[key] = value
    return [dict(outer, tag=cid, cell=cell, args=args)]


def validate_protocol(manifest, root=ROOT):
    """Only this bundle's slice of the preregistered ten-seed high-rank design."""
    bundle_name = '__BUNDLE__'
    expected_runtime = dict(workers=28, reserved_cpus=4, min_available_memory_gib=32,
                            min_free_disk_gib=10, global_seconds=3600, per_arm_seconds=1990,
                            diagnostic_reserve_seconds=180)
    if manifest['runtime'] != expected_runtime:
        raise ValueError('Runtime must equal the frozen v10 rank-study bounds')
    design_path = relative_file(root, 'DESIGN.json')
    if sha256(design_path) != '__DESIGN_SHA__' or manifest.get('design_sha256') != '__DESIGN_SHA__':
        raise ValueError('Preregistered design changed')
    design = load_json(design_path)
    declared = design['bundles'][bundle_name]
    if manifest.get('dependency') != declared['dependency']:
        raise ValueError('Dependency must equal the declared bundle chain')
    audit_path = relative_file(root, 'FRESH_SEED_AUDIT.json')
    if sha256(audit_path) != '__AUDIT_SHA__' or manifest.get('fresh_seed_audit_sha256') != '__AUDIT_SHA__':
        raise ValueError('Precreation seed audit changed')
    audit = load_json(audit_path)
    seeds = list(range(9351, 9361))
    if (design.get('seeds') != seeds or audit.get('seeds') != seeds or audit.get('read_errors') or
            audit.get('undeclared_collisions')):
        raise ValueError('Fresh-seed audit does not establish the fixed unused range')
    reference_path = relative_file(root, 'reference/rank_MANIFEST.json')
    if sha256(reference_path) != 'e9a0a074a91c498784af86da59d6de440eb5d3baa208e955161b87484f9e3b01':
        raise ValueError('Reference rank-study manifest changed')
    reference = load_json(reference_path)
    templates = {}
    for name, info in design['templates'].items():
        path = relative_file(root, info['path'])
        if sha256(path) != info['sha256']:
            raise ValueError('Template configuration changed: ' + name)
        templates[name] = load_json(path)
    cells = {c['cell']: c for c in design['cells']}
    jobs = [j for j in design['jobs'] if j['bundle'] == bundle_name]
    if not jobs or [e['id'] for e in manifest['configs']] != [j['id'] for j in jobs]:
        raise ValueError('Configs must equal this bundle slice in the declared queue order')
    if manifest.get('total_arms') != len(jobs):
        raise ValueError('Arm count differs from the declared slice')
    for entry, job in zip(manifest['configs'], jobs):
        cell = cells[job['cell']]
        if job['seed'] not in seeds:
            raise ValueError('Seed outside the preregistered fresh range')
        if job['seed'] in cell.get('reused_existing_seeds', []):
            raise ValueError('A reused existing outcome must not be rerun')
        cid = '%s_d64_m%d_seed%d' % (cell['cell'], cell['width'], job['seed'])
        cfg = build_config(cell['engine'], templates[cell['template']], cid, cell['cell'],
                           job['seed'], cell['changes'])
        if (job['id'] != cid or entry['id'] != cid or entry['config'] != cfg or
                entry['engine'] != cell['engine'] or entry['student'] != cell['engine'] or
                entry['teacher'] != cell['teacher'] or entry['rank'] != cell['rank'] or
                entry['width'] != cell['width'] or entry['seed'] != job['seed'] or
                entry['cell'] != cell['cell'] or entry['config_path'] != 'configs/%s.json' % cid):
            raise ValueError('Configuration differs from template plus declared changes: ' + cid)
        path = relative_file(root, entry['config_path'])
        if sha256(path) != entry['configuration_sha256'] or load_json(path) != cfg:
            raise ValueError('Configuration file differs from its frozen object/hash: ' + cid)
    for name, digest in reference['source_sha256'].items():
        if name.startswith(('code/', 'diagnostics/', 'swiglu/')) or name in ('TEACHERS.json', 'requirements.txt'):
            if manifest['source_sha256'].get(name) != digest or sha256(relative_file(root, name)) != digest:
                raise ValueError('Scientific source changed from the reviewed rank-study bundle: ' + name)
    if manifest.get('criterion') != reference['criterion'] or design.get('criterion') != reference['criterion']:
        raise ValueError('Original 1percent/5percent same-state criteria must remain unchanged')


'''

GATE_TESTS = r'''
    def test_frozen_highrank_slice_gate(self):
        root=Path(__file__).resolve().parent.parent
        manifest=launch.load_json(root/'MANIFEST.json')
        launch.validate_protocol(manifest,root)
        for kind in ('runtime','criteria','seed','queue','labels','dependency','recipe','drop','count'):
            changed=json.loads(json.dumps(manifest))
            if kind=='runtime': changed['runtime']['global_seconds']=3000
            elif kind=='criteria': changed['criterion']['same_checkpoint_minimum_alignment_gain']=.4
            elif kind=='queue': changed['configs'].reverse()
            elif kind=='labels': changed['configs'][0]['rank']=3
            elif kind=='dependency': changed['dependency']={'run':'other','max_wait_seconds':1}
            elif kind=='drop': changed['configs'].pop()
            elif kind=='count': changed['total_arms']+=1
            else:
                cfg=changed['configs'][0]['config']
                if kind=='seed':
                    if isinstance(cfg,list):cfg[0]['args']['seed']=641
                    else:cfg['seed']=641
                elif isinstance(cfg,list):cfg[0]['args']['h']=0.02
                else:cfg['h']=0.25
            with self.assertRaises(ValueError,msg=kind):launch.validate_protocol(changed,root)

    def test_construction_rule_admits_only_declared_changes(self):
        relu={'id':'x','seed':1,'h':.5}
        self.assertEqual(launch.build_config('relu',relu,'y','c',9351,{}),{'id':'y','seed':9351,'h':.5})
        with self.assertRaises(ValueError):launch.build_config('relu',relu,'y','c',9351,{'h':.25})
        sw=[{'tag':'x','cell':'c0','rank':8,'args':{'link':'h2','head_lr':.01,'seed':1,'s':.1}}]
        out=launch.build_config('swiglu',sw,'y','c1',9352,{'head_lr':.001})
        self.assertEqual(out,[{'tag':'y','cell':'c1','rank':8,'args':{'link':'h2','head_lr':.001,'seed':9352,'s':.1}}])
        for bad in ({'s':.2},{'n_pair':16},{'unknown':1}):
            with self.assertRaises(ValueError):launch.build_config('swiglu',sw,'y','c1',9352,bad)

    def test_seed_audit_and_reuse_frozen(self):
        root=Path(__file__).resolve().parent.parent
        audit=launch.load_json(root/'FRESH_SEED_AUDIT.json')
        design=launch.load_json(root/'DESIGN.json')
        self.assertEqual(audit['seeds'],list(range(9351,9361)))
        self.assertFalse(audit['undeclared_collisions'])
        self.assertFalse(audit['read_errors'])
        self.assertGreater(audit['files_checked'],1500)
        reused=[c for c in design['cells'] if c.get('reused_existing_seeds')]
        self.assertEqual([(c['cell'],c['reused_existing_seeds']) for c in reused],
                         [('v11_hr_relu_h2_r16',[9351,9352,9353,9354])])
        self.assertFalse(any(j['cell']=='v11_hr_relu_h2_r16' and j['seed']<9355 for j in design['jobs']))
        manifest=launch.load_json(root/'MANIFEST.json')
        manifest['fresh_seed_audit_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'seed audit changed'):launch.validate_protocol(manifest,root)

'''


def audit_record():
    """Merge the two precreation scans (Mac experiment tree, shared repo)."""
    parts = [load(HERE / 'audit/SEED_AUDIT_LOCAL_9351_9360.json'), load(HERE / 'audit/SEED_AUDIT_REPO_9351_9360.json')]
    collisions, undeclared = [], []
    for part in parts:
        for c in part['collisions']:
            collisions.append(c)
            declared = ('relu_rank16_confirmation' in c['file'] or 'relu_r16_confirmation' in c['file'])
            if not (declared and c['seed'] in REUSED['v11_hr_relu_h2_r16']['seeds']):
                undeclared.append(c)
    return dict(status='PASS_unused_except_declared_reuse' if not undeclared else 'FAIL',
                seeds=SEEDS, scans=[dict(roots=p['roots'], created_utc=p['created_utc'],
                                        files_checked=p['files_checked']) for p in parts],
                files_checked=sum(p['files_checked'] for p in parts), collisions=collisions,
                declared_reuse=REUSED, undeclared_collisions=undeclared,
                read_errors=[e for p in parts for e in p['read_errors']],
                input_sha256={k: v for p in parts for k, v in p['input_sha256'].items()},
                scope=parts[0]['scope'])


def main():
    assert sha(OLD / 'MANIFEST.json') == OLD_HASH, 'reference rank-study manifest changed'
    old = load(OLD / 'MANIFEST.json')
    assert all(sha(OLD / p) == h for p, h in old['source_sha256'].items()), 'reference sources changed'
    breadth_hash = sha(BREADTH / 'MANIFEST.json')
    breadth = load(BREADTH / 'MANIFEST.json')
    assert all(sha(BREADTH / p) == h for p, h in breadth['source_sha256'].items()), 'breadth sources changed'
    for name in breadth['source_sha256']:  # identical scientific code in both reviewed bundles
        if name.startswith(('code/', 'swiglu/')) and name in old['source_sha256']:
            assert breadth['source_sha256'][name] == old['source_sha256'][name], name
    audit = audit_record()
    assert audit['status'].startswith('PASS'), audit['undeclared_collisions']
    templates = {}
    for name, (origin, rel) in TEMPLATES.items():
        base = OLD if origin == 'rank' else BREADTH
        manifest = old if origin == 'rank' else breadth
        entry = next(e for e in manifest['configs'] if e['config_path'] == rel)
        assert load(base / rel) == entry['config'] and entry['seed'] == 641
        templates[name] = dict(path=f'reference/templates/{Path(rel).name}', sha256=sha(base / rel),
                               origin=f"{'population_agop_rank_sweep_20260928' if origin == 'rank' else 'population_agop_breadth14_cpu32_20260928'}/bundle/{rel}",
                               origin_manifest_sha256=OLD_HASH if origin == 'rank' else breadth_hash,
                               origin_seed=641, config=entry['config'])
    cells = []
    for cell, engine, teacher, rank, width, template, changes in CELLS:
        cells.append(dict(cell=cell, engine=engine, teacher=teacher, rank=rank, width=width,
                          template=template, changes=changes,
                          reused_existing_seeds=REUSED.get(cell, {}).get('seeds', []),
                          planned_new_seeds=[s for s in SEEDS if s not in REUSED.get(cell, {}).get('seeds', [])]))
    cellmap = {c['cell']: c for c in cells}
    jobs = []
    for bundle, queue in QUEUE.items():
        for cell, seeds in queue:
            for seed in seeds:
                c = cellmap[cell]
                jobs.append(dict(bundle=bundle, cell=cell, seed=seed, id=job_id(cell, c['width'], seed)))
    ids = [j['id'] for j in jobs]
    assert len(ids) == len(set(ids)) == sum(len(c['planned_new_seeds']) for c in cells) == 136
    for c in cells:  # every planned new (cell, seed) appears exactly once
        assert sorted(j['seed'] for j in jobs if j['cell'] == c['cell']) == c['planned_new_seeds'], c['cell']
    design = dict(schema='highrank_ten_fresh_seeds_v1', status='preregistered_unlaunched',
                  created_before_outcomes=True, seeds=SEEDS, development_seeds_separate=DEV_SEEDS,
                  keep_all_outcomes=True, no_seed_replacement=True,
                  purpose='Ten fresh seeds per cell for publication figures of the aggregated clean high-rank '
                          'ReLU cells and the SwiGLU rank study, fixed before any outcome.',
                  denominators={c['cell']: 10 for c in cells},
                  reference_rank_manifest_sha256=OLD_HASH, reference_breadth_manifest_sha256=breadth_hash,
                  criterion=old['criterion'], runtime=RUNTIME, bundles=BUNDLES, templates=templates,
                  cells=cells, jobs=jobs, leaf=LEAF)
    for bundle, info in BUNDLES.items():
        B = HERE / f'bundle_{bundle}'
        if B.exists():
            raise FileExistsError(f'Preserve existing {B}')
        for name in old['source_sha256']:
            if name.startswith(('code/', 'diagnostics/', 'swiglu/')) or name in ('TEACHERS.json', 'requirements.txt', 'SOURCE_REVIEW.json'):
                (B / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(OLD / name, B / name)
        for src, name in [(OLD / 'MANIFEST.json', 'rank_MANIFEST.json'), (OLD / 'REVIEW.json', 'rank_REVIEW.json'),
                          (OLD / 'launcher/launch.py', 'rank_launch.py'), (OLD / 'launcher/test_launch.py', 'rank_test_launch.py'),
                          (BREADTH / 'MANIFEST.json', 'breadth14_cpu32_MANIFEST.json')]:
            (B / 'reference').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, B / 'reference' / name)
        for name, (origin, rel) in TEMPLATES.items():
            base = OLD if origin == 'rank' else BREADTH
            (B / 'reference/templates').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / rel, B / templates[name]['path'])
        save(B / 'FRESH_SEED_AUDIT.json', audit)
        save(B / 'DESIGN.json', design)
    design_sha = sha(HERE / 'bundle_a01/DESIGN.json')
    audit_sha = sha(HERE / 'bundle_a01/FRESH_SEED_AUDIT.json')
    out = {}
    for bundle, info in BUNDLES.items():
        B = HERE / f'bundle_{bundle}'
        assert sha(B / 'DESIGN.json') == design_sha and sha(B / 'FRESH_SEED_AUDIT.json') == audit_sha
        configs = []
        for job in [j for j in jobs if j['bundle'] == bundle]:
            c = cellmap[job['cell']]
            cfg = build_config(c['engine'], templates[c['template']]['config'], job['id'], c['cell'], job['seed'], c['changes'])
            path = f"configs/{job['id']}.json"
            save(B / path, cfg)
            template_cfg = templates[c['template']]['config']
            if c['engine'] == 'relu':
                diff = {k: dict(before=template_cfg.get(k), after=v) for k, v in cfg.items() if template_cfg.get(k) != v}
            else:
                diff = {k: dict(before=template_cfg[0]['args'].get(k), after=v) for k, v in cfg[0]['args'].items()
                        if template_cfg[0]['args'].get(k) != v}
                for k in ('tag', 'cell'):
                    diff[k] = dict(before=template_cfg[0][k], after=cfg[0][k])
            configs.append(dict(id=job['id'], tag=job['id'], cell=c['cell'], engine=c['engine'], student=c['engine'],
                                teacher=c['teacher'], rank=c['rank'], seed=job['seed'], width=c['width'],
                                supplemental=False, recipe=c['template'] + ('' if not c['changes'] else '+' + '+'.join(f'{k}={v}' for k, v in c['changes'].items())),
                                template=templates[c['template']]['origin'], config_path=path,
                                configuration_sha256=sha(B / path), config=cfg, changes_from_template=diff,
                                seed_status='fresh_preregistered_unused_in_this_cell'))
        script = (OLD / 'launcher/launch.py').read_text()
        script = script.replace('Bounded v10 rank-sweep SERVER-only wrapper; dry-run is static.',
                                f'Bounded ten-seed high-rank cohort (bundle {bundle}) SERVER-only wrapper; dry-run is static.')
        script = script.replace('Rank sweep has no local execution mode.', 'High-rank cohort has no local execution mode.')
        start, end = script.index('def validate_protocol('), script.index('def require_review(')
        gate = GATE.replace('__BUNDLE__', bundle).replace('__DESIGN_SHA__', design_sha).replace('__AUDIT_SHA__', audit_sha)
        script = script[:start] + gate + script[end:]
        script = script.replace("'.rank_sweep.lock'", f"'.highrank_s10_{bundle}.lock'")
        script = script.replace('default="execution",', f'default="{info["execution_dir"]}",')
        (B / 'launcher').mkdir(parents=True, exist_ok=True)
        (B / 'launcher/launch.py').write_text(script)
        tests = (OLD / 'launcher/test_launch.py').read_text()
        s0 = tests.index('    def test_frozen_rank_design_criteria_order_and_runtime(self):')
        s1 = tests.index('    def test_review_refuses_unreviewed_or_wrong_manifest(self):')
        tests = tests[:s0] + GATE_TESTS.lstrip('\n') + tests[s1:]
        tests = tests.replace('class V10GatesTests', 'class HighRankGatesTests')
        tests = tests.replace('with self.assertRaises(FileExistsError):\n            launch.main(["--dry-run"], self.root)',
                              'with self.assertRaises(FileExistsError):\n            launch.main(["--dry-run", "--execution-dir", "execution"], self.root)')
        (B / 'launcher/test_launch.py').write_text(tests)
        (B / 'LAUNCHER.diff').write_text(''.join(difflib.unified_diff(
            (OLD / 'launcher/launch.py').read_text().splitlines(True), script.splitlines(True),
            fromfile='reference/rank_launch.py', tofile='launcher/launch.py')))
        hashes = {str(p.relative_to(B)): sha(p) for p in sorted(B.rglob('*')) if p.is_file()
                  and p.name not in ('MANIFEST.json', 'REVIEW.json')}
        counts = {}
        for e in configs:
            counts[e['cell']] = counts.get(e['cell'], 0) + 1
        manifest = dict(protocol_version='v11_highrank_ten_fresh_seeds', bundle=bundle,
                        created_utc=datetime.now(timezone.utc).isoformat(), created_before_outcomes=True,
                        phase='ten_fresh_seed_publication_cohort_for_selected_highrank_cells',
                        total_arms=len(configs), arms_per_cell_in_this_bundle=counts,
                        design_denominator_per_cell=10, independent_seed_count=10,
                        development_seeds_not_pooled=DEV_SEEDS, reused_existing_outcomes=REUSED,
                        ranks=sorted({e['rank'] for e in configs}), dimension=64, runtime=RUNTIME,
                        dependency=info['dependency'], design_sha256=design_sha, fresh_seed_audit_sha256=audit_sha,
                        criterion=old['criterion'], configs=configs,
                        arms=[{k: v for k, v in e.items() if k != 'config'} for e in configs],
                        source_sha256=hashes, pinned_sources=hashes.copy(),
                        queue_order='Declared order: ' + '; '.join(f'{cell}×{len(seeds)}' for cell, seeds in QUEUE[bundle]),
                        publication_leaf=LEAF, remote_run_name=info['run_name'],
                        default_execution_directory=info['execution_dir'],
                        analysis_note='All ten fresh seeds per cell kept in every denominator; development 641/642 '
                                      'separate; ReLU h2 r16 seeds 9351-9354 reuse the completed identical-recipe '
                                      'confirmation outcomes. No threshold change or seed replacement.')
        save(B / 'MANIFEST.json', manifest)
        save(B / 'REVIEW.json', dict(status='pending_independent_review', execution_launched=False,
                                     manifest_sha256=None,
                                     dispatch_prerequisites=['Independent exact source/config/design/runtime review',
                                                             'Fresh live-capacity check and aggregate<=30 task workers',
                                                             'Shared-code publication and manual dispatch']))
        out[bundle] = dict(manifest_sha256=sha(B / 'MANIFEST.json'), arms=len(configs), cells=counts)
    save(HERE / 'PREPARATION_RECEIPT.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                                design_sha256=design_sha, fresh_seed_audit_sha256=audit_sha,
                                                reference_rank_manifest_sha256=OLD_HASH,
                                                reference_breadth_manifest_sha256=breadth_hash, bundles=out,
                                                model_evaluations=0, network_calls=0))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
