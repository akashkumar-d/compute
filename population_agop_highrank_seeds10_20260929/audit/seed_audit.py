#!/usr/bin/env python3
"""Precreation fresh-seed audit (read-only, standard library).

Scans every existing JSON/YAML/YML/TOML file whose basename contains
'manifest' or whose path contains 'config' under each root, and reports every
occurrence of a candidate seed. JSON is inspected recursively for seed-named
fields; other formats textually. Nothing is written except the report.
"""
import argparse, hashlib, json, os, re, sys
from datetime import datetime, timezone

def seed_hits(value, seeds, path=''):
    hits = []
    if isinstance(value, dict):
        for k, v in value.items():
            p = f'{path}/{k}'
            if 'seed' in str(k).lower():
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if isinstance(x, int) and not isinstance(x, bool) and x in seeds:
                        hits.append((p, x))
            hits += seed_hits(v, seeds, p)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            hits += seed_hits(v, seeds, f'{path}[{i}]')
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True)
    ap.add_argument('--seeds', default='9351-9360')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    seeds = set(range(lo, hi + 1))
    files = fields = 0
    collisions, errors, hashes = [], [], {}
    for root in a.root:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ('.git', 'node_modules')]
            for f in filenames:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root)
                low = f.lower()
                if not low.endswith(('.json', '.yaml', '.yml', '.toml')):
                    continue
                if 'manifest' not in low and 'config' not in rel.lower():
                    continue
                files += 1
                try:
                    raw = open(full, 'rb').read()
                    hashes[f'{os.path.basename(root.rstrip("/"))}/{rel}'] = hashlib.sha256(raw).hexdigest()
                    if low.endswith('.json'):
                        obj = json.loads(raw)
                        found = seed_hits(obj, seeds)
                        fields += len(re.findall(rb'seed', raw))
                        for p, s in found:
                            collisions.append(dict(root=root, file=rel, field=p, seed=s))
                    else:
                        for s in seeds:
                            if re.search(rb'\b%d\b' % s, raw):
                                collisions.append(dict(root=root, file=rel, field='text', seed=s))
                except Exception as e:
                    errors.append(dict(root=root, file=rel, error=f'{type(e).__name__}: {e}'))
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), seeds=sorted(seeds), roots=a.root,
               scope='JSON/YAML/YML/TOML files whose basename contains manifest or path contains config; '
                     'JSON recursively inspected for seed-named fields; other formats textually.',
               files_checked=files, seed_mentions_scanned=fields, collisions=collisions,
               read_errors=errors, input_sha256=hashes)
    open(a.out, 'w').write(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(files=files, collisions=len(collisions), errors=len(errors),
                          colliding_files=sorted({c['file'] for c in collisions})[:40])))

if __name__ == '__main__':
    main()
