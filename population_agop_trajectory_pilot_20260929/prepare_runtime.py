"""Server-only exclusive copy of four pinned reference arrays from the same Studio."""
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    if sys.platform!='linux' or os.environ.get('AGOP_EXECUTION_SITE')!='SERVER':
        raise RuntimeError('Server-only input preparation')
    here=Path(__file__).resolve().parent;plan=json.loads((here/'RUNTIME_INPUTS.json').read_text())
    source=here.parent.parent/plan['source_sibling_run']/plan['source_bundle']/plan['source_output']
    target=here/'runtime_inputs'
    if target.exists(): raise FileExistsError('Preserve existing runtime inputs; no overwrite/restart')
    payloads={}
    for name,expected in plan['files'].items():
        if Path(name).name!=name: raise ValueError('Unsafe input name')
        payload=(source/name).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected: raise ValueError('Existing remote input hash mismatch: '+name)
        payloads[name]=payload
    target.mkdir()
    for name,payload in payloads.items():
        with (target/name).open('xb') as f: f.write(payload)
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=plan['files'][name]:
            raise RuntimeError('Copied input mismatch')
    print(json.dumps(dict(copied_files=len(payloads),copied_bytes=sum(map(len,payloads.values())),source=str(source))))

if __name__=='__main__': main()
