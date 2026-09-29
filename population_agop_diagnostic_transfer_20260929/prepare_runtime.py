"""Copy a pinned existing input within the authorized Studio, never from GitHub."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if sys.platform != 'linux' or os.environ.get('AGOP_EXECUTION_SITE') != 'SERVER':
    raise SystemExit('Server-only runtime preparation')
plan = json.loads((HERE / 'PROBE_MANIFEST.json').read_text())
source = (HERE.parent.parent / 'agop-transition-hermite-20260929-a01' /
          'population_agop_transition_hermite_20260929/probe_inputs/states.npz')
target = HERE / 'probe_inputs/states.npz'
expected = plan['input_sha256']['probe_inputs/states.npz']
payload = source.read_bytes()
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit('Existing Studio input hash mismatch; no copy made')
with target.open('xb') as stream:
    stream.write(payload)
if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
    raise SystemExit('Copied input hash mismatch')
print(json.dumps({'source_run': 'agop-transition-hermite-20260929-a01',
                  'input_sha256': expected, 'copied_bytes': len(payload)}))
