"""Portable, bounded setup and launch for the reviewed sixty-arm CPU bundle."""
from pathlib import Path
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'bundle'
sys.path.insert(0,str(BUNDLE/'launcher'))
import launch

def check(command, cwd, deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError('Setup120s budget exhausted')
    process=subprocess.Popen(command,cwd=cwd,env=launch.child_environment(),start_new_session=True)
    try:
        code=process.wait(timeout=remaining)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)
        raise
    if code:raise RuntimeError(f'Setup check failed with exit{code}: {command}')

def main():
    deadline=time.monotonic()+120
    launch.require_server()
    manifest,digest=launch.validate(BUNDLE)
    resources=launch.require_capacity(manifest['runtime'])
    print(json.dumps(dict(stage='capacity_verified',manifest_sha256=digest,resources=resources)),flush=True)
    for line in (BUNDLE/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        name,version=line.split('==')
        actual=importlib.metadata.version(name)
        if actual!=version:raise RuntimeError(f'Dependency mismatch: {name}={actual}, require {version}; no training started')
    check([sys.executable,'-m','unittest','-q','test_launch.py'],BUNDLE/'launcher',deadline)
    check([sys.executable,'-m','unittest','-q','test_stable_relu.py'],BUNDLE/'diagnostics',deadline)
    print(json.dumps(dict(stage='setup_complete',setup_seconds=120-(deadline-time.monotonic()))),flush=True)
    # The scientific dispatcher has its own2100s global cap. exec preserves the
    # remote process identity and signals; no untracked background job is added.
    os.execve(sys.executable,[sys.executable,str(BUNDLE/'launcher/launch.py')],launch.child_environment())

if __name__=='__main__':main()
