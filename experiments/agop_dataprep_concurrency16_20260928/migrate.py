"""Drain the owned four-worker launcher, then continue never-started arms at 16.

Original scientific files and execution receipts remain untouched. This controller
is run detached on the same server, using the original pinned Python environment.
"""
from pathlib import Path
import datetime as dt
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ORIGINAL = Path('/teamspace/studios/this_studio/compute/runs/agop-dataprep-broader-20260928-v3/experiments/agop_broader_teachers_dataprep_20260928')
EXPECTED = '26e028550adbe5edf612ffcf0a7fce0bb2e6dd6256c9bbe6dece49ab942d59ab'
HERE = Path(__file__).resolve().parent
DEAD = {'Z', 'X'}

def open_process_handle(pid):
    """Use libc's pidfd API when the conda Python lacks the optional os wrapper."""
    lib = ctypes.CDLL(None,use_errno=True)
    fn = lib.pidfd_open
    fn.argtypes = (ctypes.c_int,ctypes.c_uint)
    fn.restype = ctypes.c_int
    result = fn(pid,0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error,os.strerror(error))
    return result

def send_process_signal(handle,sig):
    lib = ctypes.CDLL(None,use_errno=True)
    fn = lib.pidfd_send_signal
    fn.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint)
    fn.restype = ctypes.c_int
    if fn(handle,sig,None,0) < 0:
        error = ctypes.get_errno()
        raise OSError(error,os.strerror(error))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        s = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=s[0], ppid=int(s[1]), pgid=int(s[2]),
                    sid=int(s[3]), start_ticks=int(s[19]), exit_status=int(s[49]),
                    argv=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None

def processes():
    return [s for p in Path('/proc').iterdir() if p.name.isdigit()
            if (s := proc(int(p.name))) is not None]

def patched_launcher(text):
    replacements = {
        'not 1 <= workers <= 4:': 'not 1 <= workers <= 16:',
        'runtime.workers must be an integer from 1 to 4':
            'runtime.workers must be an integer from 1 to 16',
        'deadline = started + float(runtime["global_seconds"])':
            'deadline = min(started + float(runtime["global_seconds"]),\n'
            '                       started + datetime.fromisoformat(runtime["absolute_deadline_utc"]).timestamp() - time.time())',
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError('Unexpected original launcher; refusing to patch')
        text = text.replace(old, new)
    return text

def remaining_configs(manifest, consumed):
    all_ids = {c['id'] for c in manifest['configs']}
    if not consumed <= all_ids:
        raise RuntimeError('Unknown consumed configuration ID')
    return [c for c in manifest['configs'] if c['id'] not in consumed]

def main():
    if sys.platform != 'linux' or os.environ.get('AGOP_EXECUTION_SITE') != 'SERVER':
        raise RuntimeError('SERVER Linux only')
    if len(os.sched_getaffinity(0)) < 16:
        raise RuntimeError('Fewer than 16 available CPU cores')
    receipt_path = HERE/'HANDOFF.json'
    if receipt_path.exists() or (HERE/'continuation').exists():
        raise RuntimeError('Refusing to overwrite existing handoff')
    if digest(ORIGINAL/'MANIFEST.json') != EXPECTED:
        raise RuntimeError('Original manifest changed')
    m = read(ORIGINAL/'MANIFEST.json')
    for name, sha in m['source_sha256'].items():
        if digest(ORIGINAL/name) != sha:
            raise RuntimeError('Original source changed: ' + name)
    # Copy everything before suspending the old scheduler; no scientific imports.
    dest = HERE/'continuation'
    dest.mkdir()
    for name in m['source_sha256']:
        target = dest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ORIGINAL/name).read_bytes())
        if digest(target) != m['source_sha256'][name]:
            raise RuntimeError('Copied source hash mismatch: '+name)
    lp = dest/'launcher/launch.py'
    lp.write_text(patched_launcher(lp.read_text()))
    initial = read(ORIGINAL/'execution/STATUS.json')
    pid = initial['launcher_pid']
    identity = proc(pid)
    if identity is None or identity['state'] in DEAD:
        raise RuntimeError('Original scheduler already exited; inspect rather than restart')
    if Path('/proc', str(pid), 'cwd').resolve() != ORIGINAL:
        raise RuntimeError('Original scheduler cwd mismatch')
    if identity['argv'][-1] != 'launcher/launch.py':
        raise RuntimeError('Original scheduler command mismatch')
    # The first starting receipt gives the actual original absolute budget cutoff.
    history = [json.loads(s) for s in (ORIGINAL/'execution/STATE_HISTORY.jsonl').read_text().splitlines()]
    start = next(s for s in history if s['status'] == 'starting')
    cutoff = min(dt.datetime.fromisoformat(start['updated_utc']).timestamp() - start['compute_elapsed_seconds'],
                 dt.datetime.fromisoformat(read(ORIGINAL/'execution/PROVENANCE.json')['created_utc']).timestamp()) + m['runtime']['global_seconds']
    if cutoff - time.time() < 180:
        raise RuntimeError('Insufficient original budget for migration')
    handle = open_process_handle(pid)
    confirmed = proc(pid)
    if confirmed is None or confirmed['start_ticks'] != identity['start_ticks']:
        os.close(handle)
        raise RuntimeError('Scheduler PID changed before pidfd acquisition')
    frozen = False
    terminated = False
    receipt = dict(created_utc=utc(), original=str(ORIGINAL), original_manifest_sha256=EXPECTED,
                   original_launcher=identity, original_global_deadline_utc=dt.datetime.fromtimestamp(cutoff,dt.timezone.utc).isoformat(),
                   original_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   workers=16, phase='preparing', events=[], children=[])
    try:
        send_process_signal(handle, signal.SIGSTOP)
        frozen = True
        for _ in range(100):
            s = proc(pid)
            if s and s['start_ticks'] == identity['start_ticks'] and s['state'] == 'T':
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Failed to confirm scheduler suspension')
        status = read(ORIGINAL/'execution/STATUS.json')
        save(HERE/'ORIGINAL_FROZEN_STATUS.json', status)
        consumed = {c['id'] for c in status['completed'] + status['active']}
        for line in (ORIGINAL/'execution/STATE_HISTORY.jsonl').read_text().splitlines():
            past = json.loads(line)
            consumed.update(c['id'] for c in past['completed'] + past['active'])
        consumed.update(p.name for p in (ORIGINAL/'execution/data').iterdir() if p.is_dir())
        for suffix in ('.stdout.log', '.stderr.log'):
            consumed.update(p.name[:-len(suffix)] for p in (ORIGINAL/'execution/logs').glob('*'+suffix))
        children = [s for s in processes() if s['ppid'] == pid]
        by_pid = {c['pid']:c for c in status['active']}
        for child in children:
            if child['pid'] != child['pgid'] or child['pid'] != child['sid']:
                raise RuntimeError('Unexpected original child process group')
            args = child['argv'] if child['state'] not in DEAD else by_pid.get(child['pid'],{}).get('command',[])
            if '--config' not in args or '--deadline-utc' not in args:
                raise RuntimeError('Cannot identify an original child safely')
            cfg = Path(args[args.index('--config')+1])
            if cfg.parent != ORIGINAL/'configs':
                raise RuntimeError('Original child configuration path mismatch')
            child['id'] = cfg.stem
            child['hard_deadline'] = min(cutoff, dt.datetime.fromisoformat(args[args.index('--deadline-utc')+1]).timestamp()+10)
            child['term_sent'] = child['kill_sent'] = False
            consumed.add(child['id'])
        pending = remaining_configs(m, consumed)
        known = {c['id'] for c in status['completed']} | {c['id'] for c in children}
        receipt.update(phase='draining', consumed_ids=sorted(consumed), pending_ids=[c['id'] for c in pending], children=children,
                       unknown_consumed_ids=sorted(consumed-known))
        m['configs'] = pending
        m['runtime']['workers'] = 16
        m['runtime']['absolute_deadline_utc'] = receipt['original_global_deadline_utc']
        m['dependency'] = None
        m['source_sha256']['launcher/launch.py'] = digest(lp)
        m['deployment']['continuation'] = dict(reason='User authorized 16 workers on DATA_PREP',original_manifest_sha256=EXPECTED,
                                               original_directory=str(ORIGINAL),consumed_ids=sorted(consumed))
        save(dest/'MANIFEST.json',m)
        if pending:
            checked = subprocess.run([sys.executable,str(lp),'--dry-run'],cwd=dest,check=True,capture_output=True,text=True)
            receipt['static_validation'] = json.loads(checked.stdout)
        save(receipt_path, receipt)
        print(json.dumps({'phase':'draining','pending':len(pending),'original_children':len(children)}), flush=True)
        while True:
            all_processes = processes()
            live = []
            for child in children:
                members = [s for s in all_processes if s['pgid'] == child['pid'] and s['state'] not in DEAD]
                leader = proc(child['pid'])
                if leader and leader['start_ticks'] != child['start_ticks']:
                    raise RuntimeError('Original child PID identity changed')
                if leader and leader['state'] in DEAD:
                    child['raw_wait_status'] = leader['exit_status']
                    child['returncode'] = os.waitstatus_to_exitcode(leader['exit_status'])
                if members:
                    live.append(child['id'])
                    for sig, flag, offset in ((signal.SIGTERM,'term_sent',8),(signal.SIGKILL,'kill_sent',2)):
                        if time.time() >= child['hard_deadline']-offset and not child[flag]:
                            try:
                                os.killpg(child['pid'],sig)
                            except ProcessLookupError:
                                continue
                            child[flag] = True
                            receipt['events'].append(dict(time_utc=utc(),pid=child['pid'],signal=sig.name,reason='original_wall_cap'))
            receipt.update(updated_utc=utc(), live_original_ids=live)
            save(receipt_path, receipt)
            if not live:
                break
            if time.time() > cutoff+2:
                raise RuntimeError('Original process group remains live beyond global cap')
            time.sleep(1)
        # Never resume the scheduler: it caches its dispatch decision before sleeps.
        # All scientific children/groups have exited; only this idle coordinator is killed.
        if pending:
            subprocess.run([sys.executable,str(lp),'--dry-run'],cwd=dest,check=True,capture_output=True,text=True)
        send_process_signal(handle, signal.SIGKILL)
        terminated = True
        for _ in range(100):
            state = proc(pid)
            if state is None or state['state'] in DEAD:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original scheduler exit not confirmed')
        receipt.update(phase='drained',scheduler_exit='SIGKILL after all original child groups exited',
                       original_status_is_historical=True,old_completed=status['completed'])
        save(receipt_path,receipt)
        if not pending:
            receipt.update(phase='nothing_pending',finished_utc=utc())
            save(receipt_path,receipt)
            return 0
        if cutoff-time.time() <= m['runtime']['diagnostic_reserve_seconds']+10:
            receipt.update(phase='budget_exhausted',finished_utc=utc())
            save(receipt_path,receipt)
            return 124
        receipt.update(phase='continuing',continuation=str(dest),manifest_sha256=digest(dest/'MANIFEST.json'),continued_utc=utc())
        save(receipt_path,receipt)
        print(json.dumps({'phase':'continuing','arms':len(pending),'workers':16}),flush=True)
        env=os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            env[key]='1'
        command=[sys.executable,str(lp)]
        code=subprocess.call(command,cwd=dest,env=env)
        receipt.update(phase='continuation_finished',continuation_returncode=code,finished_utc=utc())
        save(receipt_path,receipt)
        subprocess.run([sys.executable,str(dest/'analysis/summarize.py'),'--execution',str(dest/'execution'),'--out',str(dest/'execution/summary.json'),'--expected-arms',str(len(pending))],check=False)
        return code
    except BaseException as exc:
        receipt.update(phase='migration_failed',error=f'{type(exc).__name__}: {exc}',updated_utc=utc())
        save(receipt_path,receipt)
        raise
    finally:
        if frozen and not terminated:
            send_process_signal(handle,signal.SIGCONT)
        os.close(handle)

if __name__ == '__main__':
    raise SystemExit(main())
