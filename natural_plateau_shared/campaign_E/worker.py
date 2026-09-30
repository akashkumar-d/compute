#!/usr/bin/env python3
"""Campaign-E worker (transformer pilot): runs one session's share of the runs in manifest.json and pushes every finished run to GitHub.

A session needs only two commands (WORKER.md has the full instructions):

    python3 natural_plateau_shared/campaign_E/worker.py selftest --worker S3
    python3 natural_plateau_shared/campaign_E/worker.py watch --worker S3     # blocks ~9 min; repeat until it prints ALL DONE

`watch` starts the background daemon (`run`, fully detached) if it is not running, restarts it if it died, and prints progress.
The daemon runs the worker's runs from the manifest, each in its own process with one thread, as many at once as the session's
cores allow. Each manifest entry ("chain") is one training run. A development chain has a fixed configuration. A fresh-seed chain
lists the development chains of its cell under "after": it waits until they are finished (in any session), then applies PLAN.md's
rule to them and trains the chosen setting on its own fresh seeds, or records itself as skipped when the chosen setting did not
pass (PLAN.md).

Finished runs are compacted (7 significant digits), committed under results/<id>/ and pushed to the branch this session is on,
with status/<worker>-<instance>.json (also pushed every 30 minutes). Before starting a run the daemon fetches the other sessions'
branches (claude/*) and skips runs finished or being run elsewhere. When its own queue is empty it takes over runs from sessions
that stopped, or from the back half of a session far behind, and otherwise stands by until every run of the campaign is finished,
so that a session dying late is still covered. Everything in _work/ is local scratch (gitignored) and the source of truth for
re-publishing; a restart resumes each run from its last checkpoint (every 2000 steps). A run taken over from a stopped session
starts again from step 0 (checkpoints are not pushed).

Other commands:
    worker.py run --worker S3 [--cores N] [--no-push] [--no-steal]    the daemon (normally started by watch)
    worker.py status [--worker S3] [--all]                              --all fetches every session branch: campaign-wide view
    worker.py chain --id dev_L1_paper_s1_lw0.005 --worker S3 ...         one run (the daemon starts these)
"""
import argparse, datetime, glob, gzip, hashlib, json, math, os, platform, random, shutil, signal, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VERSION = 'campaign-E worker 1.0 (2026-09-30)'
K1 = 0.03 * 0.015
CKPT_EVERY = int(os.environ.get('CAMPAIGN_CKPT_EVERY', 2000))
HEARTBEAT_S = int(os.environ.get('CAMPAIGN_HEARTBEAT_S', 1800))    # status push while running or standing by
STALE_S = int(os.environ.get('CAMPAIGN_STALE_S', 5400))            # a status older than this is a session that has stopped
FETCH_MIN_S = int(os.environ.get('CAMPAIGN_FETCH_MIN_S', 300))     # at most one fetch of the other branches per this long
IDLE_S = int(os.environ.get('CAMPAIGN_IDLE_S', 600))               # ... and per this long while standing by
RETRY_PUSH_S = int(os.environ.get('CAMPAIGN_RETRY_PUSH_S', 300))   # retry a failed commit/push after this long
LOOP_S = float(os.environ.get('CAMPAIGN_LOOP_S', 15))
MAX_ATTEMPTS = 3        # a chain whose run raises an error this many times on one machine is recorded as FAILED
BROKEN_AFTER = 2        # a session where this many different chains raise errors stops taking chains (others take over)
MAX_KILLS = 5           # a chain whose process disappears without an error this many times counts as one error
BRANCH_PREFIXES = tuple(os.environ.get('CAMPAIGN_BRANCH_PREFIXES', 'claude/').split(','))
BASE_BRANCH = 'claude/focused-gauss-cy16t3'
WORK = os.environ.get('CAMPAIGN_WORK', os.path.join(HERE, '_work'))
RESULTS = os.path.join(HERE, 'results')
STATUS = os.path.join(HERE, 'status')
DEFAULT_MANIFEST = os.path.join(HERE, 'manifest.json')


# ------------------------------------------------------------------------------------------------------------------ small helpers
def now(): return time.time()
def iso(t=None): return datetime.datetime.fromtimestamp(now() if t is None else t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def hm(t): return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%H:%MZ') if t else 'never'
def parse_iso(s):
    try: return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception: return 0.0
def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def read_json(fn, default=None):
    try:
        with open(fn) as f: return json.load(f)
    except Exception: return default
def write_json(fn, obj):
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, fn)
def load_gz(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)
def log(msg):
    print(f'[{iso()}] {msg}', flush=True)
def touch(fn):
    with open(fn, 'w') as f: f.write(iso())
def tail(fn, n=40):
    try: return ''.join(open(fn).readlines()[-n:])
    except Exception: return '(no log)'

def effective_cores():
    """CPUs this process may use: the affinity mask, capped by a cgroup CPU quota if there is one."""
    try: n = len(os.sched_getaffinity(0))
    except Exception: n = os.cpu_count() or 1
    q = None
    try:
        a, b = open('/sys/fs/cgroup/cpu.max').read().split()[:2]
        if a != 'max': q = int(a) / int(b)
    except Exception:
        try:
            a = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read()); b = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
            if a > 0: q = a / b
        except Exception: pass
    if q: n = min(n, max(1, int(q + 0.01)))
    return max(1, n)

def load_manifest(path):
    m = read_json(path)
    if not m: raise SystemExit(f'cannot read the manifest {path}')
    m['_path'] = path; m['_sha'] = sha256(path)
    m['_by_id'] = {c['id']: c for c in m['chains']}
    m['_queue'] = {}
    for c in m['chains']: m['_queue'].setdefault(c['worker'], []).append(c)
    for w in m['_queue']: m['_queue'][w].sort(key=lambda c: c['order'])
    return m

def check_worker_id(man, W, helper=False):
    if W in man['workers'] or helper: return
    raise SystemExit(f"ERROR: unknown worker ID '{W}'. Use one of {', '.join(man['workers'])} exactly as written in your prompt.")

def check_clone_owner(W):
    """One worker ID per clone: _work/WORKER_ID records it on first use."""
    os.makedirs(WORK, exist_ok=True)
    fn = os.path.join(WORK, 'WORKER_ID')
    if os.path.exists(fn):
        old = open(fn).read().strip()
        if old and old != W:
            raise SystemExit(f"ERROR: this clone already runs worker {old}; it cannot also run {W}. Use --worker {old}, or tell the user.")
    else:
        open(fn, 'w').write(W)

def pid_alive(pid):
    try: os.kill(pid, 0)
    except Exception: return False
    try:
        st = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[0]
        return st != 'Z'
    except Exception: return True

def cmdline(pid):
    try: return open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace') + ' '
    except Exception: return ''

def chain_pid_alive(pid, cid):
    return pid_alive(pid) and f'--id {cid} ' in cmdline(pid)


# ------------------------------------------------------------------------------------------------------------------ git
class GitError(Exception): pass

class Git:
    def __init__(self, where):
        self.where = where
        r = self.run('rev-parse', '--show-toplevel', check=False)
        if r.returncode != 0: raise GitError('not inside a git clone of the repository')
        self.root = r.stdout.strip()
        self.rel = os.path.relpath(HERE, self.root)
        self.branch = self.run('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()

    def run(self, *args, timeout=240, check=True, binary=False):
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0')      # never prompt; the session's own credential setup is left alone
        try:
            r = subprocess.run(['git', *args], cwd=getattr(self, 'root', self.where), capture_output=True, text=not binary, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            if check: raise GitError(f'git {args[0]} timed out after {timeout}s')
            return subprocess.CompletedProcess(['git', *args], 124, b'' if binary else '', f'git {args[0]} timed out after {timeout}s')
        if check and r.returncode != 0: raise GitError(f"git {' '.join(args[:3])}: {(r.stderr or r.stdout).strip()[-600:]}")
        return r

    def identity_args(self):
        out = []
        if not self.run('config', 'user.name', check=False).stdout.strip(): out += ['-c', 'user.name=campaign-E worker']
        if not self.run('config', 'user.email', check=False).stdout.strip(): out += ['-c', 'user.email=noreply@anthropic.com']
        return out

    def remote_heads(self):
        r = self.run('ls-remote', '--heads', 'origin', timeout=90)
        out = {}
        for l in r.stdout.splitlines():
            if '\trefs/heads/' in l:
                sha, name = l.split('\trefs/heads/', 1)
                if name == self.branch or name.startswith(BRANCH_PREFIXES): out[name] = sha
        return out

    def fetch_changed(self, heads):
        """Fetch, one branch at a time, the session branches whose tip moved; returns the branches available locally and the errors."""
        ok, errs = [], []
        for b, sha in heads.items():
            loc = self.run('rev-parse', '-q', '--verify', f'refs/remotes/origin/{b}', check=False).stdout.strip()
            if loc != sha:
                r = self.run('fetch', '--no-tags', '--quiet', 'origin', f'+refs/heads/{b}:refs/remotes/origin/{b}', check=False, timeout=180)
                if r.returncode != 0: errs.append(f'{b}: {(r.stderr or r.stdout).strip()[-160:]}')
                if not self.run('rev-parse', '-q', '--verify', f'refs/remotes/origin/{b}', check=False).stdout.strip(): continue
            ok.append(b)
        return ok, errs

    def scan(self, branches):
        """Finished chains (results/<id>/chain.json or FAILED.json) and status files on each fetched branch."""
        done, failed, statuses = {}, {}, []
        for b in branches:
            ref = f'refs/remotes/origin/{b}'
            r = self.run('ls-tree', '-r', '--name-only', ref, '--', f'{self.rel}/results', f'{self.rel}/status', check=False)
            if r.returncode != 0: continue
            for path in r.stdout.splitlines():
                parts = path.split('/')
                if len(parts) >= 3 and parts[-3] == 'results' and parts[-1] == 'chain.json': done.setdefault(parts[-2], []).append(b)
                elif len(parts) >= 3 and parts[-3] == 'results' and parts[-1] == 'FAILED.json': failed.setdefault(parts[-2], []).append(b)
                elif len(parts) >= 2 and parts[-2] == 'status' and path.endswith('.json'):
                    s = self.run('show', f'{ref}:{path}', check=False)
                    if s.returncode == 0:
                        try: d = json.loads(s.stdout); d['_branch'] = b; statuses.append(d)
                        except Exception: pass
        return done, failed, statuses


# ------------------------------------------------------------------------------------------------------------------ one chain
def r7(x):
    if isinstance(x, float): return float(f'{x:.7g}') if math.isfinite(x) else x
    if isinstance(x, list): return [r7(v) for v in x]
    if isinstance(x, dict): return {k: r7(v) for k, v in x.items()}
    return x

def write_compact(src, dst):
    d = load_gz(src)
    d['loss'] = r7(d['loss']); d['obs'] = r7(d['obs'])
    d['compact'] = 'floats rounded to 7 significant digits by worker.py; full precision stayed in the session'
    tmp = dst + '.tmp'
    with gzip.open(tmp, 'wt', compresslevel=9) as f: json.dump(d, f, separators=(',', ':'))
    os.replace(tmp, dst)

def cfg_key(c):
    c = {k: v for k, v in c.items() if k != 'stop_at_test'}
    return json.dumps(c, sort_keys=True)

def first_cross(obs, thr):
    return next((o['step'] for o in obs if o.get('test_acc', 0) >= thr), None)

def first_persistent(obs, thr):
    te = [(o['step'], o['test_acc']) for o in obs if 'test_acc' in o]
    last_bad = max((i for i, (_, v) in enumerate(te) if v < thr), default=-1)
    return te[last_bad + 1][0] if last_bad + 1 < len(te) else None

def host_info():
    cpu = ''
    try: cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception: pass
    import numpy as np
    return dict(node=hashlib.sha256(platform.node().encode()).hexdigest()[:8], cpus=effective_cores(), cpu=cpu,
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform())

def dev_run(cid, man=None):
    """The compacted output of a finished development run: this session's copy, else the copy on any fetched session branch."""
    for fn in (os.path.join(WORK, cid, 'out', 'run.json.gz'), os.path.join(RESULTS, cid, 'run.json.gz')):
        if os.path.exists(fn): return load_gz(fn)
    g = Git(HERE)
    refs = g.run('for-each-ref', '--format=%(refname)', 'refs/remotes/origin/', check=False).stdout.split()
    for ref in sorted(refs):
        b = subprocess.run(['git', 'cat-file', 'blob', f'{ref}:{g.rel}/results/{cid}/run.json.gz'], cwd=g.root, capture_output=True, timeout=300)
        if b.returncode == 0 and b.stdout: return json.loads(gzip.decompress(b.stdout).decode())
    return None

def resolve_fresh(ch, man, find=dev_run):
    """PLAN.md: apply the rule to the finished development runs of this chain's cell (always their compacted copies, so that
    every session reaches the same choice)."""
    import tfm_analyze as A
    dev, missing = {}, []
    for cid in ch['after']:
        d = find(cid, man)
        if d is None: missing.append(cid)
        else: dev[cid] = A.fields(d)
    if not dev: return dict(chosen=None, passed=False, run=False, missing=missing, table={}, rule='PLAN.md')
    chosen, passed = A.select_setting(dev)
    table = {k: {kk: f.get(kk) for kk in ('primary', 'why_not', 'share', 'range_transition', 'final_test', 'CE_peak_over_logp')}
             for k, f in dev.items()}
    return dict(chosen=chosen, passed=passed, run=passed, missing=missing, table=table, rule='PLAN.md')

def run_chain(ch, man, worker, instance, branch, stolen_from, workdir, find=dev_run, force=False):
    """One campaign-E chain: a development run, or a fresh-seed run of the setting chosen by PLAN.md's rule (or a skip)."""
    import tfm_runner as NC
    os.makedirs(os.path.join(workdir, 'out'), exist_ok=True)
    if os.path.exists(os.path.join(workdir, 'FAILED.json')): os.remove(os.path.join(workdir, 'FAILED.json'))
    cfg = ch['cfg']; sel = None; dial = ch.get('dial')
    if ch.get('after'):
        sel = resolve_fresh(ch, man, find)
        if force and sel['chosen'] and not sel['run']: sel['run'] = True; sel['forced'] = True      # selftest only
        if sel['run']:
            base = man['_by_id'][sel['chosen']]
            cfg = dict(base['cfg'], init_seed=ch['init_seed'], split_seed=ch['split_seed']); dial = base.get('dial')
    meta = dict(id=ch['id'], stage=ch['stage'], cell=ch['cell'], opt=ch['opt'], n_layers=ch['n_layers'], p=ch['p'], dial=dial,
                seed=ch['seed'], init_seed=ch['init_seed'], split_seed=ch['split_seed'], selection=sel,
                owner=ch['worker'], ran_by=worker, instance=instance, branch=branch, stolen_from=stolen_from, started=iso(),
                version=VERSION, host=host_info(), threads=int(os.environ.get('OMP_NUM_THREADS', '1')),
                code=dict(worker=sha256(os.path.join(HERE, 'worker.py')), runner=sha256(os.path.join(HERE, 'tfm_runner.py')),
                          rule=sha256(os.path.join(HERE, 'tfm_analyze.py')), manifest=man['_sha']))
    if sel is not None and not sel['run']:
        meta.update(skipped=True, finished=iso(), done=True,
                    reason='no development run of this cell was available' if sel['chosen'] is None else
                           f"the chosen development setting ({sel['chosen']}) did not pass, so PLAN.md runs no fresh seeds for this cell")
        meta['files'] = {}
        write_json(os.path.join(workdir, 'out', 'chain.json'), meta); write_json(os.path.join(workdir, 'chain.json'), meta)
        return meta
    out = os.path.join(workdir, 'run.json.gz')
    if not os.path.exists(out): NC.run(cfg, out, CKPT_EVERY)
    M = load_gz(out)
    assert json.dumps(M['cfg'], sort_keys=True) == json.dumps(cfg, sort_keys=True), 'output does not match its configuration'
    t10 = first_cross(M['obs'], 0.1); t90 = first_persistent(M['obs'], 0.9)
    meta['cfg'] = cfg
    meta['run'] = dict(horizon=cfg['steps'], steps=len(M['loss']) - 1, stopped_at=M.get('stopped_at'), stop_reason=M.get('stop_reason'),
                       seconds=round(M['seconds'], 1), resumed_from=M.get('resumed_from', 0), t10=t10, t90=t90,
                       final_test=M['obs'][-1]['test_acc'], final_train=M['obs'][-1]['train_acc'], final_loss=M['loss'][-1])
    dst = os.path.join(workdir, 'out', 'run.json.gz'); write_compact(out, dst)
    meta['files'] = {'run.json.gz': dict(sha256=sha256(dst), bytes=os.path.getsize(dst))}
    meta['finished'] = iso(); meta['done'] = True
    write_json(os.path.join(workdir, 'out', 'chain.json'), meta)
    write_json(os.path.join(workdir, 'chain.json'), meta)       # local done marker
    return meta


# ------------------------------------------------------------------------------------------------------------------ progress
def progress(workdir):
    """(phase, step) of a run from its outputs and checkpoint."""
    if os.path.exists(os.path.join(workdir, 'chain.json')): return 'done', None
    if os.path.exists(os.path.join(workdir, 'run.json.gz')): return 'compacting', None
    fn = os.path.join(workdir, 'run.json.gz.ckpt.npz')
    if not os.path.exists(fn): return 'run', 0
    try:
        import numpy as np
        with np.load(fn, allow_pickle=False) as z: return 'run', int(z['t_next'])
    except Exception: return 'run', 0


# ------------------------------------------------------------------------------------------------------------------ daemon
class Daemon:
    def __init__(self, man, worker, cores, push=True, steal=True):
        self.man, self.W, self.push_on, self.steal_on = man, worker, push, steal
        self.cores = cores or effective_cores()
        self.git = Git(HERE)
        gd = self.git.run('rev-parse', '--git-dir').stdout.strip()
        gd = gd if os.path.isabs(gd) else os.path.join(self.git.root, gd)
        if os.path.exists(os.path.join(gd, 'rebase-merge')) or os.path.exists(os.path.join(gd, 'rebase-apply')):
            self.git.run('rebase', '--abort', check=False)       # a rebase left half-done by a killed daemon
            self.git.branch = self.git.run('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
        if self.git.branch == 'HEAD': raise SystemExit('detached HEAD: check out this session\'s branch first')
        os.makedirs(WORK, exist_ok=True); os.makedirs(RESULTS, exist_ok=True); os.makedirs(STATUS, exist_ok=True)
        self.inst = instance_id(worker)
        self.state_fn = os.path.join(WORK, f'{worker}.state.json')
        self.st = read_json(self.state_fn, {}) or {}
        for k, v in dict(failures={}, kills={}, failed=[], abandoned=[], stolen={}, started=iso(), push_ok_at=0, push_err=None,
                         push_err_since=0, commit_err=None, fetch_ok_at=0, fetch_err=None, last_sync=0, commits=0, broken=None).items():
            self.st.setdefault(k, v)
        self.procs = {}
        self.remote = dict(done={}, failed={}, statuses=[], at=0, ok_at=0, branches=[])
        self.speed = {}
        self.finished = False; self.idle = False

    # ---- bookkeeping
    def save(self): write_json(self.state_fn, self.st)
    def local_done(self): return {os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(WORK, '*', 'chain.json'))}
    def fresh(self, s): return now() - parse_iso(s.get('updated', '')) < STALE_S
    def others(self):
        """Status files of other live sessions (a previous instance of this worker on this branch is this session restarted)."""
        out = []
        for s in self.remote['statuses']:
            if s.get('instance') == self.inst: continue
            if s.get('worker') == self.W and s.get('_branch') == self.git.branch: continue
            if not self.fresh(s) or s.get('state') in ('finished', 'broken'): continue
            out.append(s)
        return out
    def claims(self, cid): return [(s['worker'], s['instance']) for s in self.others() if cid in (s.get('running') or {})]
    def alive(self, w): return any(s.get('worker') == w for s in self.others())
    def finished_set(self):
        return self.local_done() | set(self.remote['done']) | set(self.st['failed']) | set(self.remote['failed'])
    def all_finished(self):
        fin = self.finished_set(); return all(cid in fin for cid in self.man['_by_id'])
    def git_ok(self): return not self.push_on or (not self.st['push_err'] and not self.st['commit_err'] and self.st['push_ok_at'] >= self.st['last_sync'] - 1)

    # ---- remote view
    def refresh(self, min_interval):
        """Fetch the session branches that moved and rescan; returns True when the view was renewed now."""
        if now() - self.remote['at'] < min_interval: return False
        self.remote['at'] = now()
        try:
            heads = self.git.remote_heads()
            br, errs = self.git.fetch_changed(heads)
            done, failed, statuses = self.git.scan(br)
            self.remote.update(done=done, failed=failed, statuses=statuses, branches=br, ok_at=now())
            self.st['fetch_ok_at'] = now(); self.st['fetch_err'] = ('; '.join(errs))[-300:] if errs else None
            return True
        except Exception as e:
            self.st['fetch_err'] = str(e)[-300:]; log(f'fetch failed (continuing with the last view): {e}')
            return False

    # ---- scheduling
    def queue(self, w): return self.man['_queue'].get(w, [])
    def threads(self, c): return min(c['threads'], self.cores)
    def available(self, c, fin):
        if any(d not in fin for d in c.get('after', ())): return False      # a fresh-seed chain waits for its cell's development runs
        return c['id'] not in fin and c['id'] not in self.procs and c['id'] not in self.st['abandoned'] and not self.claims(c['id'])

    def pick(self, free):
        if self.st['broken']: return None, None
        fin = self.finished_set()
        own = [c for c in self.queue(self.W) if self.available(c, fin)]
        if own:
            c = own[0]
            return (c, None) if self.threads(c) <= free else (None, None)
        if not self.steal_on: return None, None
        victims = []
        workers = self.man['workers'] + sorted({s.get('worker') for s in self.remote['statuses']} - set(self.man['workers']) - {None})
        mine = lambda cs: min(cs, key=lambda c: hashlib.sha256((self.W + c['id']).encode()).hexdigest())   # each thief its own order
        fits = lambda cs: [c for c in cs if self.threads(c) <= free]     # take only what fits in the free cores now
        for w in workers:
            if w == self.W: continue
            vq = [c for c in self.queue(w) if self.available(c, fin)]
            if not vq: continue
            cores_w = next((s.get('cores', 4) for s in self.others() if s.get('worker') == w), 4) or 4
            rem = sum(c['est_core_hours'] for c in vq) / cores_w
            if self.alive(w):
                if len(vq) < 2 or rem < 1.0: continue          # a live session finishes its own last chains
                cand = fits(vq[len(vq) // 2:])                   # from the back half of its queue; it works from the front
                if cand: victims.append((1, -rem, w, mine(cand)))
            else:
                cand = fits(vq)                                  # a stopped, broken or never started session
                if cand: victims.append((0, -rem, w, mine(cand)))
        if not victims: return None, None
        victims.sort(key=lambda v: (v[0], v[1], v[2]))
        return victims[0][3], victims[0][2]

    def start(self, c, stolen_from):
        wd = os.path.join(WORK, c['id']); os.makedirs(wd, exist_ok=True)
        n = self.threads(c)
        env = dict(os.environ)
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[v] = str(n)
        cmd = [sys.executable, os.path.abspath(__file__), 'chain', '--id', c['id'], '--worker', self.W, '--instance', self.inst,
               '--branch', self.git.branch, '--manifest', self.man['_path']] + (['--stolen-from', stolen_from] if stolen_from else [])
        with open(os.path.join(wd, 'chain.log'), 'a') as lf:
            lf.write(f'\n[{iso()}] start ({n} thread(s), instance {self.inst})\n')
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True, stdin=subprocess.DEVNULL)
        open(os.path.join(wd, 'pid'), 'w').write(str(p.pid))
        self.procs[c['id']] = dict(popen=p, pid=p.pid, threads=n, started=now(), stolen=stolen_from)
        if stolen_from: self.st['stolen'][c['id']] = stolen_from
        log(f"start {c['id']} ({n} thread(s))" + (f' taken from {stolen_from}' if stolen_from else ''))

    def adopt(self):
        """Chains still running from a previous daemon (the processes outlive it)."""
        for pf in glob.glob(os.path.join(WORK, '*', 'pid')):
            cid = os.path.basename(os.path.dirname(pf))
            try: pid = int(open(pf).read().strip())
            except Exception: continue
            if cid in self.man['_by_id'] and chain_pid_alive(pid, cid):
                c = self.man['_by_id'][cid]
                self.procs[cid] = dict(popen=None, pid=pid, threads=self.threads(c), started=now(), stolen=self.st['stolen'].get(cid))
                log(f'adopted running chain {cid} (pid {pid})')

    def is_running(self, cid, pr):
        if pr['popen'] is not None: return pr['popen'].poll() is None
        return chain_pid_alive(pr['pid'], cid)

    def kill(self, pr):
        try: os.killpg(pr['pid'], signal.SIGTERM)
        except Exception:
            try: os.kill(pr['pid'], signal.SIGTERM)
            except Exception: pass

    def abandon(self, cid, why):
        pr = self.procs.pop(cid, None)
        if pr: self.kill(pr)
        if cid not in self.st['abandoned']: self.st['abandoned'].append(cid)
        self.save(); shutil.rmtree(os.path.join(WORK, cid), ignore_errors=True)
        log(f'abandon {cid}: {why}')

    def check_stolen(self):
        """After each fresh view: give up a taken-over chain if its owner, an earlier thief, or a finished copy exists."""
        out = []
        for cid, pr in list(self.procs.items()):
            if not pr['stolen']: continue
            owner = self.man['_by_id'][cid]['worker']
            rivals = [(w, i) for w, i in self.claims(cid) if w == owner or i < self.inst]
            if cid in self.remote['done'] or rivals:
                self.abandon(cid, f'also claimed by {rivals}' if rivals else 'finished elsewhere'); out.append(cid)
        return out

    # ---- publishing
    def status_doc(self):
        running = {}
        for cid, pr in self.procs.items():
            ph, step = progress(os.path.join(WORK, cid)); c = self.man['_by_id'][cid]
            running[cid] = dict(phase=ph, step=step, steps=c['cfg']['steps'], threads=pr['threads'], stolen_from=pr['stolen'],
                                since=iso(pr['started']), steps_per_s=round(self.speed.get(cid, (0, 0, 0, 0.0))[3], 2))
        fin = self.finished_set()
        own = self.queue(self.W)
        left = [c for c in own if c['id'] not in fin]
        state = 'broken' if self.st['broken'] else 'finished' if self.finished else 'standby' if (self.idle and not self.procs) else 'running'
        return dict(worker=self.W, instance=self.inst, branch=self.git.branch, cores=self.cores, version=VERSION, manifest_sha256=self.man['_sha'],
                    state=state, broken=self.st['broken'], started=self.st['started'], updated=iso(), running=running,
                    done=sorted(self.local_done()), failed=sorted(self.st['failed']), abandoned=sorted(self.st['abandoned']),
                    stolen=self.st['stolen'], own_total=len(own), own_left=len(left),
                    campaign_left=sum(1 for cid in self.man['_by_id'] if cid not in fin),
                    est_hours_left=round(sum(c['est_core_hours'] for c in left) / self.cores, 2), host=host_info(),
                    push_error=self.st['push_err'], commit_error=self.st['commit_err'], fetch_error=self.st['fetch_err'])

    def sync(self, msg):
        """Copy every locally finished chain into results/, write the status file, commit and push. Idempotent; never raises."""
        try:
            rel = self.git.rel
            head_has = set()
            r = self.git.run('ls-tree', '-r', '--name-only', 'HEAD', '--', f'{rel}/results', check=False)
            for p in r.stdout.splitlines():
                if p.endswith('/chain.json') or p.endswith('/FAILED.json'): head_has.add(p.split('/')[-2])
            for cid in sorted(self.local_done()):
                if cid in head_has: continue
                src = os.path.join(WORK, cid, 'out'); dst = os.path.join(RESULTS, cid)
                if not os.path.exists(os.path.join(src, 'chain.json')): continue
                os.makedirs(dst, exist_ok=True)
                for f in ('run.json.gz', 'chain.json'):                  # a skipped fresh chain has only chain.json
                    if os.path.exists(os.path.join(src, f)): shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            for cid in self.st['failed']:
                fsrc = os.path.join(WORK, cid, 'FAILED.json')
                if cid in head_has or not os.path.exists(fsrc): continue
                os.makedirs(os.path.join(RESULTS, cid), exist_ok=True); shutil.copy(fsrc, os.path.join(RESULTS, cid, 'FAILED.json'))
            self.st['last_sync'] = now()
            write_json(os.path.join(STATUS, f'{self.W}-{self.inst}.json'), self.status_doc())
            self.git.run('add', '--', f'{rel}/results', f'{rel}/status')
            paths = self.git.run('diff', '--cached', '--name-only', '--', f'{rel}/results', f'{rel}/status').stdout.split()
            if paths:
                self.git.run(*self.git.identity_args(), 'commit', '-q', '-m', f'campaign E {self.W}: {msg}', '--', *paths)
                self.st['commits'] += 1
            self.st['commit_err'] = None
        except Exception as e:
            self.st['commit_err'] = str(e)[-300:]; self.save(); log(f'commit failed: {e}'); return
        self.save()
        if self.push_on: self.push()

    def push(self):
        """Push HEAD to this session's branch. On a non-fast-forward rejection, rebase on the remote branch (our side wins a
        conflict: both sides are results of the same chain) and retry; any other failure is recorded and retried later."""
        b = self.git.branch; last = ''
        try:
            for attempt in range(3):
                r = self.git.run('push', '-q', 'origin', f'HEAD:refs/heads/{b}', check=False, timeout=300)
                if r.returncode == 0:
                    self.st['push_ok_at'] = now(); self.st['push_err'] = None; self.st['push_err_since'] = 0; self.save(); return True
                last = (r.stderr or r.stdout).strip(); low = last.lower()
                if any(k in low for k in ('non-fast-forward', 'fetch first', '[rejected]')) and not any(k in low for k in ('denied', '403', 'protected')):
                    f = self.git.run('fetch', '--no-tags', '--quiet', 'origin', f'+refs/heads/{b}:refs/remotes/origin/{b}', check=False, timeout=300)
                    if f.returncode != 0: last = 'fetch before rebase failed: ' + (f.stderr or f.stdout).strip()[-300:]; break
                    rr = self.git.run('-c', 'rebase.autoStash=true', *self.git.identity_args(), 'rebase', '-X', 'theirs', f'origin/{b}', check=False, timeout=300)
                    if rr.returncode != 0:
                        self.git.run('rebase', '--abort', check=False); last = 'rebase failed: ' + (rr.stderr or rr.stdout).strip()[-300:]; break
                    continue
                break
        except Exception as e:
            last = f'{last} | {e}'
        self.st['push_err'] = last[-400:] or 'unknown push error'
        if not self.st['push_err_since']: self.st['push_err_since'] = now()
        self.save(); log(f'push failed (will retry): {self.st["push_err"]}')
        return False

    # ---- main loop
    def reap(self, reasons):
        for cid, pr in list(self.procs.items()):
            if self.is_running(cid, pr): continue
            del self.procs[cid]
            wd = os.path.join(WORK, cid)
            try: os.remove(os.path.join(wd, 'pid'))
            except Exception: pass
            if os.path.exists(os.path.join(wd, 'chain.json')):
                m = read_json(os.path.join(wd, 'chain.json'), {}); mm = m.get('run', {})
                reasons.append(f"{cid} done (t10 {mm.get('t10')}, x {mm.get('x')}, final test {mm.get('final_test')}, {mm.get('steps')} steps)")
                log('done ' + reasons[-1]); continue
            if cid in self.st['abandoned']: continue
            err = os.path.exists(os.path.join(wd, 'FAILED.json'))
            tl = tail(os.path.join(wd, 'chain.log'), 25)
            if not err:
                self.st['kills'][cid] = self.st['kills'].get(cid, 0) + 1
                log(f'{cid} stopped without an error message ({self.st["kills"][cid]}x); it will be restarted')
                if self.st['kills'][cid] < MAX_KILLS: self.save(); continue
                self.st['kills'][cid] = 0
            n = self.st['failures'][cid] = self.st['failures'].get(cid, 0) + 1
            log(f'{cid} raised an error ({n}x); log tail:\n{tl}')
            if pr['stolen']:
                self.abandon(cid, 'it failed here; left to its owner or another session'); reasons.append(f'{cid} failed here, left to others')
            elif n >= MAX_ATTEMPTS:
                fj = read_json(os.path.join(wd, 'FAILED.json'), {}) or {}
                write_json(os.path.join(wd, 'FAILED.json'), dict(fj, id=cid, worker=self.W, instance=self.inst, attempts=n, at=iso(), log_tail=tl[-1500:]))
                self.st['failed'].append(cid); reasons.append(f'{cid} FAILED after {n} errors')
            bad = sorted(c for c, k in self.st['failures'].items() if k > 0)
            if len(bad) >= BROKEN_AFTER and not self.st['broken']:
                self.st['broken'] = f"{len(bad)} different chains raised errors here ({', '.join(bad[:4])}); last log lines: {tl[-600:]}"
                reasons.append('BROKEN: stops taking chains; other sessions take over its queue'); log(self.st['broken'])
            self.save()

    def tick(self):
        reasons = []
        self.reap(reasons)
        # progress speed (for the status): average steps/s since the current phase was first seen
        for cid in self.procs:
            ph, step = progress(os.path.join(WORK, cid))
            if step is None: continue
            sp = self.speed.get(cid)
            if sp is None or sp[0] != ph or step < sp[2]: self.speed[cid] = (ph, now(), step, 0.0); continue
            if step > sp[2]: self.speed[cid] = (ph, sp[1], sp[2], (step - sp[2]) / max(1.0, now() - sp[1]))
        # renew the view of the other sessions now and then; drop taken-over chains that someone with priority also runs
        if self.refresh(IDLE_S if (self.idle and not self.procs) else FETCH_MIN_S):
            gone = self.check_stolen()
            if gone: reasons.append('left ' + ', '.join(gone) + ' to other sessions')
        # start chains while cores are free
        started = []
        while not self.st['broken']:
            free = self.cores - sum(pr['threads'] for pr in self.procs.values())
            if free <= 0: break
            c, v = self.pick(free)
            if c is None: break
            self.start(c, v); started.append(c['id'] + (f' (from {v})' if v else ''))
        if started: reasons.append('start ' + ', '.join(started)); self.idle = False
        if not self.procs:
            if self.all_finished() or self.st['broken']:
                if self.git_ok(): self.finished = True; reasons.append('finished: every chain of the campaign is finished' if not self.st['broken'] else 'stopped (broken)')
                elif now() - self.st['last_sync'] >= RETRY_PUSH_S: reasons.append('retry push before finishing')
            elif not self.idle:
                self.idle = True; reasons.append('standing by: own chains finished, others still running elsewhere')
        if not reasons and now() - self.st['last_sync'] >= HEARTBEAT_S: reasons.append('heartbeat')
        if not reasons and (self.st['push_err'] or self.st['commit_err']) and now() - self.st['last_sync'] >= RETRY_PUSH_S: reasons.append('retry push')
        if reasons: self.sync('; '.join(reasons))
        if self.finished and not self.git_ok(): self.finished = False      # the final push failed: keep going and retry

    def loop(self):
        import fcntl
        lockf = open(os.path.join(WORK, 'daemon.lock'), 'w')
        try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: raise SystemExit('another worker daemon is already running in this clone')
        alive_fn = os.path.join(WORK, f'{self.W}.alive'); touch(alive_fn)
        write_json(os.path.join(WORK, f'{self.W}.daemon.pid'), dict(pid=os.getpid(), started=iso(), cores=self.cores))
        log(f'{VERSION}: worker {self.W}, instance {self.inst}, branch {self.git.branch}, {self.cores} cores, push {"on" if self.push_on else "off"}')
        self.adopt()
        try:
            self.refresh(0); self.sync('start' if not self.procs else 'restart')
        except Exception: log('startup error (continuing):\n' + traceback.format_exc())
        while not self.finished:
            touch(alive_fn)
            try: self.tick()
            except Exception: log('tick error:\n' + traceback.format_exc())
            touch(alive_fn)
            if not self.finished: time.sleep(LOOP_S)
        log('finished: ' + ('this session is broken; others take over its chains' if self.st['broken'] else 'every chain of the campaign is finished'))


def instance_id(worker):
    fn = os.path.join(WORK, f'{worker}.instance')
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(fn): return open(fn).read().strip()
    i = '%06x' % random.SystemRandom().getrandbits(24)
    open(fn, 'w').write(i); return i


# ------------------------------------------------------------------------------------------------------------------ status views
def local_status_text(man, W):
    st = read_json(os.path.join(WORK, f'{W}.state.json'), {}) or {}
    inst = open(os.path.join(WORK, f'{W}.instance')).read().strip() if os.path.exists(os.path.join(WORK, f'{W}.instance')) else '?'
    s = read_json(os.path.join(STATUS, f'{W}-{inst}.json'), {}) or {}
    q = man['_queue'].get(W, [])
    done_local = {os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(WORK, '*', 'chain.json'))}
    lines = [f"campaign E | worker {W} (instance {inst}) | branch {s.get('branch', '?')} | {s.get('cores', '?')} cores | state {s.get('state', 'starting')} | {iso()}"]
    run = s.get('running') or {}
    pids = {}
    for pf in glob.glob(os.path.join(WORK, '*', 'pid')):
        cid = os.path.basename(os.path.dirname(pf))
        try:
            pid = int(open(pf).read().strip())
            if chain_pid_alive(pid, cid): pids[cid] = pid
        except Exception: pass
    if pids:
        parts = []
        for cid in sorted(pids):
            ph, step = progress(os.path.join(WORK, cid)); c = man['_by_id'].get(cid, {})
            steps = c.get('cfg', {}).get('steps', 0); sp = (run.get(cid) or {}).get('steps_per_s') or 0
            if sp and step is not None: eta = f', at most ~{(steps - step) / sp / 3600:.1f} h left'
            else: eta = ''
            parts.append(f"{cid} {ph} {step or 0:,}/{steps:,}{eta}")
        lines.append(f'running ({len(pids)}): ' + ' | '.join(parts))
    else:
        lines.append('running: none')
    own_done = [c for c in q if c['id'] in done_local]
    lines.append(f"own queue: {len(own_done)}/{len(q)} done here, {s.get('own_left', '?')} not finished anywhere (~{s.get('est_hours_left', '?')} h at the manifest's estimate); "
                 f"campaign: {s.get('campaign_left', '?')} of {len(man['chains'])} chains not finished; taken over here: {len(st.get('stolen', {}))}; failed here: {len(st.get('failed', []))}")
    if s.get('state') == 'standby':
        lines.append('standing by: this session\'s own chains are done; it takes over any chain whose session stops, and finishes when the whole campaign is finished')
    if st.get('broken'): lines.append(f"ERROR: chains keep failing on this machine: {st['broken'][:700]}")
    if st.get('commit_err'): lines.append(f"COMMIT FAILING: {st['commit_err'][:300]}")
    pe = st.get('push_err')
    lines.append(f"git: last push {hm(st.get('push_ok_at'))}" + (f" | PUSH FAILING since {hm(st.get('push_err_since'))}: {pe[:200]}" if pe else ' ok')
                 + f" | last look at other sessions {hm(st.get('fetch_ok_at'))}" + (f" (error: {st['fetch_err'][:160]})" if st.get('fetch_err') else ''))
    return '\n'.join(lines), s, st


def campaign_status(man, fetch=True):
    g = Git(HERE)
    if fetch:
        heads = g.remote_heads(); br, errs = g.fetch_changed(heads)
    else:
        br, errs = [], []
    done, failed, statuses = g.scan(br)
    out = [f'campaign E: {len(set(done))}/{len(man["chains"])} runs finished, {len(set(failed))} failed | session branches scanned: {len(br)} | {iso()}']
    if errs: out.append('  fetch errors: ' + '; '.join(errs)[:400])
    by_w = {}
    for s in statuses: by_w.setdefault(s.get('worker'), []).append(s)
    for w in man['workers'] + sorted(set(by_w) - set(man['workers']) - {None}):
        q = man['_queue'].get(w, [])
        d = sum(1 for c in q if c['id'] in done)
        ss = sorted(by_w.get(w, []), key=lambda s: s.get('updated', ''))
        if ss:
            s = ss[-1]; age = (now() - parse_iso(s.get('updated', ''))) / 60
            flag = s.get('state', '?') if s.get('state') in ('finished', 'broken') else (s.get('state', 'running').upper() if age * 60 < STALE_S else 'STOPPED?')
            run = ', '.join(f"{k} {v.get('phase')} {(v.get('step') or 0) // 1000}k" for k, v in (s.get('running') or {}).items())
            out.append(f"  {w}: {d}/{len(q)} own chains finished | {flag}, status {age:.0f} min old, branch {s.get('_branch')} | running: {run or '-'}"
                       + (f" | PUSH ERROR {s['push_error'][:80]}" if s.get('push_error') else '') + (f" | BROKEN {str(s.get('broken'))[:80]}" if s.get('broken') else ''))
        else:
            out.append(f'  {w}: {d}/{len(q)} own chains finished | no status file yet (not started?)')
    by_stage = {}
    for c in man['chains']: by_stage.setdefault(c['stage'], [0, 0]); by_stage[c['stage']][1] += 1; by_stage[c['stage']][0] += c['id'] in done
    out.append('  by stage: ' + ', '.join(f'{k} {v[0]}/{v[1]}' for k, v in sorted(by_stage.items())))
    return '\n'.join(out)


# ------------------------------------------------------------------------------------------------------------------ commands
def daemon_info(W):
    d = read_json(os.path.join(WORK, f'{W}.daemon.pid'), {}) or {}
    pid = d.get('pid'); cl = cmdline(pid) if pid else ''
    if pid and pid_alive(pid) and 'worker.py' in cl and ' run ' in cl and f'--worker {W} ' in cl: return pid, parse_iso(d.get('started', ''))
    return None, 0.0

def start_daemon(W, args_saved):
    """Start the daemon fully detached (double fork), so it is not a child of this watch process or of the calling shell."""
    os.makedirs(WORK, exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), 'run', '--worker', W] + args_saved
    logfn = os.path.join(WORK, f'{W}.daemon.log')
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            if os.fork() != 0: os._exit(0)
            fd = os.open(logfn, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644); nul = os.open(os.devnull, os.O_RDONLY)
            os.dup2(nul, 0); os.dup2(fd, 1); os.dup2(fd, 2); os.chdir(HERE)
            os.closerange(3, 65536)          # drop every inherited descriptor (e.g. the calling tool's output pipe)
            os.execv(sys.executable, cmd)
        finally:
            os._exit(1)
    os.waitpid(pid, 0)
    for _ in range(20):
        time.sleep(0.5)
        p, _t = daemon_info(W)
        if p: return p
    return None

def next_command(W):
    try: rel = os.path.relpath(os.path.abspath(__file__), Git(HERE).root)
    except Exception: rel = os.path.abspath(__file__)
    return f'python3 {rel} watch --worker {W}'

def cmd_watch(a):
    man = load_manifest(a.manifest); W = a.worker
    check_worker_id(man, W, a.helper); check_clone_owner(W)
    saved_fn = os.path.join(WORK, f'{W}.args.json')
    saved = read_json(saved_fn)
    if saved is None or a.cores or a.no_push or a.no_steal or a.manifest != DEFAULT_MANIFEST:
        saved = ([] if not a.cores else ['--cores', str(a.cores)]) + (['--no-push'] if a.no_push else []) + (['--no-steal'] if a.no_steal else []) \
                + (['--manifest', a.manifest] if a.manifest != DEFAULT_MANIFEST else [])
        write_json(saved_fn, saved)
    restarts = 0; rechecked = False
    def state(): return (local_status_text(man, W)[1] or {}).get('state')
    def git_clean():
        st = read_json(os.path.join(WORK, f'{W}.state.json'), {}) or {}
        return '--no-push' in saved or (not st.get('push_err') and not st.get('commit_err') and st.get('push_ok_at', 0) >= st.get('last_sync', 0) - 1)
    def done_flag(): return state() in ('finished', 'broken') and not daemon_info(W)[0] and rechecked and git_clean()
    def ensure():
        nonlocal restarts, rechecked
        pid, started = daemon_info(W)
        if pid:
            alive = os.path.join(WORK, f'{W}.alive')
            last = max(os.path.getmtime(alive) if os.path.exists(alive) else 0.0, started)
            if now() - last <= 1800: return
            print(f'daemon unresponsive for 30 min: restarting it (pid {pid})', flush=True)
            try: os.kill(pid, signal.SIGTERM)
            except Exception: pass
            time.sleep(5)
        if state() in ('finished', 'broken') and git_clean():
            if rechecked: return
            rechecked = True       # one fresh look per watch call: a session that stopped may have left chains to take over
            if state() == 'broken': return
        restarts += 1
        if restarts > 5: print('ERROR: the daemon keeps stopping; last lines of its log:\n' + tail(os.path.join(WORK, f'{W}.daemon.log')), flush=True); return
        pid = start_daemon(W, saved)
        print((f'started the worker daemon (pid {pid})' + (' again' if restarts > 1 else '')) if pid else 'checked the campaign again with a short daemon run', flush=True)
    ensure()
    txt, s, st = local_status_text(man, W); print(txt, flush=True)
    t_end = now() + 60 * a.minutes
    while now() < t_end:
        time.sleep(min(30, max(1, t_end - now())))
        if done_flag(): break
        ensure()
    txt, s, st = local_status_text(man, W)
    print('---\n' + txt)
    if done_flag() and state() == 'broken':
        print(f'STOP: chains keep failing on this machine (see the ERROR line). Tell the user; the other sessions take over {W}\'s chains.')
    elif done_flag():
        print(f'ALL DONE for {W}: every chain of the campaign is finished and this session\'s results are pushed. You can stop now.')
    else:
        print(f'Still running. Next, run exactly: {next_command(W)}')

def cmd_run(a):
    man = load_manifest(a.manifest)
    check_worker_id(man, a.worker, a.helper); check_clone_owner(a.worker)
    Daemon(man, a.worker, a.cores, push=not a.no_push, steal=not a.no_steal).loop()

def cmd_chain(a):
    man = load_manifest(a.manifest); ch = man['_by_id'][a.id]
    wd = os.path.join(WORK, a.id)
    try:
        m = run_chain(ch, man, a.worker, a.instance, a.branch, a.stolen_from, wd, force=bool(ch.get('test_force')))   # test_force: test manifests only
        if m.get('skipped'): print(f"[{iso()}] chain {a.id} skipped: {m['reason']}", flush=True)
        else: print(f"[{iso()}] run {a.id} done: t10 {m['run']['t10']}, t90 {m['run']['t90']}, final test {m['run']['final_test']}, {m['run']['steps']} steps", flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        write_json(os.path.join(wd, 'FAILED.json'), dict(id=a.id, worker=a.worker, instance=a.instance, at=iso(), error=tb[-3000:]))
        sys.exit(1)

def cmd_status(a):
    man = load_manifest(a.manifest)
    if a.all: print(campaign_status(man))
    if a.worker: print(local_status_text(man, a.worker)[0])

def cmd_selftest(a):
    import importlib.util, tempfile
    ok = True; man = load_manifest(a.manifest); W = a.worker
    check_worker_id(man, W, a.helper)
    def check(name, f):
        nonlocal ok
        t = now()
        try: msg = f() or ''; print(f'PASS  {name} ({now() - t:.0f}s) {msg}', flush=True)
        except Exception as e: ok = False; print(f'FAIL  {name}: {e}\n{traceback.format_exc()[-1200:]}', flush=True)
    def py():
        import numpy as np
        assert sys.version_info >= (3, 8), 'python >= 3.8 needed'
        return f'python {platform.python_version()}, numpy {np.__version__}, {effective_cores()} usable cores'
    def clone():
        check_clone_owner(W); return f'this clone runs worker {W}'
    def hashes():
        H = read_json(os.path.join(HERE, 'HASHES.json'))
        if not H:
            if a.skip_hashes: return 'skipped (--skip-hashes)'
            raise AssertionError('HASHES.json is missing: the campaign files are incomplete')
        bad = [f for f, h in H['files'].items() if not os.path.exists(os.path.join(HERE, f)) or sha256(os.path.join(HERE, f)) != h]
        assert not bad, f'files differ from the registered versions: {bad}'
        return f"{len(H['files'])} files match the registered hashes"
    def queue():
        q = man['_queue'].get(W, [])
        if W not in man['workers']: return f'{W} is a helper: no own chains, it only takes over chains of stopped sessions'
        return f"{W}: {len(q)} chains, {sum(c['est_core_hours'] for c in q):.1f} core-hours (~{sum(c['est_core_hours'] for c in q) / 4:.1f} h on 4 cores)"
    def runner_tests():
        spec = importlib.util.spec_from_file_location('test_tfm', os.path.join(HERE, 'tests', 'test_tfm.py'))
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        names = sorted(n for n in dir(T) if n.startswith('test_'))
        for n in names: getattr(T, n)()
        return f'{len(names)} runner tests (gradients vs finite differences, last-position shortcut, checkpoint resume)'
    def mini_chain():
        d = tempfile.mkdtemp(prefix='campE_selftest_')
        def cfg(opt, seed):
            return dict(p=7, frac=0.8, split_seed=60 + seed, init_seed=50 + seed, d_model=16, n_heads=2, d_mlp=32, n_layers=1, opt=opt,
                        dtype='float32', steps=300, obs_every=20, heavy_every=100, eta_w=0.03, lam_w=0.015, eta_v=0.03, lam_v=0.2,
                        init=dict(emb=1.0, mat=1.0, head=1.0), adamw=dict(lr=1e-3, wd=1.0, b1=0.9, b2=0.98, eps=1e-8),
                        stop_generalized=dict(acc=0.99, factor=1.5, extra=10000))
        chains = [dict(id=f'SELFTEST_{o}', stage='dev', cell='mini', opt=o, n_layers=1, p=7, dial=dict(opt=o), seed=0, init_seed=50,
                       split_seed=60, worker=W, threads=1, cfg=cfg(o, 0)) for o in ('paper', 'hybrid', 'adamw')]
        fresh = dict(id='SELFTEST_fresh', stage='fresh', cell='mini', opt='mixed', n_layers=1, p=7, dial=None, seed=1, init_seed=51,
                     split_seed=61, worker=W, threads=1, cfg=dict(steps=300), after=[c['id'] for c in chains])
        mm = dict(man); mm['_by_id'] = dict(man['_by_id']); mm['_by_id'].update({c['id']: c for c in chains + [fresh]})
        runs = {}
        for c in chains:
            m = run_chain(c, mm, W, 'selftest', 'none', None, os.path.join(d, c['id']))
            for f in ('run.json.gz', 'chain.json'): assert os.path.exists(os.path.join(d, c['id'], 'out', f)), f
            runs[c['id']] = load_gz(os.path.join(d, c['id'], 'out', 'run.json.gz'))
            assert len(runs[c['id']]['loss']) == m['run']['steps'] + 1
        find = lambda cid, _m: runs.get(cid)
        sk = run_chain(fresh, mm, W, 'selftest', 'none', None, os.path.join(d, 'fresh_rule'), find=find)
        fr = run_chain(fresh, mm, W, 'selftest', 'none', None, os.path.join(d, 'fresh_forced'), find=find, force=True)
        assert fr['selection']['chosen'] in runs and fr['cfg']['init_seed'] == 51 and fr['cfg']['opt'] == mm['_by_id'][fr['selection']['chosen']]['cfg']['opt']
        shutil.rmtree(d, ignore_errors=True)
        return (f"3 development runs ({', '.join(c['opt'] for c in chains)}), the rule "
                f"({'skip: ' + sk['reason'][:40] + '...' if sk.get('skipped') else 'run'}), and a fresh-seed run of the chosen setting "
                f"({fr['selection']['chosen']}); toy problem, the numbers mean nothing")
    def git_checks():
        g = Git(HERE)
        assert g.branch != 'HEAD', 'detached HEAD: check out this session\'s branch'
        assert g.branch.startswith(BRANCH_PREFIXES), (f'this session is on branch {g.branch}, but results are collected only from branches named '
                                                      f'{"/".join(BRANCH_PREFIXES)}*: tell the user')
        url = g.run('remote', 'get-url', 'origin').stdout.strip()
        heads = g.remote_heads()
        # a real push of this worker's selftest note, so a branch the proxy refuses is caught now rather than after 11 hours
        os.makedirs(STATUS, exist_ok=True)
        fn = os.path.join(STATUS, f'{W}-selftest.json')
        write_json(fn, dict(worker=W, at=iso(), branch=g.branch, version=VERSION, host=host_info()))
        g.run('add', '--', fn)
        g.run(*g.identity_args(), 'commit', '-q', '-m', f'campaign E {W}: selftest', '--', os.path.relpath(fn, g.root))
        r = g.run('push', '-q', 'origin', f'HEAD:refs/heads/{g.branch}', check=False, timeout=180)
        if r.returncode != 0 and any(k in (r.stderr or '').lower() for k in ('non-fast-forward', 'fetch first')):
            g.run('fetch', '--no-tags', '--quiet', 'origin', f'+refs/heads/{g.branch}:refs/remotes/origin/{g.branch}', check=False, timeout=180)
            g.run('-c', 'rebase.autoStash=true', *g.identity_args(), 'rebase', '-X', 'theirs', f'origin/{g.branch}', check=False, timeout=180)
            r = g.run('push', '-q', 'origin', f'HEAD:refs/heads/{g.branch}', check=False, timeout=180)
        assert r.returncode == 0, 'pushing to this session\'s branch failed, so results could not reach GitHub: ' + (r.stderr or r.stdout).strip()[-400:]
        note = ' (note: this is the shared base branch; that works, but a session normally has its own branch)' if g.branch == BASE_BRANCH else ''
        return f"branch {g.branch}{note}; origin {url.split('@')[-1]}; {len(heads)} session branches visible; test push ok"
    check('python and numpy', py)
    check('one worker per clone', clone)
    check('registered file hashes', hashes)
    check('worker queue', queue)
    check('runner tests', runner_tests)
    check('one miniature run (train -> compact -> chain.json)', mini_chain)
    if not a.no_git: check('git (branch, fetch, real test push)', git_checks)
    print(f'SELFTEST PASS. Next, run: {next_command(W)}' if ok else 'SELFTEST FAIL: tell the user which check failed (copy the FAIL lines).')
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    def common(p, worker=True):
        p.add_argument('--manifest', default=DEFAULT_MANIFEST)
        if worker: p.add_argument('--worker', required=True); p.add_argument('--helper', action='store_true', help='allow an ID outside the manifest (takes over only)')
        return p
    p = common(sub.add_parser('run')); p.add_argument('--cores', type=int, default=0)
    p.add_argument('--no-push', action='store_true'); p.add_argument('--no-steal', action='store_true')
    p = common(sub.add_parser('watch')); p.add_argument('--minutes', type=float, default=9.0)
    p.add_argument('--cores', type=int, default=0); p.add_argument('--no-push', action='store_true'); p.add_argument('--no-steal', action='store_true')
    p = common(sub.add_parser('status'), worker=False); p.add_argument('--worker'); p.add_argument('--all', action='store_true')
    p = sub.add_parser('chain'); p.add_argument('--manifest', default=DEFAULT_MANIFEST); p.add_argument('--id', required=True); p.add_argument('--worker', required=True)
    p.add_argument('--instance', default='manual'); p.add_argument('--branch', default='?'); p.add_argument('--stolen-from', default=None)
    p = common(sub.add_parser('selftest')); p.add_argument('--no-git', action='store_true'); p.add_argument('--skip-hashes', action='store_true')
    a = ap.parse_args()
    dict(run=cmd_run, watch=cmd_watch, status=cmd_status, chain=cmd_chain, selftest=cmd_selftest)[a.cmd](a)

if __name__ == '__main__':
    main()
