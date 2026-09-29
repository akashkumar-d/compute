"""Standard-library tests; scientific runners are never imported or executed."""
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import launch


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.waits = []

    def monotonic(self):
        return self.now

    def time(self):
        return 1700000000.0 + self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds

    def clear(self):
        pass

    def set(self):
        pass


class LauncherTests(unittest.TestCase):
    def test_28_rank_workers_leave_four_cpus_available(self):
        runtime=dict(workers=28,reserved_cpus=4,min_available_memory_gib=24,min_free_disk_gib=10)
        snapshot=dict(effective_cpus=32,available_memory_gib=30,free_disk_gib=100)
        with mock.patch.object(launch,'resource_snapshot',return_value=snapshot.copy()):
            self.assertEqual(launch.require_capacity(runtime)['workers'],28)
        for allocated in (4,28,30,31.9):
            with mock.patch.object(launch,'resource_snapshot',return_value=dict(snapshot,effective_cpus=allocated)):
                with self.assertRaisesRegex(RuntimeError,'Insufficient allocated CPUs'):
                    launch.require_capacity(runtime)
        with mock.patch.object(launch,'resource_snapshot',return_value=dict(snapshot,available_memory_gib=23.9)):
            with self.assertRaisesRegex(RuntimeError,'memory'):
                launch.require_capacity(runtime)
        with mock.patch.object(launch,'resource_snapshot',return_value=dict(snapshot,free_disk_gib=9.9)):
            with self.assertRaisesRegex(RuntimeError,'disk'):
                launch.require_capacity(runtime)

    def setUp(self):
        self.gate = mock.patch.object(launch, 'validate_protocol')
        self.gate.start()
        self.addCleanup(self.gate.stop)
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "runtime": {"workers": 2, "global_seconds": 60, "per_arm_seconds": 30,
                        "diagnostic_reserve_seconds": 5},
            "source_sha256": {}, "configs": [],
        }
        for engine, (script, _) in launch.ENGINES.items():
            p = self.root / script
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("raise AssertionError('Scientific runner must never execute in these tests')\n")
            self.manifest["source_sha256"][script] = launch.sha256(p)
        (self.root / "configs").mkdir()
        self.add_config("arm0")
        self.write_manifest()

    def add_config(self, cid, engine="relu"):
        cfg = {"id": cid, "seed": 7}
        if engine == "swiglu":
            cfg = [cfg]
        rel = f"configs/{cid}.json"
        (self.root / rel).write_text(json.dumps(cfg))
        self.manifest["configs"].append({"id": cid, "engine": engine,
                                         "config_path": rel, "config": cfg})

    def write_manifest(self):
        (self.root / "MANIFEST.json").write_text(json.dumps(self.manifest))

    def test_dry_run_is_static_and_creates_nothing(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        with mock.patch.object(launch.subprocess, "Popen", side_effect=AssertionError("No execution")):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(launch.main(["--dry-run"], self.root), 0)
        self.assertEqual(json.loads(out.getvalue())["model_evaluations"], 0)
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_source_and_config_changes_rejected(self):
        cfg = self.root / "configs/arm0.json"
        cfg.write_text('{"id":"arm0","seed":8}')
        with self.assertRaisesRegex(ValueError, "Config JSON differs"):
            launch.validate(self.root)
        cfg.write_text(json.dumps(self.manifest["configs"][0]["config"]))
        (self.root / "code/scaling_run.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            launch.validate(self.root)

    def test_manifest_change_rejected(self):
        _, digest = launch.validate(self.root)
        self.manifest["runtime"]["workers"] = 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "MANIFEST.json changed"):
            launch.validate(self.root, digest)

    def test_relative_paths_and_duplicate_ids_rejected(self):
        self.manifest["configs"][0]["config_path"] = "../outside.json"
        self.write_manifest()
        with self.assertRaises(ValueError):
            launch.validate(self.root)
        self.manifest["configs"][0]["config_path"] = "configs/arm0.json"
        self.manifest["configs"].append(self.manifest["configs"][0].copy())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate config id"):
            launch.validate(self.root)

    def test_source_symlink_escape_rejected(self):
        target = Path(__file__).resolve()
        p = self.root / "escape.py"
        p.symlink_to(target)
        self.manifest["source_sha256"]["escape.py"] = launch.sha256(p)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes bundle"):
            launch.validate(self.root)

    def test_unsupported_engine_and_invalid_budget_rejected(self):
        self.manifest["configs"][0]["engine"] = "unknown"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsupported engine"):
            launch.validate(self.root)
        self.manifest["configs"][0]["engine"] = "relu"
        self.manifest["runtime"]["global_seconds"] = 3601
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "global <= 3600"):
            launch.validate(self.root)
        self.manifest["runtime"]["global_seconds"] = 3600
        self.manifest["runtime"]["workers"] = 29
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "from 1 to 28"):
            launch.validate(self.root)
        self.manifest["runtime"]["workers"] = 2
        self.manifest["runtime"]["per_arm_seconds"] = 1991
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "per_arm <= 1990"):
            launch.validate(self.root)

    def test_server_guard_and_existing_directory(self):
        with mock.patch.object(launch.sys, "platform", "darwin"), mock.patch.dict(launch.os.environ, {"AGOP_EXECUTION_SITE":"SERVER"}):
            with self.assertRaisesRegex(RuntimeError, "Execution requires"):
                launch.main([], self.root)
        self.assertFalse((self.root / "execution").exists())
        (self.root / "execution").mkdir()
        with self.assertRaises(FileExistsError):
            launch.main(["--dry-run"], self.root)

    def test_confirmed_engine_commands(self):
        self.add_config("swiglu", "swiglu")
        self.write_manifest()
        manifest, _ = launch.validate(self.root)
        runtime = {"per_arm_seconds": 480, "diagnostic_reserve_seconds": 140}
        relu, swiglu = [launch.build_command(self.root, self.root / "execution", c,
                                           runtime, 480, "cutoff") for c in manifest["configs"]]
        self.assertEqual(relu[relu.index("--max-seconds") + 1], "470.0")
        self.assertEqual(swiglu[swiglu.index("--wall-seconds") + 1], "330.0")
        self.assertEqual(swiglu[swiglu.index("--diagnostic-seconds") + 1], "140.0")
        self.assertEqual(swiglu[swiglu.index("--tag") + 1], "swiglu")
        shortened = launch.build_command(self.root, self.root / "execution", manifest["configs"][1],
                                         runtime, 200, "cutoff")
        self.assertEqual(shortened[shortened.index("--wall-seconds") + 1], "50.0")

    def test_named_dependency_requires_exit_and_nonlive_pid(self):
        dep = {"run": "one-job", "max_wait_seconds": 70}
        d = self.root / "compute/runs/one-job/.lrun"
        d.mkdir(parents=True)
        (d / "pid").write_text("1234\n")
        with mock.patch.object(launch, "pid_alive", return_value=False) as alive:
            self.assertFalse(launch.dependency_state(dep, self.root)["ready"])
            (d / "exit_code").write_text("7\n")
            state = launch.dependency_state(dep, self.root)
            self.assertTrue(state["ready"])
            self.assertEqual(state["exit_code"], 7)
            self.assertEqual(alive.call_args.args, (1234,))
        with mock.patch.object(launch, "pid_alive", return_value=True):
            self.assertFalse(launch.dependency_state(dep, self.root)["ready"])

    def mocked_execute(self, *, hung=False, dependency=None, unkillable=False):
        self.write_manifest()
        manifest, digest = launch.validate(self.root)
        clock, processes, signals = Clock(), {}, []
        root, output = self.root, self.root / "execution"

        class Process:
            def __init__(self, cmd, **kwargs):
                self.pid = 10000 + len(processes)
                self.started = clock.now
                self.returncode = None
                self.command = cmd
                self.kwargs = kwargs
                processes[self.pid] = self

            def poll(self):
                if not hung and self.returncode is None and clock.now >= self.started + 5:
                    key = "--out" if "--out" in self.command else "--out-dir"
                    dest = Path(self.command[self.command.index(key) + 1])
                    dest.mkdir(parents=True)
                    (dest / ("DONE.json" if key == "--out" else "result.json")).write_text("{}")
                    self.returncode = 0
                return self.returncode

            def wait(self, timeout):
                code = self.poll()
                if code is None:
                    raise launch.subprocess.TimeoutExpired(self.command, timeout)
                return code

        def killpg(pid, sig):
            self.assertIn(pid, processes, "May only signal process groups created by this invocation")
            signals.append((pid, sig, clock.now))
            if sig == signal.SIGKILL and processes[pid].returncode is None and not unkillable:
                processes[pid].returncode = -9

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(launch, "require_server", return_value="SERVER"))
            stack.enter_context(mock.patch.object(launch, "require_review"))
            stack.enter_context(mock.patch.object(launch.os, "nice", return_value=10))
            stack.enter_context(mock.patch.object(launch, "require_capacity", return_value={"mock": True}))
            stack.enter_context(mock.patch.object(launch.time, "monotonic", side_effect=clock.monotonic))
            stack.enter_context(mock.patch.object(launch.time, "time", side_effect=clock.time))
            stack.enter_context(mock.patch.object(launch.threading, "Event", return_value=clock))
            stack.enter_context(mock.patch.object(launch.subprocess, "Popen", side_effect=Process))
            stack.enter_context(mock.patch.object(launch.os, "killpg", side_effect=killpg))
            if dependency:
                stack.enter_context(mock.patch.object(launch, "dependency_state", side_effect=lambda _: dependency(clock)))
            code = launch.execute(root, output, manifest, digest)
        return code, clock, processes, signals, json.loads((output / "STATUS.json").read_text())

    def test_mocked_batch_and_logs(self):
        self.add_config("arm1", "swiglu")
        self.add_config("arm2")
        code, clock, processes, signals, status = self.mocked_execute()
        self.assertEqual(code, 0)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(len(status["completed"]), 3)
        self.assertEqual(len(list((self.root / "execution/logs").glob("*.stderr.log"))), 3)
        for process in processes.values():
            self.assertTrue(process.kwargs["start_new_session"])
            for name in launch.THREAD_VARIABLES:
                self.assertEqual(process.kwargs["env"][name], "1")
        self.assertTrue((self.root / "execution/STATE_HISTORY.jsonl").stat().st_size)

    def test_rank_queue_32_arms_never_exceeds_28_workers(self):
        self.manifest['runtime']['workers'] = 28
        for i in range(1,32): self.add_config(f'arm{i}', 'swiglu' if i%2 else 'relu')
        code, clock, processes, signals, status = self.mocked_execute()
        self.assertEqual(code,0)
        self.assertEqual(len(status['completed']),32)
        started = [p.started for p in processes.values()]
        self.assertEqual(started.count(1000.),28)
        self.assertEqual(started.count(1005.),4)
        self.assertEqual(clock.now,1010.)
        self.assertFalse(status['pending'])
        self.assertFalse(status['active'])

    def test_mocked_hard_caps_and_pending_preservation(self):
        self.manifest["runtime"]["global_seconds"] = 30
        for i in range(1, 5):
            self.add_config(f"arm{i}")
        code, clock, processes, signals, status = self.mocked_execute(hung=True)
        self.assertEqual(code, 124)
        self.assertEqual(status["pending"], ["arm2", "arm3", "arm4"])
        self.assertLessEqual(clock.now, 1030)
        for process in processes.values():
            kills = [t for pid, sig, t in signals if pid == process.pid and sig == signal.SIGKILL]
            self.assertTrue(kills)
            self.assertLessEqual(min(kills) - process.started, 30)
        self.assertEqual(status["status"], "capped")

    def test_queue_timeout_is_separate_and_uses_30_second_waits(self):
        self.manifest["dependency"] = {"run": "one-job", "max_wait_seconds": 70}
        code, clock, processes, signals, status = self.mocked_execute(dependency=lambda _: {"ready": False})
        self.assertEqual(code, 124)
        self.assertEqual(clock.waits, [30, 30, 10])
        self.assertFalse(processes)
        self.assertFalse(signals)
        self.assertIsNone(status["compute_elapsed_seconds"])
        self.assertEqual(status["status"], "queue_timeout")

    def test_kill_without_reap_stays_active_in_final_receipt(self):
        self.manifest["runtime"]["global_seconds"] = 30
        code, clock, processes, signals, status = self.mocked_execute(hung=True, unkillable=True)
        self.assertEqual(code, 124)
        self.assertLessEqual(clock.now, 1030)
        self.assertFalse(status["completed"])
        self.assertEqual(len(status["active"]), 1)
        child = status["active"][0]
        self.assertFalse(child["reap_acknowledged"])
        self.assertEqual(child["cleanup_status"], "kill_requested_unreaped")
        self.assertIsNone(child["returncode"])

    def test_per_arm_cap_reaps_before_larger_global_cap(self):
        code, clock, processes, signals, status = self.mocked_execute(hung=True)
        self.assertEqual(code, 124)
        self.assertLess(clock.now, 1060)
        self.assertFalse(status["active"])
        self.assertTrue(status["completed"][0]["reap_acknowledged"])
        self.assertEqual(status["completed"][0]["stop_reason"], "per_arm_wall_cap")

    def test_queue_does_not_spend_compute_budget(self):
        self.manifest["dependency"] = {"run": "one-job", "max_wait_seconds": 70}
        code, clock, processes, signals, status = self.mocked_execute(
            dependency=lambda c: {"ready": c.now >= 1060, "exit_code": 7})
        self.assertEqual(code, 0)
        self.assertEqual(status["queue_elapsed_seconds"], 60)
        self.assertEqual(status["compute_elapsed_seconds"], 5)
        self.assertEqual(min(p.started for p in processes.values()), 1060)


class V10GatesTests(unittest.TestCase):
    def test_server_only_modes(self):
        for platform, site, valid in [('darwin','LOCAL_AUTHORIZED',False),
              ('linux','SERVER',True),('darwin','SERVER',False),('linux','',False),('linux','LOCAL_AUTHORIZED',False)]:
            with mock.patch.object(launch.sys,'platform',platform), mock.patch.dict(
                    launch.os.environ, {'AGOP_EXECUTION_SITE':site}):
                if valid:
                    self.assertEqual(launch.require_server(), site)
                else:
                    with self.assertRaises(RuntimeError): launch.require_server()

    def test_server_allocation_gate(self):
        runtime = dict(workers=2,reserved_cpus=2,min_available_memory_gib=8,min_free_disk_gib=5)
        with mock.patch.dict(launch.os.environ, {'AGOP_EXECUTION_SITE':'SERVER'}):
            for count, valid in [(4,False),(30,False),(32,True)]:
                with mock.patch.object(launch,'resource_snapshot',return_value=dict(
                        effective_cpus=count,available_memory_gib=20,free_disk_gib=30)):
                    if valid: launch.require_capacity(runtime)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'at least32'): launch.require_capacity(runtime)

    def test_exact_requested_command_budgets(self):
        entry = dict(engine='swiglu',config_path='configs/a.json',id='a')
        cmd=launch.build_command(Path('/bundle'),Path('/output'),entry,
              dict(per_arm_seconds=1990,diagnostic_reserve_seconds=180),1990,'cutoff')
        self.assertEqual(cmd[cmd.index('--wall-seconds')+1],'1800.0')
        self.assertEqual(cmd[cmd.index('--diagnostic-seconds')+1],'180.0')
        with mock.patch.dict(launch.os.environ, {'AGOP_EXECUTION_SITE':'SERVER'}), mock.patch.object(launch.sys,'platform','linux'):
            env=launch.child_environment()
        self.assertEqual(env['AGOP_EXECUTION_SITE'],'SERVER')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')

    def test_priority_failure_is_fatal_before_any_child(self):
        with mock.patch.object(launch.os,'nice',side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):launch.require_priority()
        with mock.patch.object(launch.os,'nice',return_value=0):
            with self.assertRaisesRegex(RuntimeError,'Reduced priority'):launch.require_priority()

    def test_frozen_rank_design_criteria_order_and_runtime(self):
        root=Path(__file__).resolve().parent.parent
        manifest=launch.load_json(root/'MANIFEST.json')
        launch.validate_protocol(manifest,root)
        self.assertEqual(len(manifest['configs']),32)
        for kind in ('runtime','criteria','config','queue','labels'):
            changed=json.loads(json.dumps(manifest))
            if kind=='runtime': changed['runtime']['global_seconds']=4200
            elif kind=='criteria': changed['criterion']['same_checkpoint_minimum_alignment_gain']=.4
            elif kind=='queue': changed['configs'].reverse()
            elif kind=='labels': changed['configs'][0]['rank']=2
            else:
                cfg=changed['configs'][0]['config']
                if isinstance(cfg,list):cfg[0]['args']['seed']=999
                else:cfg['seed']=999
            with self.assertRaises(ValueError,msg=kind):launch.validate_protocol(changed,root)

    def test_review_refuses_unreviewed_or_wrong_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);p=root/'REVIEW.json'
            for value in [dict(status='pending_independent_review'),dict(
                    status='approved_for_execution',manifest_sha256='wrong',reviewer='peer',checks=['a'])]:
                p.write_text(json.dumps(value))
                with self.assertRaises(RuntimeError):launch.require_review(root,'correct')
            p.write_text(json.dumps(dict(status='approved_for_execution',
                manifest_sha256='correct',reviewer='peer',checks=['a'])))
            launch.require_review(root,'correct')

    def test_restart_refuses_uncertain_prior_and_retains_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);first=root/'attempt1';first.mkdir();sentinel=first/'preserved'
            sentinel.write_text('unchanged')
            with launch.attempt_lock(root,first):
                with self.assertRaises(BlockingIOError):
                    with launch.attempt_lock(root,root/'attempt2'): pass
            with self.assertRaises(OSError):
                with launch.attempt_lock(root,root/'attempt2'): pass
            (first/'STATUS.json').write_text(json.dumps(dict(status='running',active=[],completed=[])))
            with self.assertRaises(RuntimeError):
                with launch.attempt_lock(root,root/'attempt2'): pass
            (first/'STATUS.json').write_text(json.dumps(dict(status='capped',active=[],completed=[dict(pid=777)])))
            with mock.patch.object(launch.os,'killpg'):
                with self.assertRaisesRegex(RuntimeError,'remains live'):
                    with launch.attempt_lock(root,root/'attempt2'): pass
            with mock.patch.object(launch.os,'killpg',side_effect=ProcessLookupError):
                with launch.attempt_lock(root,root/'attempt2'): pass
            self.assertEqual(sentinel.read_text(),'unchanged')


if __name__ == "__main__":
    unittest.main()
