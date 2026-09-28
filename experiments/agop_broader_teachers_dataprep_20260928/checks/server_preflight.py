#!/usr/bin/env python3
"""SERVER-only fixed-state checks for the declared broader-teacher pilot.

No training, optimizer update, model fitting, or full SwiGLU bank evaluation.
Only standard-library imports occur before the Linux/SERVER execution guard.
Exit zero and status PASS are necessary preconditions for the batch launch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback


RELU_TEACHERS = (
    "h3", "damped_2_3", "three_atom", "h3_plus_h5", "h3_plus_sine",
    "h2", "h4", "sine", "relu", "abs",
)
RANK16_TEACHERS = ("h3", "three_atom", "h3_plus_h5", "h2", "relu", "abs")
SWIGLU_LINKS = ("he3", "he2", "relu", "tanh")
IMMUTABLE_RELU_KERNELS = {
    "code/population.py": "dfb863648efaccac9f4be3d103c6d920dcb1f59cd9458bff4a000c7dbbbea488",
    "code/teachers_extra.py": "a0e9d12035a3fbe57b47a292a95a88fbe19355aee8b06ffa75e55a99a4b7cb11",
    "code/teachers_product.py": "b36f03bbb8f7f3a1b24b420131ec882f95e13da224cdda63c131041e747f25d1",
    "code/refit.py": "1835b87dcb33149ddfbf4f4f34c901836ec0090e5f3f4a92de23d40fba7dac22",
    "code/frames.py": "bb19ea9ac3705304b1308e79e13798a916c34349a3962a77126f560d696a4037",
}
REQUIRED_SOURCES = tuple(IMMUTABLE_RELU_KERNELS) + (
    "code/scaling_run.py", "swiglu/code/run_one.py", "swiglu/code/diagnostics.py",
    "swiglu/code/engine/swpop.py", "swiglu/code/engine/swsmall.py",
    "swiglu/code/engine/swsmall_periodic.py", "swiglu/code/engine/vlab.py",
    "checks/server_preflight.py",
)
FD_STEPS = (1e-4, 2.5e-5)
FD_ABSOLUTE_TOLERANCE = 2e-8
FD_RELATIVE_TOLERANCE = 2e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_relative(root, name):
    path = root / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"Unsafe relative path in manifest: {name}")
    require(path.is_file(), f"Missing declared file: {name}")
    return path


def verify_manifest(root):
    """Structural/hash validation; no model imports or numerical calls."""
    path = root / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    require(manifest["protocol_version"] == "broader_teachers_v2", "Unexpected protocol")
    sources = manifest["source_sha256"]
    require(bool(sources), "Manifest source hashes must be finalized before preflight")
    require(set(REQUIRED_SOURCES).issubset(sources), "Manifest lacks required scientific/preflight sources")
    for name, expected in sources.items():
        require(sha(_safe_relative(root, name)) == expected, f"Source hash mismatch: {name}")
    for name, expected in IMMUTABLE_RELU_KERNELS.items():
        require(sha(root / name) == expected, f"Original numerical kernel changed: {name}")
    jobs = manifest["configs"]
    require(len(jobs) == 28, "Expected 28 declared arms")
    require(len({j["id"] for j in jobs}) == 28, "Duplicate configuration IDs")
    observed = Counter()
    config_hashes = {}
    swiglu_shared = None
    for entry in jobs:
        config_path = _safe_relative(root, entry["config_path"])
        cfg = json.loads(config_path.read_text())
        require(cfg == entry["config"], f"Config differs from manifest: {entry['id']}")
        require(config_path.name == entry["id"] + ".json", "Configuration filename mismatch")
        config_hashes[entry["config_path"]] = sha(config_path)
        if entry["engine"] == "relu":
            require(cfg["id"] == entry["id"], "ReLU ID mismatch")
            expected = dict(student="relu", optimizer="gd", r=8, d=64, m=128,
                            scale=1e-4, h=.05, steps=20000)
            require(all(cfg[k] == v for k, v in expected.items()), "ReLU fixed recipe mismatch")
            require(cfg["teacher"] in RELU_TEACHERS and cfg["seed"] in (501, 502),
                    "Undeclared ReLU teacher/seed")
            observed[("relu", cfg["teacher"], cfg["seed"])] += 1
        else:
            require(entry["engine"] == "swiglu", "Unknown engine")
            require(isinstance(cfg, list) and len(cfg) == 1, "SwiGLU config must contain one job")
            job = cfg[0]
            a = job["args"]
            require(job["tag"] == entry["id"] and job["rank"] == 8, "SwiGLU identity/rank mismatch")
            require(a["d"] == 64 and a["m"] == 64 and a["s"] == .05
                    and a["c"] == [1.] * 8, "SwiGLU fixed recipe mismatch")
            require(a["link"] in SWIGLU_LINKS and a["seed"] in (501, 502),
                    "Undeclared SwiGLU teacher/seed")
            shared = {k: v for k, v in a.items() if k not in ("link", "seed")}
            if swiglu_shared is None:
                swiglu_shared = shared
            require(shared == swiglu_shared, "SwiGLU non-teacher arguments differ across cells")
            observed[("swiglu", a["link"], a["seed"])] += 1
    expected = Counter({(engine, link, seed): 1
                        for engine, links in (("relu", RELU_TEACHERS), ("swiglu", SWIGLU_LINKS))
                        for link in links for seed in (501, 502)})
    require(observed == expected, "Declared teacher/seed factorial mismatch")
    require({p.name for p in (root / "configs").glob("*.json")} ==
            {Path(j["config_path"]).name for j in jobs}, "Unexpected/missing arm config file")
    return manifest, dict(manifest_sha256=sha(path), source_hashes=dict(sources),
                          config_sha256=config_hashes, arms=28, relu_arms=20, swiglu_arms=8)


def raw_link(name, z):
    """Independent scalar formulas, rather than calling teacher.raw_values."""
    z = np.asarray(z)
    h3 = (z**3 - 3*z) / math.sqrt(6.)
    if name in ("h3", "damped_2_3", "three_atom"):
        atoms = {"h3": ((1., 1.),), "damped_2_3": ((2/3, 1.),),
                 "three_atom": ((.4, .2), (.7, .3), (1., .5))}[name]
        return sum(p*v**(-3.5)*(z**3-3*v*z)*np.exp(-.5*(1/v-1)*z*z) for v, p in atoms)
    if name == "h3_plus_h5":
        return h3 + .1*(z**5-10*z**3+15*z)/math.sqrt(120.)
    if name == "h3_plus_sine":
        return h3 + .1*np.sin(z)
    if name == "h2":
        return (z*z-1)/math.sqrt(2.)
    if name == "h4":
        return (z**4-6*z*z+3)/math.sqrt(24.)
    if name == "sine":
        return np.sin(z)
    if name == "relu":
        return np.maximum(z, 0.)
    if name == "abs":
        return np.abs(z)
    raise ValueError(name)


def raw_link_prime(name, z):
    z = np.asarray(z)
    if name in ("h3", "damped_2_3", "three_atom"):
        atoms = {"h3": ((1., 1.),), "damped_2_3": ((2/3, 1.),),
                 "three_atom": ((.4, .2), (.7, .3), (1., .5))}[name]
        return sum(p*v**(-3.5)*np.exp(-.5*(1/v-1)*z*z)*
                   (3*z*z-3*v-(1/v-1)*z*(z**3-3*v*z)) for v, p in atoms)
    if name == "h3_plus_h5":
        return (3*z*z-3)/math.sqrt(6.) + .1*(5*z**4-30*z*z+15)/math.sqrt(120.)
    if name == "h3_plus_sine":
        return (3*z*z-3)/math.sqrt(6.) + .1*np.cos(z)
    if name == "h2":
        return math.sqrt(2.)*z
    if name == "h4":
        return (4*z**3-12*z)/math.sqrt(24.)
    if name == "sine":
        return np.cos(z)
    if name == "relu":
        return (z > 0).astype(float)
    if name == "abs":
        return np.sign(z)
    raise ValueError(name)


def gaussian_integral(function):
    # Split at zero for ReLU/abs; all declared formulas have negligible Gaussian
    # weighted polynomial tails beyond 12. This is a numerical consistency test.
    def integrand(x):
        return float(function(x))*math.exp(-x*x/2)/math.sqrt(2*math.pi)
    return sum(quad(integrand, a, b, epsabs=3e-13, epsrel=3e-13, limit=150)[0]
               for a, b in ((-12., 0.), (0., 12.)))


def scalar_reference(name):
    q = lambda z: raw_link(name, z)
    qp = lambda z: raw_link_prime(name, z)
    mu = gaussian_integral(q)
    second = gaussian_integral(lambda z: q(z)**2)
    linear = gaussian_integral(lambda z: z*q(z))
    derivative_mean = gaussian_integral(qp)
    derivative_second = gaussian_integral(lambda z: qp(z)**2)
    require(abs(linear-derivative_mean) < 2e-10, f"Scalar Stein check failed: {name}")
    return dict(mean=mu, second=second, linear=linear,
                derivative_mean=derivative_mean, derivative_second=derivative_second)


def compare_values(actual, expected, atol, rtol, label):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    require(actual.shape == expected.shape, f"{label}: shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{label}: nonfinite")
    error = float(np.max(np.abs(actual-expected), initial=0.))
    reference_scale = float(np.max(np.abs(expected), initial=0.))
    limit = atol + rtol*reference_scale
    require(error <= limit, f"{label}: error {error:.6g} > tolerance {limit:.6g}")
    return dict(max_absolute_error=error, reference_scale=reference_scale, tolerance=limit)


def finite_difference_report(actual, evaluator, label):
    actual = np.asarray(actual, float)
    require(np.isfinite(actual).all(), f"{label}: nonfinite analytic derivative")
    rows = []
    for step in FD_STEPS:
        fd = (np.asarray(evaluator(step))-np.asarray(evaluator(-step)))/(2*step)
        require(fd.shape == actual.shape and np.isfinite(fd).all(), f"{label}: invalid finite difference")
        error = float(np.max(np.abs(fd-actual), initial=0.))
        scale = float(max(np.max(np.abs(actual), initial=0.), np.max(np.abs(fd), initial=0.)))
        limit = FD_ABSOLUTE_TOLERANCE + FD_RELATIVE_TOLERANCE*scale
        rows.append(dict(step=step, max_absolute_error=error, derivative_scale=scale,
                         tolerance=limit, passed=error <= limit))
    require(all(row["passed"] for row in rows), f"{label}: directional derivative failed: {rows}")
    return dict(label=label, steps=rows, passed=True)


def fixed_state(d=64):
    # Small bank, deliberately moderate finite amplitudes so differences of the
    # full MSE do not disappear into the constant teacher energy. These are not
    # production initializations, and no step is ever applied to these arrays.
    rng = np.random.default_rng(712034)
    W = .045*rng.normal(size=(4, d))
    for i in range(4):
        W[i, i] += .65
        W[i, i+4] += .21
    A = np.array([.4, -.7, .3, .55])
    b = np.array([-.3, .2, .6, -.45])
    DA = np.array([.3, -.4, .5, .2])
    DW = rng.normal(size=W.shape)
    DW /= np.linalg.norm(DW)
    Db = np.array([-.25, .4, .3, -.2])
    return A, W, b, DA, DW, Db


def check_relu_teacher(name, rank, reference, population):
    teacher = population.Teacher(name, rank, 64)
    require(teacher.r == rank and teacher.d == 64, "Factory rank/dimension mismatch")
    require(teacher.U.shape == (64, rank) and np.array_equal(teacher.U, np.eye(64)[:, :rank]),
            "Unexpected teacher/reference axes")
    mu, second = reference["mean"], reference["second"]
    norm = math.sqrt(second+(rank-1)*mu*mu)
    target_mean = math.sqrt(rank)*mu/norm
    normalization = dict(
        norm=compare_values(teacher.norm, norm, 2e-11, 2e-10, name+" norm"),
        mean=compare_values(teacher.mean, target_mean, 2e-11, 2e-10, name+" mean"),
        second_moment=compare_values(teacher.second_moment, 1., 1e-13, 0., name+" second moment"))
    X = np.reshape(np.sin(np.arange(3*64)*.37), (3, 64))
    expected_values = np.sum(raw_link(name, X[:, :rank]), axis=1)/(math.sqrt(rank)*norm)
    normalization["point_values"] = compare_values(teacher.values(X), expected_values, 2e-12, 2e-11,
                                                   name+" teacher values")
    derivative_variance = reference["derivative_second"]-reference["derivative_mean"]**2
    require(derivative_variance > 1e-10, f"True-rank derivative variance not positive: {name}")
    weak = derivative_variance/(rank*norm*norm)
    leading = (derivative_variance+rank*reference["derivative_mean"]**2)/(rank*norm*norm)
    A, W, b, DA, DW, Db = fixed_state()
    C, gW, gb = teacher.cross(W, b, 0., True)
    require(C.shape == (4,) and gW.shape == (4, 64) and gb.shape == (4,), "Cross shapes mismatch")
    checks = []
    for label, dw, db in (("W", DW, np.zeros_like(Db)), ("b", np.zeros_like(DW), Db),
                          ("W_and_b", DW, Db)):
        actual = np.sum(gW*dw, axis=1)+gb*db
        checks.append(finite_difference_report(actual,
            lambda t, dw=dw, db=db: teacher.cross(W+t*dw, b+t*db, 0., False)[0],
            f"{name}/r{rank}/cross/{label}"))
    # alpha=1 is exactly a linear student, and is also the production driver's
    # source of the target first-Hermite coefficient diagnostic.
    target_linear = np.zeros(64)
    target_linear[:rank] = reference["linear"]/(math.sqrt(rank)*norm)
    Cl, gWl, gbl = teacher.cross(W, b, 1., True)
    linear_checks = dict(
        cross=compare_values(Cl, W@target_linear+b*target_mean, 3e-11, 2e-9, name+" linear cross"),
        weight_gradient=compare_values(gWl, np.broadcast_to(target_linear, W.shape), 3e-11, 2e-9,
                                       name+" linear weight derivative"),
        bias_gradient=compare_values(gbl, np.full(4, target_mean), 3e-11, 2e-9,
                                     name+" linear bias derivative"))
    L, forces, *_ = population.loss_force(A, W, b, teacher, 0.)
    require(np.isfinite(L) and all(np.isfinite(x).all() for x in forces), "Nonfinite loss/force")
    require(L >= -1e-11, "Negative full population MSE")
    for label, da, dw, db in (
        ("A", DA, np.zeros_like(DW), np.zeros_like(Db)),
        ("W", np.zeros_like(DA), DW, np.zeros_like(Db)),
        ("b", np.zeros_like(DA), np.zeros_like(DW), Db),
        ("joint", DA, DW, Db)):
        actual = np.asarray(-2/len(A)*sum(float(np.sum(f*v)) for f, v in zip(forces, (da, dw, db))))
        checks.append(finite_difference_report(actual,
            lambda t, da=da, dw=dw, db=db: population.loss_force(A+t*da, W+t*dw, b+t*db, teacher, 0.)[0],
            f"{name}/r{rank}/full_MSE/{label}"))
    return dict(teacher=name, rank=rank, dimension=64, bank_width=4, status="PASS",
                normalization=normalization, target_mean=target_mean, target_second_moment=1.,
                target_variance=1-target_mean**2, target_linear_energy=float(target_linear@target_linear),
                true_rank=rank, true_rank_basis="Positive Var(q') for independent nonlinear additive coordinates",
                target_agop_weak_eigenvalue=weak, target_agop_leading_eigenvalue=leading,
                balanced_weak_energy_reference=derivative_variance/reference["derivative_second"],
                numerical_reference_scope="Independent scalar quadrature on [-12,12], not a theorem certificate",
                linear_identity_checks=linear_checks, directional_checks=checks, fixed_state_loss=float(L))


def check_swiglu_factories(root):
    sys.path.insert(0, str(root / "swiglu/code/engine"))
    import swsmall
    import swsmall_periodic
    import swpop
    z = np.array([-2.7, -.8, 0., .3, 1.9])
    reference_links = {
        "he3": lambda x: (np.asarray(x)**3-3*np.asarray(x))/math.sqrt(6.),
        "he2": lambda x: (np.asarray(x)**2-1)/math.sqrt(2.),
        "relu": lambda x: np.maximum(x, 0.),
        "tanh": lambda x: np.tanh(x),
    }
    records = []
    for name, f in reference_links.items():
        train_f, train_breaks = swsmall_periodic.make_link(name)
        diagnostic_f, diagnostic_breaks = swsmall.make_link(name)
        require(train_breaks == diagnostic_breaks, "Training/diagnostic teacher kink mismatch")
        point_check = compare_values(train_f(z), f(z), 2e-14, 2e-14, name+" training link")
        compare_values(diagnostic_f(z), f(z), 2e-14, 2e-14, name+" diagnostic link")
        T = swpop.Teacher(train_f, np.ones(8), breaks=train_breaks, n_x=12)
        T2 = swpop.Teacher(diagnostic_f, np.ones(8), breaks=diagnostic_breaks, n_x=24)
        mu = gaussian_integral(f)
        second = gaussian_integral(lambda x: f(x)**2)
        gamma = 1/math.sqrt(8*second+8*7*mu*mu)
        expected_mean = gamma*8*mu
        checks = dict(gamma=compare_values(T.gamma, gamma, 2e-10, 2e-9, name+" teacher gamma"),
                      mean=compare_values(T.EY, expected_mean, 2e-10, 2e-9, name+" teacher mean"),
                      variance=compare_values(T.V, 1-expected_mean**2, 2e-10, 2e-9, name+" teacher variance"),
                      refinement=compare_values(np.array([T.gamma, T.EY, T.V]),
                                                np.array([T2.gamma, T2.EY, T2.V]),
                                                2e-10, 2e-9, name+" teacher order refinement"))
        require(T.r == 8 and T.V > 0, "SwiGLU teacher rank/variance invalid")
        records.append(dict(link=name, rank=8, status="PASS", gamma=float(T.gamma),
                            mean=float(T.EY), variance=float(T.V), point_check=point_check,
                            normalization_checks=checks,
                            scope="Teacher factory only; no SwiGLU bank, gradient, refit, or trajectory evaluation"))
    return records


def scientific_checks(root, manifest):
    require(platform.system() == "Linux" and os.environ.get("AGOP_EXECUTION_SITE") == "SERVER",
            "Scientific checks require Linux and AGOP_EXECUTION_SITE=SERVER")
    global np, quad
    import numpy as np
    from scipy.integrate import quad
    sys.path.insert(0, str(root / "code"))
    import population
    import scaling_run
    swi_driver = load_module("broader_swiglu_driver_preflight", root / "swiglu/code/run_one.py")
    for entry in manifest["configs"]:
        if entry["engine"] == "relu":
            require(scaling_run.config_check(entry["config"]) == entry["config"], "Config validator changed fields")
        else:
            swi_driver.validate_jobs(entry["config"])
    references = {name: scalar_reference(name) for name in RELU_TEACHERS}
    records = []
    for rank, names in ((8, RELU_TEACHERS), (16, RANK16_TEACHERS)):
        for name in names:
            record = check_relu_teacher(name, rank, references[name], population)
            records.append(record)
            print(json.dumps(dict(event="fixed_teacher_check", teacher=name, rank=rank, status="PASS")), flush=True)
    try:
        population.Teacher("quadratic_product", 8, 64)
    except ValueError:
        product_rank_guard = "PASS: existing bivariate teacher refuses a false rank-eight label"
    else:
        raise AssertionError("Bivariate teacher unexpectedly accepts rank eight")
    return dict(relu_fixed_state_checks=records, swiglu_teacher_factory_checks=check_swiglu_factories(root),
                product_rank_guard=product_rank_guard, independent_scalar_references=references,
                input_gradient_or_population_recovery_certified=False,
                numpy_version=np.__version__, python_version=sys.version.split()[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=90)
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or os.environ.get("AGOP_EXECUTION_SITE") != "SERVER":
        print(json.dumps(dict(status="REFUSED", reason="Linux and AGOP_EXECUTION_SITE=SERVER required; no scientific imports/evaluation occurred")), flush=True)
        return 2
    require(1 <= args.seconds <= 180, "Preflight time limit must be 1..180 seconds")
    sys.dont_write_bytecode = True
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.root.resolve()
    require(root.is_dir(), "Missing pilot root")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write("{\"status\":\"STARTED\"}\n")
    started = time.monotonic()
    report = dict(status="STARTED", started_utc=datetime.now(timezone.utc).isoformat(),
                  execution_site="SERVER", platform=platform.system(), training_executed=False,
                  full_swiglu_bank_evaluated=False, maximum_synthetic_bank_width=4,
                  seconds_limit=args.seconds, preflight_sha256=sha(__file__),
                  scope="Bounded fixed-state implementation/normalization checks; not trajectory or recovery evidence")
    before = None
    def timeout(*_):
        raise TimeoutError("Fixed-state preflight time budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        manifest, before = verify_manifest(root)
        report["provenance"] = before
        report.update(scientific_checks(root, manifest))
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if before is not None:
            try:
                _, after = verify_manifest(root)
                report["inputs_unchanged"] = before == after
                if not report["inputs_unchanged"]:
                    report.update(status="FAIL", error="Inputs changed during preflight")
            except Exception as exc:
                report.update(status="FAIL", inputs_unchanged=False, final_integrity_error=repr(exc))
        report["elapsed_seconds"] = time.monotonic()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.out)
    print(json.dumps(dict(status=report["status"], elapsed_seconds=report["elapsed_seconds"], output=str(args.out))), flush=True)
    return 0 if report["status"] == "PASS" and report.get("inputs_unchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
