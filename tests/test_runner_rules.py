import numpy as np
import pandas as pd

from aerosurrogate.config import ConvergenceConfig, LauncherConfig
from aerosurrogate.runner.launchers import DockerLauncher, NativeLauncher, make_launcher
from aerosurrogate.runner.monitor import diverged, forces_converged, window_stats


def hist(cl, cd=None, cm=None):
    n = len(cl)
    return pd.DataFrame({"Time": np.arange(1, n + 1), "Cd": cd if cd is not None else np.full(n, 0.02),
                         "Cl": cl, "CmPitch": cm if cm is not None else np.full(n, 0.1)})


def test_converged_flat_history():
    cfg = ConvergenceConfig(min_iter=100, window=50)
    cl = np.concatenate([np.linspace(-2, -1, 100), np.full(100, -1.0)])
    ok, s = forces_converged(hist(cl), cfg)
    assert ok and np.isclose(s.cl, -1.0)


def test_not_converged_oscillating():
    cfg = ConvergenceConfig(min_iter=100, window=50)
    cl = -1 + 0.01 * np.sin(np.arange(300) / 3)
    ok, s = forces_converged(hist(cl), cfg)
    assert not ok and s.cl_range_rel > cfg.cl_rel_tol


def test_min_iter_respected():
    cfg = ConvergenceConfig(min_iter=500, window=50)
    ok, _ = forces_converged(hist(np.full(200, -1.0)), cfg)
    assert not ok


def test_window_stats_empty():
    assert window_stats(pd.DataFrame(), 10) is None


def test_divergence_detection():
    assert diverged(hist(np.array([-1, -1, np.nan]))) == "non-finite force coefficients"
    assert "100" in diverged(hist(np.array([-1, -1, -1e4])))
    assert diverged(hist(np.full(10, -1.0))) is None


def test_launcher_argv(tmp_path):
    d = DockerLauncher("img:1", "/opt/of/bashrc", n_procs=2, user="")
    argv = d.build_argv("checkMesh", tmp_path)
    assert argv[:3] == ["docker", "run", "--rm"] and "img:1" in argv
    assert f"{tmp_path}:{tmp_path}" in argv
    assert argv[-1].endswith(f"cd {tmp_path} && checkMesh")
    assert d.parallel("simpleFoam") == "mpirun -np 2 simpleFoam -parallel"
    n = NativeLauncher("/opt/of/bashrc", n_procs=1)
    assert n.parallel("simpleFoam") == "simpleFoam"
    assert "source /opt/of/bashrc" in n.build_argv("ls", tmp_path)[2]


def test_launcher_run_and_timeout(tmp_path):
    n = NativeLauncher("/dev/null")
    assert n.run("echo hi", tmp_path, tmp_path / "a.log") == 0
    assert "hi" in (tmp_path / "a.log").read_text()
    assert n.run("exit 3", tmp_path, tmp_path / "b.log") == 3
    assert n.run("sleep 5", tmp_path, tmp_path / "c.log", timeout=0.5) == -9


def test_make_launcher_kinds():
    assert isinstance(make_launcher(LauncherConfig(kind="native")), NativeLauncher)
    assert isinstance(make_launcher(LauncherConfig(kind="docker")), DockerLauncher)
