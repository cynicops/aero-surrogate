"""How OpenFOAM commands are executed: in the official Docker image or a native install.

Both launchers take a shell command string (e.g. ``"checkMesh"``) plus a working directory
and return a ``subprocess.Popen`` writing to a log file. The runner never needs to know
which one is in use.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import uuid
from pathlib import Path

from ..config import LauncherConfig

MPI_ENV = {"OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"}


class Launcher:
    version_label = "unknown"
    n_procs = 1

    def build_argv(self, cmd: str, cwd: Path) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def popen(self, cmd: str, cwd: str | Path, log_path: str | Path) -> subprocess.Popen:
        cwd = Path(cwd).resolve()
        log = open(log_path, "w")
        argv = self.build_argv(cmd, cwd)
        log.write(f"# launcher argv: {' '.join(shlex.quote(a) for a in argv)}\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        proc._aero_log = log  # closed by the runner
        return proc

    def run(self, cmd: str, cwd: str | Path, log_path: str | Path, timeout: float | None = None) -> int:
        proc = self.popen(cmd, cwd, log_path)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill(proc)
            return -9
        finally:
            proc._aero_log.close()

    def kill(self, proc: subprocess.Popen) -> None:
        proc.kill()
        proc.wait()

    def parallel(self, solver: str) -> str:
        if self.n_procs <= 1:
            return solver
        return f"mpirun -np {self.n_procs} {solver} -parallel"


class NativeLauncher(Launcher):
    """OpenFOAM installed on the machine (laptop / cluster). UNTESTED in the cloud workspace."""

    def __init__(self, bashrc: str, n_procs: int = 1, version_label: str = "OpenFOAM (native)"):
        self.bashrc = bashrc
        self.n_procs = n_procs
        self.version_label = version_label

    def build_argv(self, cmd: str, cwd: Path) -> list[str]:
        env = " ".join(f"{k}={v}" for k, v in MPI_ENV.items())
        script = f"source {shlex.quote(self.bashrc)} >/dev/null 2>&1; export {env}; cd {shlex.quote(str(cwd))} && {cmd}"
        return ["bash", "-c", script]


class DockerLauncher(Launcher):
    """Runs each command in a fresh container of the official ESI image.

    The case directory is bind-mounted at the same absolute path inside the container.
    """

    def __init__(
        self,
        image: str,
        bashrc: str,
        n_procs: int = 1,
        version_label: str = "OpenFOAM (docker)",
        user: str = "auto",
    ):
        self.image = image
        self.bashrc = bashrc
        self.n_procs = n_procs
        self.version_label = version_label
        if user == "auto":
            uid = os.getuid() if hasattr(os, "getuid") else 0
            user = "" if uid == 0 else f"{uid}:{os.getgid()}"
        self.user = user
        self._names: dict[int, str] = {}

    def build_argv(self, cmd: str, cwd: Path) -> list[str]:
        name = f"aero-{uuid.uuid4().hex[:10]}"
        argv = ["docker", "run", "--rm", "--name", name, "--shm-size=1g",
                "-v", f"{cwd}:{cwd}", "-w", str(cwd)]
        for k, v in MPI_ENV.items():
            argv += ["-e", f"{k}={v}"]
        if self.user:
            argv += ["--user", self.user]
        argv += [self.image, "bash", "-c",
                 f"source {shlex.quote(self.bashrc)} >/dev/null 2>&1; cd {shlex.quote(str(cwd))} && {cmd}"]
        self._last_name = name
        return argv

    def popen(self, cmd, cwd, log_path):
        proc = super().popen(cmd, cwd, log_path)
        self._names[proc.pid] = self._last_name
        return proc

    def kill(self, proc: subprocess.Popen) -> None:
        name = self._names.pop(proc.pid, None)
        if name:
            subprocess.run(["docker", "kill", name], capture_output=True)
        proc.kill()
        proc.wait()


def make_launcher(cfg: LauncherConfig) -> Launcher:
    if cfg.kind == "docker":
        return DockerLauncher(cfg.image, cfg.bashrc, cfg.n_procs, cfg.version_label, cfg.docker_user)
    if cfg.kind == "native":
        return NativeLauncher(cfg.bashrc, cfg.n_procs, cfg.version_label)
    raise ValueError(f"unknown launcher kind '{cfg.kind}' (use docker | native; ssh/slurm: see runner.remote)")
