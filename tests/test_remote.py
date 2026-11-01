"""Only command construction is tested; SSH/SLURM execution is UNTESTED (no remote host)."""
from aerosurrogate.runner.remote import UNTESTED, SlurmConfig, SSHConfig, allrun_script, slurm_script, ssh_commands


def test_marked_untested():
    assert UNTESTED


def test_ssh_commands(tmp_path):
    case = tmp_path / "c42"
    case.mkdir()
    push, run, pull = ssh_commands(case, SSHConfig(host="me@box", remote_root="/scratch/aero/"))
    assert push[0] == "rsync" and push[-1] == "me@box:/scratch/aero/c42/"
    assert run[0] == "ssh" and "cd /scratch/aero/c42 && bash Allrun" in run[-1]
    assert pull[-2] == "me@box:/scratch/aero/c42/"


def test_slurm_script(tmp_path):
    s = slurm_script(tmp_path, "aero1", SlurmConfig(partition="short", n_procs=8, modules=("openfoam/2512",)))
    assert "#SBATCH --ntasks=8" in s and "#SBATCH --partition=short" in s
    assert "module load openfoam/2512" in s and s.strip().endswith("bash Allrun")


def test_allrun_parallel_and_serial():
    assert "mpirun -np 4 simpleFoam -parallel" in allrun_script(4)
    assert "mpirun" not in allrun_script(1)
