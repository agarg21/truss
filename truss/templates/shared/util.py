import multiprocessing
import os
import sys
from typing import Optional

import psutil

BLOB_DOWNLOAD_TIMEOUT_SECS = 600  # 10 minutes
# number of seconds to wait for truss server child processes before sending kill signal
CHILD_PROCESS_WAIT_TIMEOUT_SECONDS = 3600


def model_supports_predict_proba(model: object) -> bool:
    if not hasattr(model, "predict_proba"):
        return False
    if hasattr(model, "_check_proba"):  # noqa eg Support Vector Machines *can* predict proba if they made certain choices while training
        try:
            model._check_proba()
            return True
        except AttributeError:
            return False
    return True


def cpu_count():
    """Get the available CPU count for this system.
    Takes the minimum value from the following locations:
    - Total system cpus available on the host.
    - CPU Affinity (if set)
    - Cgroups limit (if set)
    """
    count = os.cpu_count()

    # Check CPU affinity if available
    try:
        affinity_count = len(psutil.Process().cpu_affinity())
        if affinity_count > 0:
            count = min(count, affinity_count)
    except Exception:
        pass

    # Check cgroups if available
    if sys.platform == "linux":
        try:
            with open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us") as f:
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us") as f:
                period = int(f.read())
            cgroups_count = int(quota / period)
            if cgroups_count > 0:
                count = min(count, cgroups_count)
        except Exception:
            pass

    return count


def all_processes_dead(procs: list[multiprocessing.Process]) -> bool:
    for proc in procs:
        if proc.is_alive():
            return False
    return True


def kill_child_processes(
    parent_pid: int,
    timeout_seconds: Optional[float] = CHILD_PROCESS_WAIT_TIMEOUT_SECONDS,
):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    _diag("terminate", parent, children)
    _t0 = __import__("time").monotonic()
    for process in children:
        process.terminate()
    gone, alive = psutil.wait_procs(children, timeout=timeout_seconds)
    _diag(f"after wait {__import__('time').monotonic() - _t0:.2f}s (timeout={timeout_seconds}) alive", parent, alive)
    for process in alive:
        process.kill()


def _diag(label, parent, procs):  # fork-only diagnostic (agarg21/truss diag/teardown)
    path = os.environ.get("TRUSS_DIAG_LOG")
    if not os.environ.get("TRUSS_DIAG_TEARDOWN") or not path:
        return
    import subprocess

    lines = [f"--- {label}: parent pid={parent.pid} cmd={' '.join(parent.cmdline()[:4])!r} n={len(procs)}"]
    for p in procs:
        try:
            info = f"pid={p.pid} ppid={p.ppid()} name={p.name()} status={p.status()} cmd={' '.join(p.cmdline())[:200]!r}"
        except Exception as e:  # noqa
            info = f"pid={p.pid} <{type(e).__name__}>"
        ign = ""
        try:
            if sys.platform == "linux":
                for ln in open(f"/proc/{p.pid}/status"):
                    if ln.startswith("SigIgn"):
                        ign = ln.split()[1]
            else:
                ign = subprocess.run(["ps", "-o", "ignored=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
        lines.append(f"    {info} sigignored={ign}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
