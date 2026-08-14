from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..benchmarkCore import BenchmarkRun


@dataclass
class LaunchedSide:
    proc: subprocess.Popen
    sid: int
    benchmark_pgid: Optional[int] = None
    benchmark_pid: Optional[int] = None
    release_fd: Optional[int] = None
    start_gate_stopped: bool = False


def compose_submit_command(
    prefix: str,
    submit: str,
    loop_until: Optional[int],
    benchmarks_root,
) -> str:
    """
    Preserve the historical order:
      normal:      <prefix> <submit>
      loop_until:  <prefix> timeout <sec> loopForever.sh <submit>
    """
    parts: list[str] = []

    if prefix.strip():
        parts.extend(shlex.split(prefix))
    if loop_until is not None and loop_until > 0:
        loop_forever = benchmarks_root / "loopForever.sh"
        parts.extend(["timeout", str(loop_until), str(loop_forever)])
    if submit.strip():
        parts.extend(shlex.split(submit))

    return " ".join(parts)


def _build_env(num_threads: int, output_dir) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(num_threads),
            "OMP_THREAD_LIMIT": str(num_threads),
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "true",
            "OMP_SCHEDULE": "static",
            "MOSMODEL_RUN_OUT_DIR": str(output_dir),
        }
    )
    return env



def _read_int_file(path: Path) -> Optional[int]:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None



def _wait_for_benchmark_pgid(
    path: Path,
    supervisor_proc: subprocess.Popen,
) -> Optional[int]:
    """Wait until the supervisor publishes the benchmark PGID or exits."""
    while True:
        value = _read_int_file(path)
        if value is not None:
            return value
        if supervisor_proc.poll() is not None:
            return _read_int_file(path)
        time.sleep(0.01)



def _spawn_side(
    run: BenchmarkRun,
    env: dict,
    cmd: list[str],
    wrapped_cmd: str,
    *,
    benchmark_pgid_file: Optional[Path] = None,
    pass_fds: tuple[int, ...] = (),
    release_fd: Optional[int] = None,
) -> LaunchedSide:
    print(f"{run._benchmark_dir}: launching")
    print(f"  cwd={run._run_dir}")
    print(f"  wrapped_cmd={wrapped_cmd!r}")
    print(f"  argv={cmd!r}")

    if benchmark_pgid_file is not None:
        try:
            benchmark_pgid_file.unlink()
        except FileNotFoundError:
            pass

    proc = subprocess.Popen(
        cmd,
        cwd=run._run_dir,
        env=env,
        stdout=run._log_file,
        stderr=run._log_file,
        start_new_session=True,
        pass_fds=pass_fds,
    )
    sid = os.getsid(proc.pid)
    benchmark_pgid = None
    if benchmark_pgid_file is not None:
        benchmark_pgid = _wait_for_benchmark_pgid(benchmark_pgid_file, proc)
    benchmark_pid = benchmark_pgid
    print(f"  sid={sid}")
    print(f"  benchmark_pgid={benchmark_pgid}")
    print(f"  benchmark_pid={benchmark_pid}")
    return LaunchedSide(
        proc=proc,
        sid=sid,
        benchmark_pgid=benchmark_pgid,
        benchmark_pid=benchmark_pid,
        release_fd=release_fd,
    )



def _inner_benchmark_launcher(
    run: BenchmarkRun,
    shell_cmd: str,
    *,
    release_read_fd: Optional[int] = None,
) -> tuple[list[str], Path]:
    benchmark_pgid_file = Path(run._run_dir) / ".benchmark_pgid"
    helper_script = Path(__file__).with_name("side_supervisor.py").resolve()

    benchmark_argv = shlex.split(shell_cmd) if shell_cmd else []
    benchmark_argv.append("./run.sh")

    cmd = [sys.executable, str(helper_script)]
    if release_read_fd is not None:
        cmd.extend(["--release-fd", str(release_read_fd)])
    cmd.extend([str(benchmark_pgid_file), "--", *benchmark_argv])
    return cmd, benchmark_pgid_file




_RUN_SH_GATE_MARKER = "# MOSMODEL_NATIVE_START_GATE"


def _install_run_sh_start_gate(run: BenchmarkRun) -> Path:
    """Stop the copied run.sh immediately after it starts.

    The gate is deliberately inside run.sh so CPU/NUMA affinity and runMosalloc
    setup happen before the controller enables measurement.
    """
    run_sh = Path(run._run_dir) / "run.sh"
    text = run_sh.read_text(encoding="utf-8")
    if _RUN_SH_GATE_MARKER not in text:
        snippet = (
            f"{_RUN_SH_GATE_MARKER}\n"
            'if [ -n "${MOSMODEL_START_GATE_PID_FILE:-}" ]; then\n'
            '    printf "%s\\n" "$$" > "$MOSMODEL_START_GATE_PID_FILE"\n'
            '    unset MOSMODEL_START_GATE_PID_FILE\n'
            '    kill -STOP "$$"\n'
            "fi\n"
        )
        lines = text.splitlines(keepends=True)
        if lines and lines[0].startswith("#!"):
            text = lines[0] + snippet + "".join(lines[1:])
        else:
            text = snippet + text
        run_sh.write_text(text, encoding="utf-8")
    return run_sh


def _wait_for_benchmark_pid(
    path: Path,
    supervisor_proc: subprocess.Popen,
) -> int:
    """Wait until run.sh reaches the native start gate and publishes its PID."""
    while True:
        value = _read_int_file(path)
        if value is not None:
            return value
        if supervisor_proc.poll() is not None:
            raise RuntimeError(
                f"benchmark exited before reaching start gate: {path}"
            )
        time.sleep(0.01)


def launch_run_at_benchmark_start(
    run: BenchmarkRun,
    num_threads: int,
    submit_command: str,
) -> LaunchedSide:
    """Run affinity/mosalloc setup, then stop at the first line of run.sh."""
    _install_run_sh_start_gate(run)

    benchmark_pid_file = Path(run._run_dir) / ".benchmark_start_pid"
    benchmark_pid_file.unlink(missing_ok=True)

    env = _build_env(num_threads, run._output_dir)
    env["MOSMODEL_START_GATE_PID_FILE"] = str(benchmark_pid_file)

    shell_cmd = submit_command.strip() if submit_command else ""
    cmd, benchmark_pgid_file = _inner_benchmark_launcher(run, shell_cmd)
    wrapped_cmd = " ".join(shlex.quote(part) for part in cmd)
    launched = _spawn_side(
        run,
        env,
        cmd,
        wrapped_cmd,
        benchmark_pgid_file=benchmark_pgid_file,
    )

    benchmark_pid = _wait_for_benchmark_pid(
        benchmark_pid_file,
        launched.proc,
    )
    launched.benchmark_pid = benchmark_pid
    launched.start_gate_stopped = True
    print(f"  benchmark_start_pid={benchmark_pid}")
    return launched


def resume_benchmark_start_gate(launched: Optional[LaunchedSide]) -> bool:
    if (
        launched is None
        or not launched.start_gate_stopped
        or launched.benchmark_pid is None
    ):
        return False
    try:
        os.kill(launched.benchmark_pid, signal.SIGCONT)
    except (ProcessLookupError, PermissionError):
        return False
    launched.start_gate_stopped = False
    return True


def launch_run_with_start_barrier(
    run: BenchmarkRun,
    num_threads: int,
    submit_command: str,
) -> LaunchedSide:
    """
    Launch the benchmark helper in a fresh side session, but block the benchmark
    leader on an inherited release pipe before it execs ./run.sh. The controller
    can attach a detached sampler to the stable benchmark leader PID and then
    release it, avoiding both the old attach race and the direct-launcher perf
    coupling.
    """
    env = _build_env(num_threads, run._output_dir)
    shell_cmd = submit_command.strip() if submit_command else ""

    release_read_fd, release_write_fd = os.pipe()
    try:
        cmd, benchmark_pgid_file = _inner_benchmark_launcher(
            run,
            shell_cmd,
            release_read_fd=release_read_fd,
        )
        wrapped_cmd = " ".join(shlex.quote(part) for part in cmd)
        launched = _spawn_side(
            run,
            env,
            cmd,
            wrapped_cmd,
            benchmark_pgid_file=benchmark_pgid_file,
            pass_fds=(release_read_fd,),
            release_fd=release_write_fd,
        )
    finally:
        os.close(release_read_fd)

    return launched



def release_benchmark(launched: Optional[LaunchedSide]) -> bool:
    if launched is None or launched.release_fd is None:
        return False
    try:
        os.write(launched.release_fd, b"1")
        return True
    except OSError:
        return False
    finally:
        try:
            os.close(launched.release_fd)
        except OSError:
            pass
        launched.release_fd = None




def terminate_many_and_wait(
    launched_sides: tuple[Optional[LaunchedSide], ...],
    grace_sec: float = 2.0,
) -> tuple[Optional[int], ...]:
    """Gracefully stop multiple side sessions at approximately the same time.

    SIGTERM is sent to every live side before waiting for any one side. This lets
    wrappers such as runMosalloc.py unwind and release their logical huge-page
    reservations. SIGKILL is used only for sessions that remain after the grace
    period.
    """
    sides = tuple(launched_sides)

    for launched in sides:
        if launched is None or launched.release_fd is None:
            continue
        try:
            os.close(launched.release_fd)
        except OSError:
            pass
        launched.release_fd = None

    for launched in sides:
        if _side_has_live_processes(launched):
            signal_side(launched, signal.SIGTERM)
            if (
                launched is not None
                and launched.start_gate_stopped
                and launched.benchmark_pid is not None
            ):
                try:
                    os.kill(launched.benchmark_pid, signal.SIGCONT)
                except (ProcessLookupError, PermissionError):
                    pass
                launched.start_gate_stopped = False

    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if not any(_side_has_live_processes(launched) for launched in sides):
            break
        time.sleep(0.05)

    for launched in sides:
        if _side_has_live_processes(launched):
            signal_side(launched, signal.SIGKILL)

    results: list[Optional[int]] = []
    for launched in sides:
        if launched is None:
            results.append(None)
            continue
        try:
            results.append(launched.proc.wait(timeout=1.0))
        except subprocess.TimeoutExpired:
            results.append(launched.proc.poll())
    return tuple(results)


def terminate_and_wait(
    launched: Optional[LaunchedSide],
    grace_sec: float = 2.0,
) -> Optional[int]:
    return terminate_many_and_wait((launched,), grace_sec=grace_sec)[0]



def signal_side(launched: Optional[LaunchedSide], sig: int) -> None:
    if launched is None:
        return

    sig_name = signal.Signals(sig).name
    subprocess.run(
        ["pkill", f"-{sig_name}", "-s", str(launched.sid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )





def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True



def _ps_snapshot() -> list[tuple[int, int, int]]:
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid=,pgid="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    rows: list[tuple[int, int, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, pgid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        if pid > 0:
            rows.append((pid, ppid, pgid))
    return rows


def _ps_snapshot_detailed() -> dict[int, dict[str, object]]:
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid=,pgid=,sid=,stat=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    rows: dict[int, dict[str, object]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pgid = int(parts[2])
            sid = int(parts[3])
        except ValueError:
            continue
        if pid <= 0:
            continue
        rows[pid] = {
            "pid": pid,
            "ppid": ppid,
            "pgid": pgid,
            "sid": sid,
            "stat": parts[4],
            "cmd": parts[5],
        }
    return rows


def benchmark_group_pids(launched: Optional[LaunchedSide]) -> list[int]:
    if launched is None:
        return []

    rows = _ps_snapshot()
    targets: set[int] = set()
    if launched.benchmark_pgid is not None:
        targets.update(pid for pid, _ppid, pgid in rows if pgid == launched.benchmark_pgid)

    if launched.benchmark_pid is not None and _pid_alive(launched.benchmark_pid):
        by_parent: dict[int, list[int]] = {}
        for pid, ppid, _pgid in rows:
            by_parent.setdefault(ppid, []).append(pid)
        stack = [launched.benchmark_pid]
        while stack:
            pid = stack.pop()
            if pid in targets:
                continue
            targets.add(pid)
            stack.extend(by_parent.get(pid, []))

    return sorted(pid for pid in targets if _pid_alive(pid))


def signal_benchmark_group(launched: Optional[LaunchedSide], sig: int) -> bool:
    if launched is None or launched.benchmark_pgid is None:
        return False
    try:
        os.killpg(launched.benchmark_pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False



def _side_has_live_processes(launched: Optional[LaunchedSide]) -> bool:
    if launched is None:
        return False
    for row in _ps_snapshot_detailed().values():
        if row["sid"] != launched.sid:
            continue
        if not str(row["stat"]).startswith("Z"):
            return True
    return False


