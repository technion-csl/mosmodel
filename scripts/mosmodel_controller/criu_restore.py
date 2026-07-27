from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .launcher import LaunchedSide


def _privileged_prefix() -> list[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    return ["sudo"]


def _run_privileged(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_privileged_prefix(), *command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def copy_checkpoint(checkpoint_dir: Path, run_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _run_privileged(["cp", "-a", f"{checkpoint_dir}/.", str(run_dir)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return run_dir


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except PermissionError:
        result = _run_privileged(["cat", str(path)])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout


def _read_process_group(pid: int) -> tuple[int, int]:
    text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[2]), int(fields[3])


def _children(pid: int) -> list[int]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    text = path.read_text(encoding="utf-8").strip()
    return [int(value) for value in text.split()] if text else []


def single_child(pid: int) -> int:
    children = _children(pid)
    if len(children) != 1:
        raise RuntimeError(f"expected one child of pid {pid}, found {children}")
    return children[0]


def deepest_single_child(pid: int) -> int:
    current = pid
    while True:
        children = _children(current)
        if not children:
            return current
        if len(children) != 1:
            raise RuntimeError(f"expected one child of pid {current}, found {children}")
        current = children[0]


def wait_until_stopped(pid: int) -> None:
    stat_path = Path("/proc") / str(pid) / "stat"
    while True:
        text = stat_path.read_text(encoding="utf-8")
        state = text[text.rfind(")") + 2 :].split()[0]
        if state in {"T", "t"}:
            return
        time.sleep(0.001)


@dataclass(frozen=True)
class StoppedRestore:
    root_pid: int
    benchmark_pid: int
    pgid: int
    sid: int
    criu_process: subprocess.Popen[str]

    def as_launched_side(self) -> LaunchedSide:
        return LaunchedSide(
            proc=self.criu_process,
            sid=self.sid,
            benchmark_pgid=self.pgid,
            benchmark_pid=self.benchmark_pid,
        )


def _kill_restore(
    root_pid: Optional[int],
    pgid: Optional[int],
    criu_process: subprocess.Popen[str],
) -> None:
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        elif root_pid is not None:
            os.kill(root_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    if criu_process.poll() is None:
        criu_process.kill()


def restore_stopped(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    prefix: str,
) -> StoppedRestore:
    checkpoint_dir = checkpoint_dir.resolve()
    images_dir = checkpoint_dir / "images"
    if not (checkpoint_dir / "checkpoint.done").is_file():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing CRIU images: {images_dir}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pidfile = output_dir / "restore.pid"
    restore_log = output_dir / "restore.log"

    command = [
        *_privileged_prefix(),
        *shlex.split(prefix),
        "criu",
        "restore",
        "-D",
        str(images_dir),
        "--leave-stopped",
        "--pidfile",
        str(pidfile),
        "--shell-job",
        "--manage-cgroups=ignore",
        "-v4",
        "-o",
        str(restore_log),
    ]
    criu_process = subprocess.Popen(
        command,
        cwd=checkpoint_dir / "work",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    root_pid: Optional[int] = None
    pgid: Optional[int] = None
    try:
        while not pidfile.is_file():
            returncode = criu_process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"CRIU restore failed with rc={returncode}; see {restore_log}"
                )
            time.sleep(0.001)

        root_pid = int(_read_text(pidfile).strip())
        while True:
            try:
                benchmark_pid = deepest_single_child(root_pid)
                if benchmark_pid == root_pid:
                    raise RuntimeError("restored benchmark child is not visible yet")
                wait_until_stopped(root_pid)
                wait_until_stopped(benchmark_pid)
                break
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                returncode = criu_process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"CRIU restore failed with rc={returncode}; see {restore_log}"
                    )
                time.sleep(0.001)

        pgid, sid = _read_process_group(root_pid)
        return StoppedRestore(root_pid, benchmark_pid, pgid, sid, criu_process)
    except Exception:
        _kill_restore(root_pid, pgid, criu_process)
        raise
