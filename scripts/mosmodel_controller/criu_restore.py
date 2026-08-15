from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .cgroup import CgroupV2
from .checkpoint import (
    IMAGES_DIR,
    RUNTIME_SNAPSHOT_DIR,
    RESTORE_WORK_DIR,
    read_metadata,
    reset_restore_work,
    run_root,
    runtime_root,
)
from .launcher import LaunchedSide
from .namespaces import restore_namespace_command
from .process_tree import (
    process_tree_pids,
    deepest_single_child,
    host_pid_for_namespace_pid,
    read_process_group,
    wait_until_stopped,
)

POLL_SEC = 0.001


@dataclass(frozen=True)
class StoppedRestore:
    root_pid: int
    benchmark_pid: int
    pgid: int
    sid: int
    criu_process: subprocess.Popen[str]
    cgroup: CgroupV2

    def as_launched_side(self) -> LaunchedSide:
        return LaunchedSide(
            proc=self.criu_process,
            sid=self.sid,
            benchmark_pgid=self.pgid,
            benchmark_pid=self.benchmark_pid,
        )


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except PermissionError:
        result = run_root(['cat', str(path)])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout


def _failed(process: subprocess.Popen[str], log: Path) -> None:
    rc = process.poll()
    if rc is not None:
        raise RuntimeError(f'CRIU restore failed rc={rc}; see {log}')


def _cleanup(
    process: subprocess.Popen[str],
    root_pid: int | None,
    pgid: int | None,
    cgroup: CgroupV2 | None,
) -> None:
    if cgroup is not None:
        try:
            cgroup.kill_and_remove()
        except Exception:
            pass
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        elif root_pid is not None:
            os.kill(root_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if process.poll() is None:
        process.kill()


def restore_stopped(
    *,
    checkpoint_dir: Path,
    checkpoint_archive_dir: Path,
    output_dir: Path,
    prefix: str,
) -> StoppedRestore:
    checkpoint_dir = checkpoint_dir.resolve()
    archive_dir = checkpoint_archive_dir.resolve()

    metadata = read_metadata(archive_dir)
    tcp_established = bool(metadata.get('tcp_established', False))
    runtime_artifacts = metadata.get('runtime_artifacts')
    if runtime_artifacts is None:
        # Schema v1 checkpoints predate runtime artifact snapshots. Native
        # checkpoints remain portable; mosalloc-backed checkpoints must be
        # regenerated because CRIU validates the mapped libmosalloc build-ID.
        if metadata.get('layout') is not None:
            raise RuntimeError(
                f'mosalloc-backed checkpoint predates runtime artifact snapshots: {archive_dir}; '
                'regenerate this checkpoint with the current checkpoint creator'
            )
        runtime_artifacts = []

    artifact_dir = archive_dir / RUNTIME_SNAPSHOT_DIR
    for relative in runtime_artifacts:
        path = artifact_dir / relative
        if not path.is_file():
            raise RuntimeError(
                f'checkpoint runtime artifact is missing: {path}; '
                'regenerate this checkpoint'
            )

    # checkpoint_dir is only the mutable restore workspace.  The immutable
    # archive is the source of truth: CRIU reads images directly from it, while
    # reset_restore_work() recreates only the mutable /work contents.
    reset_restore_work(archive_dir, checkpoint_dir)
    images = archive_dir / IMAGES_DIR

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pidfile = output_dir / 'restore.pid'
    log = output_dir / 'restore.log'
    pidfile.unlink(missing_ok=True)

    # Keep affinity/NUMA wrappers outside the private PID namespace.
    # In particular, setCpuMemoryAffinity.sh --smt uses helper subprocesses;
    # allowing those helpers into the namespace can occupy image PIDs (7, 9,
    # 10, ...), making CRIU fail with EEXIST while recreating the saved tree.
    criu = [
        'criu', 'restore',
        '-D', str(images),
        '-W', str(output_dir),
        '--leave-stopped',
        '--pidfile', str(pidfile),
        '--manage-cgroups=ignore',
        '-v4',
        '-o', str(log),
    ]
    if tcp_established:
        criu.append('--tcp-established')
        print(
            f'[CRIU tcp] enabling established TCP restore for {archive_dir}'
        )
    runtime = runtime_root()
    namespace_command = restore_namespace_command(
        checkpoint_dir / RESTORE_WORK_DIR,
        runtime,
        criu,
        runtime_artifact_dir=artifact_dir if runtime_artifacts else None,
    )
    launch_command = [*shlex.split(prefix), *namespace_command]
    process = subprocess.Popen(
        launch_command,
        cwd=runtime,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    root_pid = None
    pgid = None
    cgroup = None
    try:
        while not pidfile.is_file():
            _failed(process, log)
            time.sleep(POLL_SEC)
        namespace_pid = int(_read(pidfile).strip())

        while root_pid is None:
            root_pid = host_pid_for_namespace_pid(process.pid, namespace_pid)
            if root_pid is None:
                _failed(process, log)
                time.sleep(POLL_SEC)
        print(f'[CRIU pidns] namespace_root_pid={namespace_pid} host_root_pid={root_pid}')

        while True:
            try:
                benchmark_pid = deepest_single_child(root_pid)
                if benchmark_pid != root_pid:
                    wait_until_stopped(root_pid)
                    wait_until_stopped(benchmark_pid)
                    break
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                pass
            _failed(process, log)
            time.sleep(POLL_SEC)

        pgid, sid = read_process_group(root_pid)
        restored = process_tree_pids(root_pid)
        for pid in restored:
            wait_until_stopped(pid)

        cgroup = CgroupV2.create_for_pid(root_pid, str(output_dir))
        cgroup.add_pids(restored)
        print(
            f'[CRIU cgroup] created {cgroup.perf_name}: '
            f'root_pid={root_pid} restored_pids={restored}'
        )
        return StoppedRestore(root_pid, benchmark_pid, pgid, sid, process, cgroup)
    except Exception:
        _cleanup(process, root_pid, pgid, cgroup)
        raise
