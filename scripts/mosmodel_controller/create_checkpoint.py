#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

from ..benchmarkCore import BenchmarkRun
from .checkpoint import (
    BENCHMARK_LOG_PATH,
    IMAGES_DIR,
    LAYOUT_PATH,
    PERF_PROGRESS_PATH,
    RUNTIME_MOUNT,
    TREE_EXEC_PATH,
    WORK_MOUNT,
    WORK_SNAPSHOT_DIR,
    checkpoint_is_complete,
    dump,
    hugepage_owner,
    mark_complete,
    memory_node,
    mosalloc_library_from_submit,
    read_progress,
    release_hugepages,
    runtime_root,
    snapshot_runtime_artifact,
    snapshot_work,
    virtualize_command,
    write_metadata,
)
from .launcher import (
    compose_submit_command,
    launch_run_with_start_barrier,
    release_benchmark,
    terminate_and_wait,
)
from .namespaces import in_checkpoint_namespace, run_checkpoint_namespace
from .process_tree import deepest_single_child, single_child, wait_until_stopped
from .tcp import tree_has_established_tcp

PROGRESS_MS = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a portable CRIU checkpoint')
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--runtime-dir', default=str(runtime_root()))
    parser.add_argument('--layout')
    parser.add_argument('--i-start', type=int, required=True)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--submit', default='')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--inside-namespace', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.i_start <= 0 or args.num_threads <= 0:
        parser.error('--i-start must be > 0 and --num-threads must be > 0')
    return args


def _close_log(run: BenchmarkRun) -> None:
    log = getattr(run, '_log_file', None)
    if log is not None:
        log.flush()
        log.close()
        run._log_file = None


def _kill_tree(root_pid: int | None) -> None:
    if root_pid is None:
        return
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _perf_command(submit: str) -> str:
    return shlex.join([
        'perf', 'stat', '-e', 'instructions:u', '-I', str(PROGRESS_MS), '-x', ';',
        '--no-big-num', f'--output={PERF_PROGRESS_PATH}', '--',
        sys.executable, str(TREE_EXEC_PATH), '--', *shlex.split(submit),
    ])


def _wait_for_target(launched, target: int) -> int:
    observed = 0
    while observed < max(target, 1):
        observed = read_progress(PERF_PROGRESS_PATH)
        if launched.proc.poll() is not None:
            raise RuntimeError(
                f'benchmark exited before checkpoint: observed={observed} target={target}; '
                f'see {BENCHMARK_LOG_PATH}'
            )
        time.sleep(PROGRESS_MS / 1000)
    return observed


def _create(args: argparse.Namespace) -> int:
    if not in_checkpoint_namespace() or os.getpid() != 1:
        raise RuntimeError('checkpoint creation must run as PID 1 in its private namespace')

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    layout = Path(args.layout).resolve() if args.layout else None
    if layout is not None and not args.submit.strip():
        raise ValueError('--layout requires --submit')

    run = BenchmarkRun(args.benchmark, str(WORK_MOUNT), str(WORK_MOUNT))
    if layout is not None:
        shutil.copy2(layout, LAYOUT_PATH)
    run.prerun()

    prefix = virtualize_command(args.prefix, runtime_dir, False)
    submit = virtualize_command(args.submit, runtime_dir, layout is not None)
    mosalloc_library = mosalloc_library_from_submit(submit)
    owner = ''
    if layout is not None:
        owner = hugepage_owner(checkpoint_dir)
        submit = (
            'env MOSALLOC_KEEP_HUGEPAGE_POOL=1 '
            f'MOSALLOC_HUGEPAGES_NODE={memory_node(prefix)} '
            f'MOSALLOC_HUGEPAGES_OWNER={owner} {submit}'
        )

    command = compose_submit_command(prefix, submit, None, Path(args.benchmark).parent)
    launched = None
    root_pid = None
    try:
        launched = launch_run_with_start_barrier(run, args.num_threads, _perf_command(command))
        if launched.benchmark_pid is None or not release_benchmark(launched):
            raise RuntimeError('failed to start checkpoint benchmark')

        observed = _wait_for_target(launched, args.i_start)
        root_pid = single_child(launched.benchmark_pid)
        benchmark_pid = deepest_single_child(root_pid)
        os.kill(benchmark_pid, signal.SIGSTOP)
        wait_until_stopped(benchmark_pid)

        tcp_established = tree_has_established_tcp(root_pid)
        if tcp_established:
            print(
                '[CRIU tcp] detected established TCP socket(s); '
                'enabling --tcp-established for checkpoint dump'
            )
        dump(
            root_pid, checkpoint_dir / IMAGES_DIR, run_dir / 'dump.log',
            tcp_established=tcp_established,
        )
        log_size = BENCHMARK_LOG_PATH.stat().st_size

        terminate_and_wait(launched)
        launched = None
        _close_log(run)
        with BENCHMARK_LOG_PATH.open('r+b') as stream:
            stream.truncate(log_size)

        if owner:
            release_hugepages(submit, prefix, owner)
            owner = ''

        snapshot_work(checkpoint_dir / WORK_SNAPSHOT_DIR)
        runtime_artifacts: list[str] = []
        if mosalloc_library is not None:
            if not mosalloc_library.is_file():
                raise FileNotFoundError(f'mosalloc runtime library is missing: {mosalloc_library}')
            runtime_artifacts.append(snapshot_runtime_artifact(checkpoint_dir, mosalloc_library))
        write_metadata(
            checkpoint_dir, Path(args.benchmark), layout,
            args.i_start, observed, args.num_threads, runtime_artifacts,
            tcp_established=tcp_established,
        )
        mark_complete(checkpoint_dir)
        print(
            f"created portable checkpoint layout={layout.stem if layout else 'native'} "
            f'dir={checkpoint_dir} observed={observed}'
        )
        return 0
    finally:
        if root_pid is None and launched is not None and launched.benchmark_pid is not None:
            try:
                root_pid = single_child(launched.benchmark_pid)
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                pass
        _kill_tree(root_pid)
        terminate_and_wait(launched)
        _close_log(run)
        if owner:
            release_hugepages(submit, prefix, owner)


def _inside_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable, '-m', 'scripts.mosmodel_controller.create_checkpoint',
        '--inside-namespace', '--benchmark', args.benchmark,
        '--checkpoint-dir', str(Path(args.checkpoint_dir).resolve()),
        '--run-dir', str(Path(args.run_dir).resolve()),
        '--runtime-dir', str(Path(args.runtime_dir).resolve()),
        '--i-start', str(args.i_start), '--num-threads', str(args.num_threads),
    ]
    for flag, value in (('--layout', args.layout), ('--prefix', args.prefix), ('--submit', args.submit)):
        if value:
            argv.extend([flag, str(Path(value).resolve()) if flag == '--layout' else value])
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inside_namespace:
        return _create(args)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    if not runtime_dir.is_dir():
        raise FileNotFoundError(runtime_dir)
    if args.layout and not Path(args.layout).resolve().is_file():
        raise FileNotFoundError(args.layout)

    if checkpoint_dir.exists():
        if checkpoint_is_complete(checkpoint_dir) and not args.force:
            print(f'checkpoint already exists: {checkpoint_dir}')
            return 0
        if not args.force:
            raise FileExistsError(f'checkpoint exists but is incomplete: {checkpoint_dir}')
        shutil.rmtree(checkpoint_dir)

    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / 'work').mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    rc = run_checkpoint_namespace(run_dir / 'work', runtime_dir, _inside_argv(args))
    if rc == 0:
        shutil.rmtree(run_dir)
    else:
        print(f'checkpoint creation failed rc={rc}; preserving {run_dir}', file=sys.stderr)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
