#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ..benchmarkCore import BenchmarkRun
from .criu_restore import (
    _privileged_prefix,
    deepest_single_child,
    single_child,
    wait_until_stopped,
)
from .launcher import (
    LaunchedSide,
    compose_submit_command,
    launch_run_with_start_barrier,
    release_benchmark,
    terminate_and_wait,
)


PROGRESS_INTERVAL_MS = 50

# perf opens its --output file before launching the workload. Close inherited
# non-standard descriptors in the workload before exec so CRIU does not save
# perf_progress.out as an open file while perf continues updating it.
_CLOSE_FDS_EXEC_CODE = r"""
import os
import sys

for name in os.listdir('/proc/self/fd'):
    try:
        fd = int(name)
    except ValueError:
        continue
    if fd <= 2:
        continue
    try:
        os.close(fd)
    except OSError:
        pass

os.execvp(sys.argv[1], sys.argv[1:])
"""



def _memory_node_from_prefix(prefix: str) -> int:
    argv = shlex.split(prefix)
    for index, token in enumerate(argv[:-1]):
        if Path(token).name == "setCpuMemoryAffinity.sh":
            return int(argv[index + 1])
    raise ValueError(
        "could not determine memory node from --prefix; expected "
        "setCpuMemoryAffinity.sh <node>"
    )


def _hugepage_owner(checkpoint_dir: Path) -> str:
    digest = hashlib.sha256(str(checkpoint_dir).encode("utf-8")).hexdigest()[:16]
    return f"criu_checkpoint_{digest}"


def _reserve_script(submit: str) -> Path:
    for token in shlex.split(submit):
        path = Path(token)
        if path.name == "runMosalloc.py":
            script = path.resolve().parent / "reserveHugePages.sh"
            if script.is_file():
                return script
    raise RuntimeError("could not find reserveHugePages.sh from --submit")


def _release_checkpoint_owner(submit: str, prefix: str, owner: str) -> None:
    script = _reserve_script(submit)
    node = _memory_node_from_prefix(prefix)
    env = os.environ.copy()
    env["MOSALLOC_KEEP_HUGEPAGE_POOL"] = "1"
    subprocess.run(
        [str(script), "release", f"--owner={owner}", f"--node={node}"],
        env=env,
        check=True,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one layout-specific CRIU checkpoint for a Mosmodel benchmark."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--layout",
        default=None,
        help=(
            "Optional Mosalloc layout. When omitted, create a native checkpoint "
            "without Mosalloc; this is used for a fixed 4KB SMT co-runner."
        ),
    )
    parser.add_argument("--i-start", type=int, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--submit", default="")
    args = parser.parse_args(argv)

    if args.i_start < 0:
        parser.error("--i-start must be non-negative")
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    return args


def _dump_checkpoint(root_pid: int, images_dir: Path, dump_log: Path) -> None:
    images_dir.mkdir(parents=True)
    command = [
        *_privileged_prefix(),
        "criu",
        "dump",
        "-t",
        str(root_pid),
        "-D",
        str(images_dir),
        "--shell-job",
        "-v4",
        "-o",
        str(dump_log),
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CRIU dump failed with rc={result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}; see {dump_log}"
        )


def _parse_progress_delta(line: str) -> Optional[int]:
    fields = [field.strip() for field in line.split(";")]
    event_index = next(
        (index for index, field in enumerate(fields) if field.startswith("instructions")),
        None,
    )
    if event_index is None:
        return None

    for field in reversed(fields[:event_index]):
        try:
            return int(round(float(field.replace(",", ""))))
        except ValueError:
            continue
    return None


def _read_progress_total(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return 0

    total = 0
    for line in lines:
        delta = _parse_progress_delta(line)
        if delta is not None:
            total += delta
    return total


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    layout = Path(args.layout).resolve() if args.layout else None

    if layout is not None and not layout.is_file():
        raise FileNotFoundError(f"missing layout: {layout}")
    if layout is not None and not args.submit.strip():
        raise ValueError("--layout requires a Mosalloc --submit command")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    work_dir = checkpoint_dir / "work"
    output_dir = checkpoint_dir / "create-output"
    images_dir = checkpoint_dir / "images"
    done_file = checkpoint_dir / "checkpoint.done"
    checkpoint_layout = checkpoint_dir / "layout.csv"
    dump_log = checkpoint_dir / "dump.log"
    progress_output = checkpoint_dir / "perf_progress.out"

    if layout is not None:
        shutil.copy2(layout, checkpoint_layout)
    else:
        try:
            checkpoint_layout.unlink()
        except FileNotFoundError:
            pass

    run = BenchmarkRun(args.benchmark, str(work_dir), str(output_dir))
    run.prerun()
    owner = ""
    submit = args.submit
    if layout is not None:
        owner = _hugepage_owner(checkpoint_dir)
        node = _memory_node_from_prefix(args.prefix)
        submit = (
            f"env MOSALLOC_KEEP_HUGEPAGE_POOL=1 "
            f"MOSALLOC_HUGEPAGES_NODE={node} "
            f"MOSALLOC_HUGEPAGES_OWNER={owner} {args.submit}"
        )
    submit_command = compose_submit_command(
        args.prefix,
        submit,
        None,
        Path(args.benchmark).parent,
    )
    checkpoint_command = shlex.join(
        [
            "perf",
            "stat",
            "-e",
            "instructions:u",
            "-I",
            str(PROGRESS_INTERVAL_MS),
            "-x",
            ";",
            "--no-big-num",
            f"--output={progress_output}",
            "--",
            sys.executable,
            "-c",
            _CLOSE_FDS_EXEC_CODE,
            *shlex.split(submit_command),
        ]
    )

    launched: Optional[LaunchedSide] = None
    try:
        launched = launch_run_with_start_barrier(
            run,
            args.num_threads,
            checkpoint_command,
        )
        if launched.benchmark_pid is None:
            raise RuntimeError("failed to discover checkpoint perf pid")

        if not release_benchmark(launched):
            raise RuntimeError("failed to release benchmark start barrier")

        observed = 0
        while observed < max(args.i_start, 1):
            observed = _read_progress_total(progress_output)
            if launched.proc.poll() is not None:
                raise RuntimeError(
                    "benchmark exited before checkpoint creation: "
                    f"observed={observed}, target={args.i_start}; "
                    f"see {output_dir / 'benchmark.log'}"
                )
            time.sleep(0.05)

        perf_pid = launched.benchmark_pid
        root_pid = single_child(perf_pid)
        benchmark_pid = deepest_single_child(root_pid)
        os.kill(benchmark_pid, signal.SIGSTOP)
        wait_until_stopped(benchmark_pid)

        _dump_checkpoint(root_pid, images_dir, dump_log)

        # benchmark.log is stdout/stderr of the checkpointed process tree.
        # The wrapper can append cleanup output after the dump, but CRIU expects
        # the file size seen at dump time. Restore that exact size after cleanup.
        benchmark_log = output_dir / "benchmark.log"
        benchmark_log_size = benchmark_log.stat().st_size

        terminate_and_wait(launched)
        launched = None
        with benchmark_log.open("r+b") as stream:
            stream.truncate(benchmark_log_size)

        if owner:
            _release_checkpoint_owner(args.submit, args.prefix, owner)
            owner = ""

        done_file.write_text("ok\n", encoding="utf-8")
        print(
            f"created checkpoint layout={layout.stem if layout else 'native'} "
            f"dir={checkpoint_dir} observed={observed}"
        )
        return 0
    finally:
        terminate_and_wait(launched)
        if owner:
            _release_checkpoint_owner(args.submit, args.prefix, owner)


if __name__ == "__main__":
    raise SystemExit(main())
