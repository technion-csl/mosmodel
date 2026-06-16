#! /usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .cli import find_benchmarks_root
from .single_controller import SingleController, build_single_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one ST benchmark with optional instruction-interval control. "
            "In interval mode the benchmark is launched behind a start barrier, "
            "a detached instruction sampler is attached, the benchmark is released, "
            "measurement perf is enabled at I_start, disabled at I_end, and the "
            "benchmark is terminated cleanly before finalizing outputs without postrun."
        )
    )

    parser.add_argument("--benchmark", required=True, help="Path to benchmark directory")
    parser.add_argument("--run-dir", required=True, help="Run directory for the benchmark")
    parser.add_argument("--output-dir", required=True, help="Output directory for the benchmark")
    parser.add_argument(
        "--output-target",
        required=True,
        help="Artifact that Make expects, usually .../perf.out",
    )
    parser.add_argument("-n", "--num-threads", type=int, default=1)
    parser.add_argument("--prefix", default="", help="Command prefix")
    parser.add_argument("--submit", default="", help="Submit command before ./run.sh")
    parser.add_argument(
        "--loop-until",
        type=int,
        default=None,
        help="If > 0, run under timeout <sec> loopForever.sh",
    )
    parser.add_argument(
        "--clean-threshold",
        type=int,
        default=1024 * 1024,
        help="Delete files larger than this size from output dirs",
    )
    parser.add_argument(
        "--exclude-files",
        nargs="*",
        default=[],
        help="Do not delete these files during cleanup",
    )
    parser.add_argument(
        "--sample-instructions",
        action="store_true",
        default=False,
        help="Attach a detached perf stat sampler to track cumulative instructions",
    )
    parser.add_argument(
        "--progress-perf-binary",
        default="perf",
        help="perf binary used for sampled instructions and measurement perf",
    )
    parser.add_argument(
        "--progress-interval-ms",
        type=int,
        default=50,
        help="perf stat instruction-sampling interval in milliseconds",
    )
    parser.add_argument(
        "--i-start",
        type=int,
        default=None,
        help="Sampled cumulative instructions threshold marking interval start",
    )
    parser.add_argument(
        "--i-end",
        type=int,
        default=None,
        help="Sampled cumulative instructions threshold marking interval end",
    )
    parser.add_argument(
        "--termination-grace-sec",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGTERM before SIGKILL during interval cleanup",
    )

    args = parser.parse_args()

    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if args.progress_interval_ms <= 0:
        parser.error("--progress-interval-ms must be positive")
    if args.termination_grace_sec < 0:
        parser.error("--termination-grace-sec must be non-negative")
    interval_mode = args.i_start is not None or args.i_end is not None
    if interval_mode and not (args.i_start is not None and args.i_end is not None):
        parser.error("instruction-interval mode requires both --i-start and --i-end")
    if interval_mode and not args.sample_instructions:
        parser.error("instruction-interval mode requires --sample-instructions")
    if interval_mode and args.i_start < 0:
        parser.error("--i-start must be non-negative")
    if interval_mode and args.i_end < 0:
        parser.error("--i-end must be non-negative")
    if interval_mode and args.i_end < args.i_start:
        parser.error("--i-end must be >= --i-start")
    if interval_mode and args.loop_until is not None and args.loop_until > 0:
        parser.error("instruction-interval mode cannot be combined with a positive --loop-until")
    if args.loop_until is not None and args.loop_until <= 0:
        args.loop_until = None

    args.output_target = str(Path(args.output_target).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    args.run_dir = str(Path(args.run_dir).resolve())
    args.benchmark = str(Path(args.benchmark).resolve())
    return args


def main() -> int:
    args = parse_args()
    benchmarks_root = find_benchmarks_root()
    run = build_single_run(args, benchmarks_root)
    controller = SingleController(args, benchmarks_root, run)
    controller.install_signal_handlers()
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
