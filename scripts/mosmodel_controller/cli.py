from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_benchmarks_root() -> Path:
    root = os.environ.get("BENCHMARKS_ROOT", sys.path[0])
    path = Path(root).resolve()
    if not path.exists():
        raise SystemExit(
            "Error: benchmarks root was not found.\n"
            f"Resolved path: {path}\n"
            "Search order:\n"
            "  1) BENCHMARKS_ROOT\n"
            f"  2) directory containing this script: {sys.path[0]}"
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one SMT benchmark pair with the legacy semantics: launch side2, "
            "launch side1, wait for side1, terminate side2, finalize outputs. "
            "When instruction sampling is enabled, attach a detached perf stat "
            "sampler to a barrier-held benchmark leader before releasing real work."
        )
    )

    parser.add_argument("--benchmark1", required=True, help="Path to benchmark1 directory")
    parser.add_argument("--benchmark2", required=True, help="Path to benchmark2 directory")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Base run dir for the pair; sides use <run-dir>/1 and <run-dir>/2",
    )
    parser.add_argument(
        "--side1-output-dir",
        required=True,
        help="Concrete output dir for side1",
    )
    parser.add_argument(
        "--side2-output-dir",
        required=True,
        help="Concrete output dir for side2",
    )
    parser.add_argument(
        "--output-target",
        required=True,
        help="Artifact that Make expects, usually .../perf.out",
    )

    parser.add_argument("-n", "--num-threads", type=int, default=1)
    parser.add_argument("--prefix1", default="", help="Command prefix for side1")
    parser.add_argument("--prefix2", default="", help="Command prefix for side2")
    parser.add_argument("--submit1", default="", help="Submit command for side1")
    parser.add_argument("--submit2", default="", help="Submit command for side2")
    parser.add_argument(
        "--loop-until1",
        type=int,
        default=None,
        help="If > 0, side1 runs under timeout <sec> loopForever.sh",
    )
    parser.add_argument(
        "--loop-until2",
        type=int,
        default=None,
        help="If > 0, side2 runs under timeout <sec> loopForever.sh",
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
        "--keep-side2-output",
        action="store_true",
        default=False,
        help="Keep side2 output dir instead of removing it at the end",
    )
    parser.add_argument(
        "--sample-instructions",
        action="store_true",
        default=False,
        help="Wrap each side under outer perf stat and print cumulative sampled instructions",
    )
    parser.add_argument(
        "--progress-perf-binary",
        default="perf",
        help="perf binary used for sampled instructions",
    )
    parser.add_argument(
        "--progress-interval-ms",
        type=int,
        default=50,
        help="perf stat interval in milliseconds",
    )
    parser.add_argument(
        "--looped-fallback-until",
        type=int,
        default=1000000000,
        help=(
            "Timeout in seconds for a missing-interval SMT side when the other "
            "side has a real instruction interval. The controller normally kills "
            "this side when the valid side reaches I_end."
        ),
    )



    parser.add_argument(
        "--i-start-side1",
        type=int,
        default=None,
        help="Sampled cumulative side1 instructions threshold marking the start of side1 interval",
    )
    parser.add_argument(
        "--i-end-side1",
        type=int,
        default=None,
        help="Sampled cumulative side1 instructions threshold marking the end of side1 interval",
    )
    parser.add_argument(
        "--i-start-side2",
        type=int,
        default=None,
        help="Sampled cumulative side2 instructions threshold marking the start of side2 interval",
    )
    parser.add_argument(
        "--i-end-side2",
        type=int,
        default=None,
        help="Sampled cumulative side2 instructions threshold marking the end of side2 interval",
    )
    parser.add_argument(
        "--sync-interval-windows",
        action="store_true",
        default=False,
        help=(
            "Synchronize the two interval windows: stop the first side that reaches I_start, "
            "resume both when both starts have been observed, and terminate both when the first "
            "side reaches I_end after synchronization begins"
        ),
    )

    parser.add_argument(
        "--external-resume-gate-dir",
        default="",
        help=(
            "Optional directory used as an external resume gate for synchronized interval mode. "
            "When set, the controller stops both benchmark groups once both start thresholds are observed, "
            "writes READY/STATE files in that directory, and waits until RESUME appears before continuing."
        ),
    )
    parser.add_argument(
        "--external-resume-socket-path",
        default=os.environ.get("MOSMODEL_CONTROLLER_EXTERNAL_RESUME_SOCKET_PATH", ""),
        help=(
            "Optional Unix domain socket path for the external resume gate. When set, the controller "
            "connects to the scheduler, sends a READY payload, and waits for RESUME on the same socket."
        ),
    )
    parser.add_argument(
        "--external-resume-token",
        default=os.environ.get("MOSMODEL_CONTROLLER_EXTERNAL_RESUME_TOKEN", ""),
        help="Opaque token identifying this run to the external scheduler when using the socket gate.",
    )

    parser.add_argument(
        "--debug-sync-ps",
        action="store_true",
        default=False,
        help=(
            "Print `ps -o pid,ppid,pgid,sid,stat,cmd -s <sid>` snapshots around sync STOP/CONT "
            "operations to debug which processes are being paused/resumed"
        ),
    )

    args = parser.parse_args()

    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if args.progress_interval_ms <= 0:
        parser.error("--progress-interval-ms must be positive")
    if args.looped_fallback_until <= 0:
        parser.error("--looped-fallback-until must be positive")
    for name in ("i_start_side1", "i_start_side2"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    interval_values = [
        args.i_start_side1,
        args.i_end_side1,
        args.i_start_side2,
        args.i_end_side2,
    ]
    interval_mode = any(value is not None for value in interval_values)
    if interval_mode and not all(value is not None for value in interval_values):
        parser.error(
            "interval boundary mode requires --i-start-side1, --i-end-side1, "
            "--i-start-side2, and --i-end-side2"
        )
    if interval_mode and not args.sample_instructions:
        parser.error("interval boundary mode requires --sample-instructions")
    if interval_mode:
        # A negative I_end means: this benchmark is missing an instruction
        # interval and should be handled as a wall-time/loopForever fallback.
        # The corresponding I_start must be 0 because there is no instruction
        # threshold to fast-forward to.
        if args.i_end_side1 >= 0 and args.i_end_side1 < args.i_start_side1:
            parser.error("--i-end-side1 must be >= --i-start-side1, or negative for wall-time fallback")
        if args.i_end_side2 >= 0 and args.i_end_side2 < args.i_start_side2:
            parser.error("--i-end-side2 must be >= --i-start-side2, or negative for wall-time fallback")
        if args.i_end_side1 < 0 and args.i_start_side1 != 0:
            parser.error("--i-start-side1 must be 0 when --i-end-side1 is negative")
        if args.i_end_side2 < 0 and args.i_start_side2 != 0:
            parser.error("--i-start-side2 must be 0 when --i-end-side2 is negative")
    if args.sync_interval_windows and not interval_mode:
        parser.error("--sync-interval-windows requires the four --i-start/--i-end flags")
    if interval_mode and ((args.loop_until1 is not None and args.loop_until1 > 0) or (args.loop_until2 is not None and args.loop_until2 > 0)):
        parser.error("interval boundary mode cannot be combined with loop-until mode")

    args.output_target = str(Path(args.output_target).resolve())
    args.side1_output_dir = str(Path(args.side1_output_dir).resolve())
    args.side2_output_dir = str(Path(args.side2_output_dir).resolve())
    args.run_dir = str(Path(args.run_dir).resolve())
    if args.external_resume_gate_dir:
        args.external_resume_gate_dir = str(Path(args.external_resume_gate_dir).resolve())
    if args.external_resume_socket_path:
        args.external_resume_socket_path = str(Path(args.external_resume_socket_path))
    if args.external_resume_token:
        args.external_resume_token = str(args.external_resume_token)
    return args
