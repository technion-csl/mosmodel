from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..benchmarkCore import BenchmarkRun
from .criu_restore import restore_stopped
from .finalize import ensure_parent, finalize_run
from .launcher import (
    LaunchedSide,
    benchmark_group_pid_details,
    compose_submit_command,
    launch_run_with_start_barrier,
    release_benchmark,
    signal_benchmark_group,
    terminate_and_wait,
)
from .progress import (
    DetachedMeasurementPerfSession,
    LiveProgressConfig,
    MeasurementPerfConfig,
    WrappedPerfInstructionsMonitor,
)


@dataclass
class SingleIntervalState:
    label: str
    start_target: int
    end_target: int
    start_seen: bool = False
    end_seen: bool = False
    start_observed_instructions: Optional[int] = None
    end_observed_instructions: Optional[int] = None
    start_observed_monotonic_sec: Optional[float] = None
    end_observed_monotonic_sec: Optional[float] = None


@dataclass
class SingleRun:
    run: BenchmarkRun
    cmd: str


class SingleController:
    def __init__(self, args: argparse.Namespace, benchmarks_root: Path, run: SingleRun):
        self.args = args
        self.benchmarks_root = benchmarks_root
        self.run_spec = run
        self.criu_mode = bool(self.args.checkpoint_dir)
        self.proc: Optional[LaunchedSide] = None
        self.monitor = WrappedPerfInstructionsMonitor(
            LiveProgressConfig(
                perf_binary=self.args.progress_perf_binary,
                interval_ms=self.args.progress_interval_ms,
                label="side1",
            )
        )
        self.measurement = DetachedMeasurementPerfSession(
            MeasurementPerfConfig(
                perf_binary=self.args.progress_perf_binary,
                interval_ms=1000,
                label="side1",
                output_path=Path(self.args.output_target),
            )
        )
        self.measurement_attached = False
        self.measurement_enabled = False
        self.interval_measurement_started = False
        self.interval_mode = self.args.i_start is not None and self.args.i_end is not None
        if self.interval_mode and self.criu_mode:
            self.interval_state = SingleIntervalState(
                label="side1",
                start_target=0,
                end_target=self.args.i_end - self.args.i_start,
                start_seen=True,
                start_observed_instructions=0,
                start_observed_monotonic_sec=time.monotonic(),
            )
        elif self.interval_mode:
            self.interval_state = SingleIntervalState(
                label="side1",
                start_target=self.args.i_start,
                end_target=self.args.i_end,
            )
        else:
            self.interval_state = None
        self._cleanup_started = False
        debug_root = Path(self.args.output_dir) if self.criu_mode else Path(self.args.run_dir)
        self.debug_log_path = debug_root / "controller_debug.log"
        ensure_parent(self.debug_log_path)
        try:
            self.debug_log_path.unlink()
        except FileNotFoundError:
            pass
        self._debug_log("single controller initialized")

    def _debug_log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"[{ts}] {message}"
        print(line)
        try:
            with self.debug_log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass

    def install_signal_handlers(self) -> None:
        def cleanup_on_signal(signum, frame) -> None:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                sig_name = signal.Signals(signum).name
            except ValueError:
                sig_name = str(signum)
            self._debug_log(f"received {sig_name}, terminating single side")
            self._disable_measurement_if_needed()
            terminate_and_wait(self.proc)
            self.join_monitors()
            self.print_interval_boundaries()
            self.print_sampled_instructions()
            self.print_measurement_perf_notes()
            self.write_interval_boundaries_summary()
            os._exit(128 + int(signum))

        signal.signal(signal.SIGINT, cleanup_on_signal)
        signal.signal(signal.SIGTERM, cleanup_on_signal)

    def prerun(self) -> None:
        if self.criu_mode:
            return
        print("running side1 prerun")
        self.run_spec.run.prerun()

    def _require_benchmark_pid(self, launched: Optional[LaunchedSide], label: str) -> int:
        if launched is None or launched.benchmark_pid is None:
            raise RuntimeError(f"missing benchmark pid for {label}")
        return launched.benchmark_pid

    def launch(self) -> None:
        if self.criu_mode:
            self._launch_criu()
            return

        try:
            self.proc = launch_run_with_start_barrier(
                self.run_spec.run,
                self.args.num_threads,
                self.run_spec.cmd,
            )
            side1_pid = self._require_benchmark_pid(self.proc, "side1")

            if self.args.sample_instructions:
                self.monitor.attach_to_pid(side1_pid)
                self.monitor.enable()

            # In instruction-interval mode, measurement perf is attached only
            # after I_start.
            release_benchmark(self.proc)
            self._debug_log(
                "launch sample path "
                f"side1_pid={side1_pid} "
                f"side1_benchmark_pgid={getattr(self.proc, 'benchmark_pgid', None)}"
            )
        except Exception:
            terminate_and_wait(self.proc)
            self.join_monitors()
            raise

    def _launch_criu(self) -> None:
        try:
            restored = restore_stopped(
                checkpoint_dir=Path(self.args.checkpoint_dir),
                checkpoint_archive_dir=Path(self.args.checkpoint_archive_dir),
                output_dir=Path(self.args.output_dir),
                prefix=self.args.prefix,
            )
            self.proc = restored.as_launched_side()

            if self.args.sample_instructions:
                self.monitor.attach_to_pid(restored.benchmark_pid)
            self.measurement.attach_to_pid(restored.benchmark_pid)
            self.measurement_attached = True

            if self.args.sample_instructions:
                self.monitor.enable()
            self._enable_measurement_if_needed()
            self.interval_measurement_started = True

            if not signal_benchmark_group(self.proc, signal.SIGCONT):
                raise RuntimeError("failed to resume restored benchmark process group")
            self._debug_log(
                "CRIU interval resumed after restore placement and perf enable: "
                f"target_pid={restored.benchmark_pid} "
                f"target_instructions={self.args.i_end - self.args.i_start}"
            )
        except Exception:
            terminate_and_wait(self.proc)
            self.join_monitors()
            raise

    def _attach_measurement_if_needed(self) -> None:
        if self.measurement_attached:
            return

        details = benchmark_group_pid_details(self.proc)
        target_pids = details["final_target_pids"]
        benchmark_pid = getattr(self.proc, "benchmark_pid", None)
        benchmark_pgid = getattr(self.proc, "benchmark_pgid", None)

        print(
            f"[measurement perf] ST discovery side1: "
            f"benchmark_pgid={benchmark_pgid} benchmark_pid={benchmark_pid} "
            f"pgid_members={details['pgid_members']} descendants={details['descendants']} "
            f"final_target_pids={target_pids}"
        )
        target_rows = details.get("target_process_rows", [])
        if target_rows:
            print("[measurement perf] ST target process rows for side1:")
            for row in target_rows:
                print(
                    "  pid={pid} ppid={ppid} pgid={pgid} sid={sid} stat={stat} cmd={cmd}".format(**row)
                )
        else:
            print("[measurement perf] ST target process rows for side1: <none>")

        if not target_pids:
            raise RuntimeError(
                "no side1 benchmark-group pids available for ST measurement perf "
                f"(benchmark_pgid={benchmark_pgid}, benchmark_pid={benchmark_pid})"
            )

        self.measurement.attach_to_pids(target_pids)
        self.measurement_attached = True

    def _start_interval_measurement_if_needed(self) -> None:
        if self.interval_measurement_started:
            return

        # Match the SMT interval path: when I_start is reached, freeze the
        # benchmark group, discover the real benchmark/runMosalloc subtree,
        # attach and enable measurement perf, then resume the benchmark.
        stopped = signal_benchmark_group(self.proc, signal.SIGSTOP)
        if stopped:
            self._debug_log("[interval boundary] side1 STOPped at I_start before measurement attach")
            time.sleep(0.02)
        else:
            self._debug_log("[interval boundary] WARNING: failed to STOP side1 at I_start")

        self._attach_measurement_if_needed()
        self._enable_measurement_if_needed()

        if stopped:
            if signal_benchmark_group(self.proc, signal.SIGCONT):
                self._debug_log("[interval boundary] side1 CONT after measurement enable")
            else:
                self._debug_log("[interval boundary] WARNING: failed to CONT side1 after measurement enable")

        self.interval_measurement_started = True

    def _enable_measurement_if_needed(self) -> None:
        if self.measurement_enabled:
            return
        if not self.measurement_attached:
            return
        self.measurement.enable()
        self.measurement_enabled = True

    def _disable_measurement_if_needed(self) -> None:
        if not self.measurement_attached or not self.measurement_enabled:
            return
        self.measurement.disable()
        self.measurement_enabled = False

    def join_monitors(self) -> None:
        if self.args.sample_instructions:
            self.monitor.stop(timeout=5.0)
        self.measurement.stop(timeout=5.0)

    def _update_interval_state(self, current_instructions: int, now_monotonic: float) -> None:
        state = self.interval_state
        if state is None:
            return
        if (not state.start_seen) and current_instructions >= state.start_target:
            state.start_seen = True
            state.start_observed_instructions = current_instructions
            state.start_observed_monotonic_sec = now_monotonic
            self._debug_log(
                f"[interval boundary] side1 reached I_start at "
                f"instructions={current_instructions} threshold={state.start_target}"
            )
            self._start_interval_measurement_if_needed()
        if state.start_seen and (not state.end_seen) and current_instructions >= state.end_target:
            state.end_seen = True
            state.end_observed_instructions = current_instructions
            state.end_observed_monotonic_sec = now_monotonic
            self._debug_log(
                f"[interval boundary] side1 reached I_end at "
                f"instructions={current_instructions} threshold={state.end_target}"
            )
            self._disable_measurement_if_needed()

    def _interval_completed(self) -> bool:
        return bool(self.interval_state is not None and self.interval_state.end_seen)

    def wait_with_sampled_instruction_control(self) -> tuple[Optional[int], bool]:
        if self.proc is None:
            return 1, False

        while True:
            if self.criu_mode and self.proc.benchmark_pid is not None:
                rc = None if Path(f"/proc/{self.proc.benchmark_pid}").exists() else 0
            else:
                rc = self.proc.proc.poll()
            current_instructions = self.monitor.total_instructions()

            if self.interval_mode:
                self._update_interval_state(current_instructions, time.monotonic())
                if self._interval_completed():
                    return rc, True
                if rc is not None:
                    return rc, False
            else:
                if rc is not None:
                    return rc, False

            time.sleep(0.05)

    def print_sampled_instructions(self) -> None:
        if not self.args.sample_instructions:
            return
        total = self.monitor.total_instructions()
        print("[sampled instructions]")
        print(f"  side1={total}")
        err = self.monitor.last_error_line()
        if err:
            print(f"  side1_perf_note={err}", file=sys.stderr)

    def print_measurement_perf_notes(self) -> None:
        err = self.measurement.last_error_line()
        if err:
            print(f"  side1_measurement_perf_note={err}", file=sys.stderr)

    def print_interval_boundaries(self) -> None:
        if not self.interval_mode:
            return
        state = self.interval_state
        print("[interval boundaries]")
        if state is not None:
            print(
                f"  side1: "
                f"start_target={state.start_target}, "
                f"start_observed={state.start_observed_instructions}, "
                f"end_target={state.end_target}, "
                f"end_observed={state.end_observed_instructions}"
            )

    def write_interval_boundaries_summary(self) -> None:
        if not self.interval_mode:
            return
        summary_path = Path(self.args.output_dir) / "interval_boundaries.json"
        ensure_parent(summary_path)
        payload = {
            "interval_completed": self._interval_completed(),
            "side1": asdict(self.interval_state) if self.interval_state is not None else None,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def touch_output_target(self) -> None:
        output_target = Path(self.args.output_target)
        ensure_parent(output_target)
        output_target.touch(exist_ok=True)

    def run(self) -> int:
        self.prerun()
        self.launch()

        interval_completed = False
        rc: Optional[int]

        if self.args.sample_instructions:
            rc, interval_completed = self.wait_with_sampled_instruction_control()
            if interval_completed:
                self._disable_measurement_if_needed()
                term_rc = terminate_and_wait(self.proc)
                if rc is None:
                    rc = term_rc
            else:
                if rc is None and self.proc is not None:
                    rc = self.proc.proc.wait()
        else:
            rc = self.proc.proc.wait() if self.proc is not None else 1

        self._disable_measurement_if_needed()
        self.join_monitors()
        print(f"[run rc] side1={rc}")
        self.print_interval_boundaries()
        self.print_sampled_instructions()
        self.print_measurement_perf_notes()
        self.write_interval_boundaries_summary()

        loop_mode = self.args.loop_until is not None and self.args.loop_until > 0
        if interval_completed:
            success = True
        else:
            success = (rc == 0) if not loop_mode else (rc in (0, 124))

        if success:
            if interval_completed:
                # The benchmark was intentionally terminated at I_end. Do not move
                # files from the restored checkpoint copy into the measurement output.
                if not self.criu_mode:
                    self.run_spec.run.move_files_to_output_dir()
                    self.run_spec.run.clean_output_dir(
                        self.args.clean_threshold,
                        self.args.exclude_files,
                    )
            else:
                finalize_run(
                    self.run_spec.run,
                    do_postrun=(not loop_mode),
                    clean_threshold=self.args.clean_threshold,
                    exclude_files=self.args.exclude_files,
                )
            self.touch_output_target()
        elif not self.criu_mode:
            self.run_spec.run.move_files_to_output_dir()

        return 0 if success else (rc if rc is not None else 1)


def build_single_run(args: argparse.Namespace, benchmarks_root: Path) -> SingleRun:
    run_dir = Path(args.run_dir) / "work" if args.checkpoint_dir else Path(args.run_dir)
    run = BenchmarkRun(args.benchmark, str(run_dir), str(Path(args.output_dir)))
    cmd = compose_submit_command(
        args.prefix,
        args.submit,
        args.loop_until,
        benchmarks_root,
    )
    return SingleRun(run=run, cmd=cmd)
