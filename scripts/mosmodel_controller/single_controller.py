from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..benchmarkCore import BenchmarkRun
from .finalize import ensure_parent, finalize_run
from .launcher import compose_submit_command, terminate_and_wait
from .progress import DetachedMeasurementPerfSession, MeasurementPerfConfig
from .side import ControllerSide, SideConfig


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
    def __init__(self, args: argparse.Namespace, run: SingleRun):
        self.args = args
        self.run_spec = run
        self.side = ControllerSide(
            SideConfig(
                label="side1",
                run=run.run,
                cmd=run.cmd,
                prefix=args.prefix,
                output_dir=Path(args.output_dir),
                i_start=args.i_start,
                checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
                checkpoint_archive_dir=(
                    Path(args.checkpoint_archive_dir)
                    if args.checkpoint_archive_dir
                    else None
                ),
            ),
            perf_binary=args.progress_perf_binary,
            progress_interval_ms=args.progress_interval_ms,
        )
        self.measurement = DetachedMeasurementPerfSession(
            MeasurementPerfConfig(
                perf_binary=args.progress_perf_binary,
                interval_ms=1000,
                label="side1",
                output_path=Path(args.output_target),
            )
        )
        self.measurement_attached = False
        self.measurement_enabled = False
        self.interval_measurement_started = False
        self.interval_mode = args.i_start is not None and args.i_end is not None
        self.interval_state = self._make_interval_state() if self.interval_mode else None
        self._cleanup_started = False
        self._log("single controller initialized")

    @property
    def criu_mode(self) -> bool:
        return self.side.restored

    def _make_interval_state(self) -> SingleIntervalState:
        assert self.args.i_start is not None and self.args.i_end is not None
        if self.criu_mode:
            return SingleIntervalState(
                label="side1",
                start_target=0,
                end_target=self.args.i_end - self.args.i_start,
                start_seen=True,
                start_observed_instructions=0,
                start_observed_monotonic_sec=time.monotonic(),
            )
        return SingleIntervalState(
            label="side1",
            start_target=self.args.i_start,
            end_target=self.args.i_end,
        )

    def _log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"[{ts}] {message}")

    def install_signal_handlers(self) -> None:
        def cleanup_on_signal(signum, frame) -> None:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            self._log(f"received {signal.Signals(signum).name}, terminating single side")
            try:
                self._disable_measurement_if_needed()
            finally:
                self._terminate_workload()
                self.join_monitors()
                self.side.remove_cgroup(self._log)
            self.print_interval_boundaries()
            self.print_sampled_instructions()
            self.print_measurement_perf_notes()
            self.write_interval_boundaries_summary()
            os._exit(128 + int(signum))

        signal.signal(signal.SIGINT, cleanup_on_signal)
        signal.signal(signal.SIGTERM, cleanup_on_signal)

    def prerun(self) -> None:
        if not self.criu_mode:
            print("running side1 prerun")
        self.side.prerun()

    def launch(self) -> None:
        try:
            self.side.prepare(
                num_threads=self.args.num_threads,
                sample_instructions=self.args.sample_instructions,
                log=self._log,
            )
            if self.criu_mode:
                self.side.attach_measurement(self.measurement)
                self.measurement_attached = True
                self._enable_measurement_if_needed()
                self.interval_measurement_started = True
            self.side.resume()
            if self.criu_mode:
                self._log(
                    "CRIU interval resumed after cgroup placement and perf enable: "
                    f"cgroup={self.side.cgroup.perf_name if self.side.cgroup else None} "
                    f"target_instructions={self.args.i_end - self.args.i_start}"
                )
        except Exception:
            self._terminate_workload()
            self.join_monitors()
            self.side.remove_cgroup(self._log)
            raise

    def _freeze_criu_cgroup_if_needed(self) -> None:
        if not self.criu_mode or self.side.cgroup is None or self.side.cgroup_frozen:
            return
        self.side.cgroup.freeze()
        self.side.cgroup_frozen = True
        self._log(
            f"[interval boundary] side1 cgroup frozen at I_end: "
            f"{self.side.cgroup.perf_name}"
        )

    def _terminate_workload(self) -> Optional[int]:
        try:
            self.side.kill_cgroup()
        except Exception as exc:
            self._log(f"WARNING: failed to kill side1 CRIU cgroup: {exc}")
        return terminate_and_wait(
            self.side.proc,
            grace_sec=0.0 if self.criu_mode else 2.0,
        )

    def _attach_measurement_if_needed(self) -> None:
        if self.measurement_attached:
            return
        self.side.attach_measurement(self.measurement)
        self.measurement_attached = True

    def _start_interval_measurement_if_needed(self) -> None:
        if self.interval_measurement_started:
            return
        stopped = self.side.signal_benchmark(signal.SIGSTOP)
        if stopped:
            time.sleep(0.02)
        self._attach_measurement_if_needed()
        self._enable_measurement_if_needed()
        if stopped and not self.side.signal_benchmark(signal.SIGCONT):
            raise RuntimeError("failed to resume side1 after measurement attach")
        self.interval_measurement_started = True

    def _enable_measurement_if_needed(self) -> None:
        if self.measurement_attached and not self.measurement_enabled:
            self.measurement.enable()
            self.measurement_enabled = True

    def _disable_measurement_if_needed(self) -> None:
        if self.measurement_attached and self.measurement_enabled:
            self.measurement.disable()
            self.measurement_enabled = False

    def join_monitors(self) -> None:
        if self.args.sample_instructions:
            self.side.monitor.stop()
        self.measurement.stop()

    def _update_interval_state(self, current: int, now: float) -> None:
        state = self.interval_state
        if state is None:
            return
        if not state.start_seen and current >= state.start_target:
            state.start_seen = True
            state.start_observed_instructions = current
            state.start_observed_monotonic_sec = now
            self._log(
                f"[interval boundary] side1 reached I_start at "
                f"instructions={current} threshold={state.start_target}"
            )
            self._start_interval_measurement_if_needed()
        if state.start_seen and not state.end_seen and current >= state.end_target:
            state.end_seen = True
            state.end_observed_instructions = current
            state.end_observed_monotonic_sec = now
            self._log(
                f"[interval boundary] side1 reached I_end at "
                f"instructions={current} threshold={state.end_target}"
            )
            self._freeze_criu_cgroup_if_needed()
            self._disable_measurement_if_needed()

    def _interval_completed(self) -> bool:
        return bool(self.interval_state and self.interval_state.end_seen)

    def wait_with_sampled_instruction_control(self) -> tuple[Optional[int], bool]:
        if self.side.proc is None:
            return 1, False
        while True:
            if self.criu_mode and self.side.cgroup is not None:
                rc = None if self.side.cgroup.is_populated() else 0
            else:
                rc = self.side.proc.proc.poll()
            current = self.side.instruction_count()
            if self.interval_mode:
                self._update_interval_state(current, time.monotonic())
                if self._interval_completed():
                    return rc, True
                if rc is not None:
                    return rc, False
            elif rc is not None:
                return rc, False
            time.sleep(0.05)

    def print_sampled_instructions(self) -> None:
        if not self.args.sample_instructions:
            return
        print("[sampled instructions]")
        print(f"  side1={self.side.instruction_count()}")
        err = self.side.monitor.last_error_line()
        if err:
            print(f"  side1_perf_note={err}", file=sys.stderr)

    def print_measurement_perf_notes(self) -> None:
        err = self.measurement.last_error_line()
        if err:
            print(f"  side1_measurement_perf_note={err}", file=sys.stderr)

    def print_interval_boundaries(self) -> None:
        if self.interval_state is None:
            return
        state = self.interval_state
        print("[interval boundaries]")
        print(
            f"  side1: start_target={state.start_target}, "
            f"start_observed={state.start_observed_instructions}, "
            f"end_target={state.end_target}, "
            f"end_observed={state.end_observed_instructions}"
        )

    def write_interval_boundaries_summary(self) -> None:
        if self.interval_state is None:
            return
        summary_path = Path(self.args.output_dir) / "interval_boundaries.json"
        ensure_parent(summary_path)
        summary_path.write_text(
            json.dumps(
                {
                    "interval_completed": self._interval_completed(),
                    "side1": asdict(self.interval_state),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def touch_output_target(self) -> None:
        output_target = Path(self.args.output_target)
        ensure_parent(output_target)
        output_target.touch(exist_ok=True)

    def run(self) -> int:
        self.prerun()
        self.launch()

        interval_completed = False
        if self.args.sample_instructions:
            rc, interval_completed = self.wait_with_sampled_instruction_control()
            if interval_completed or self.criu_mode:
                term_rc = self._terminate_workload()
                if rc is None:
                    rc = term_rc
            elif rc is None and self.side.proc is not None:
                rc = self.side.proc.proc.wait()
        else:
            rc = self.side.proc.proc.wait() if self.side.proc is not None else 1

        self._disable_measurement_if_needed()
        self.join_monitors()
        self.side.remove_cgroup(self._log)
        print(f"[run rc] side1={rc}")
        self.print_interval_boundaries()
        self.print_sampled_instructions()
        self.print_measurement_perf_notes()
        self.write_interval_boundaries_summary()

        loop_mode = self.args.loop_until is not None and self.args.loop_until > 0
        success = interval_completed or ((rc == 0) if not loop_mode else (rc in (0, 124)))

        if success:
            if interval_completed:
                if not self.criu_mode:
                    self.run_spec.run.move_files_to_output_dir()
                    self.run_spec.run.clean_output_dir(
                        self.args.clean_threshold,
                        self.args.exclude_files,
                    )
            else:
                finalize_run(
                    self.run_spec.run,
                    do_postrun=not loop_mode,
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
