from __future__ import annotations

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
from .finalize import ensure_parent, finalize_run, remove_dir_if_exists
from .launcher import compose_submit_command, terminate_many_and_wait
from .progress import DetachedMeasurementPerfSession, MeasurementPerfConfig
from .side import ControllerSide, SideConfig


@dataclass
class IntervalBoundaryState:
    label: str
    start_target: int
    end_target: int
    start_seen: bool = False
    end_seen: bool = False
    start_observed_instructions: Optional[int] = None
    end_observed_instructions: Optional[int] = None
    start_observed_monotonic_sec: Optional[float] = None
    end_observed_monotonic_sec: Optional[float] = None
    pre_sync_end_crossed: bool = False
    pre_sync_end_observed_instructions: Optional[int] = None
    pre_sync_end_observed_monotonic_sec: Optional[float] = None


@dataclass
class IntervalControlResult:
    rc1: Optional[int]
    rc2: Optional[int]
    interval_completed: bool


@dataclass
class PairRuns:
    run1: BenchmarkRun
    run2: BenchmarkRun
    cmd1: str
    cmd2: str


class PairController:
    def __init__(self, args, runs: PairRuns):
        self.args = args
        self.runs = runs
        self.criu_run = bool(args.criu_run)

        self.side1 = self._build_side(
            label="side1",
            run=runs.run1,
            cmd=runs.cmd1,
            prefix=args.prefix1,
            output_dir=Path(args.side1_output_dir),
            i_start=args.i_start_side1,
            checkpoint_dir=args.checkpoint_dir1,
            checkpoint_archive_dir=args.checkpoint_archive_dir1,
        )
        self.side2 = self._build_side(
            label="side2",
            run=runs.run2,
            cmd=runs.cmd2,
            prefix=args.prefix2,
            output_dir=Path(args.side2_output_dir),
            i_start=args.i_start_side2,
            checkpoint_dir=args.checkpoint_dir2,
            checkpoint_archive_dir=args.checkpoint_archive_dir2,
        )
        self.sides = (self.side1, self.side2)

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
        self._cleanup_started = False

        self.interval_mode = all(
            value is not None
            for value in (
                args.i_start_side1,
                args.i_end_side1,
                args.i_start_side2,
                args.i_end_side2,
            )
        )
        self.sync_interval_mode = bool(self.interval_mode and args.sync_interval_windows)
        self.interval1 = self._make_interval_state(self.side1, args.i_start_side1, args.i_end_side1)
        self.interval2 = self._make_interval_state(self.side2, args.i_start_side2, args.i_end_side2)

        self.sync_started = False
        self.sync_completed = False
        self.sync_paused_side: Optional[str] = None
        self.sync_completed_reason: Optional[str] = None
        self.sync_started_monotonic_sec: Optional[float] = None
        self.sync_started_side1_instructions: Optional[int] = None
        self.sync_started_side2_instructions: Optional[int] = None
        self._log("pair controller initialized")

    def _build_side(
        self,
        *,
        label: str,
        run: BenchmarkRun,
        cmd: str,
        prefix: str,
        output_dir: Path,
        i_start: Optional[int],
        checkpoint_dir: Optional[str],
        checkpoint_archive_dir: Optional[str],
    ) -> ControllerSide:
        return ControllerSide(
            SideConfig(
                label=label,
                run=run,
                cmd=cmd,
                prefix=prefix,
                output_dir=output_dir,
                i_start=i_start,
                checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
                checkpoint_archive_dir=(
                    Path(checkpoint_archive_dir) if checkpoint_archive_dir else None
                ),
            ),
            perf_binary=self.args.progress_perf_binary,
            progress_interval_ms=self.args.progress_interval_ms,
        )

    def _make_interval_state(
        self,
        side: ControllerSide,
        start: Optional[int],
        end: Optional[int],
    ) -> Optional[IntervalBoundaryState]:
        if not self.interval_mode:
            return None
        assert start is not None and end is not None
        if side.restored:
            return IntervalBoundaryState(
                label=side.label,
                start_target=0,
                end_target=end - start,
                start_seen=True,
                start_observed_instructions=0,
                start_observed_monotonic_sec=time.monotonic(),
            )
        return IntervalBoundaryState(side.label, start, end)

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
            self._log(f"received {signal.Signals(signum).name}, terminating both sides")
            try:
                self._disable_measurement_if_needed()
            finally:
                self.terminate_both()
                self.join_monitors()
                self._remove_criu_cgroups()
            self.print_interval_boundaries()
            self.print_sampled_instructions()
            self.print_measurement_perf_notes()
            self.write_interval_boundaries_summary()
            os._exit(128 + int(signum))

        signal.signal(signal.SIGINT, cleanup_on_signal)
        signal.signal(signal.SIGTERM, cleanup_on_signal)

    def prerun_both(self) -> None:
        for side in (self.side2, self.side1):
            if not side.restored:
                print(f"running {side.label} prerun")
            side.prerun()

    def _prepare(self, side: ControllerSide) -> None:
        side.prepare(
            num_threads=self.args.num_threads,
            sample_instructions=self.args.sample_instructions,
            log=self._log,
        )

    def _mark_zero_start_prepared(self, side: ControllerSide, state: IntervalBoundaryState) -> None:
        if state.start_target == 0 and not state.start_seen:
            state.start_seen = True
            state.start_observed_instructions = 0
            state.start_observed_monotonic_sec = time.monotonic()

    def _launch_criu_pair(self) -> None:
        if not self.sync_interval_mode:
            raise RuntimeError("SMT CRIU mode requires synchronized instruction intervals")
        assert self.interval1 is not None and self.interval2 is not None
        try:
            # I_start>0 sides are restored stopped at their saved interval start.
            # I_start=0 sides have no checkpoint and wait behind the native barrier.
            self._prepare(self.side2)
            self._prepare(self.side1)
            self._mark_zero_start_prepared(self.side1, self.interval1)
            self._mark_zero_start_prepared(self.side2, self.interval2)
            if not (self.interval1.start_seen and self.interval2.start_seen):
                raise RuntimeError("both sides must be prepared at I_start before CRIU resume")

            self._enable_measurement_if_needed()
            self.side2.resume()
            self.side1.resume()
            self.sync_started = True
            self.sync_started_monotonic_sec = time.monotonic()
            self.sync_started_side1_instructions = self.side1.instruction_count()
            self.sync_started_side2_instructions = self.side2.instruction_count()
            self._log(
                "[interval sync] SMT CRIU pair prepared at I_start; "
                "measurement enabled and both sides resumed"
            )
        except Exception:
            self.terminate_both()
            self.join_monitors()
            self._remove_criu_cgroups()
            raise

    def launch_both(self) -> None:
        if self.criu_run:
            self._launch_criu_pair()
            return

        try:
            # Preserve historical launch order: side2 starts first, then side1.
            self._prepare(self.side2)
            self.side2.resume()
            self._prepare(self.side1)

            if not self.sync_interval_mode:
                self._attach_measurement_if_needed()
                self._enable_measurement_if_needed()
            self.side1.resume()
        except Exception:
            self.terminate_both()
            self.join_monitors()
            self._remove_criu_cgroups()
            raise

    def terminate_both(self) -> tuple[Optional[int], Optional[int]]:
        for side in self.sides:
            try:
                side.kill_cgroup()
            except Exception as exc:
                self._log(f"WARNING: failed to kill {side.label} CRIU cgroup: {exc}")
        self._log("sending SIGTERM to both side sessions for cleanup")
        rc1, rc2 = terminate_many_and_wait((self.side1.proc, self.side2.proc))
        return rc1, rc2

    def _remove_criu_cgroups(self) -> None:
        for side in self.sides:
            side.remove_cgroup(self._log)

    def _attach_measurement_if_needed(self) -> None:
        if self.measurement_attached:
            return
        self.side1.attach_measurement(self.measurement)
        self.measurement_attached = True

    def _enable_measurement_if_needed(self) -> None:
        if self.measurement_enabled:
            return
        self._attach_measurement_if_needed()
        self.measurement.enable()
        self.measurement_enabled = True

    def _disable_measurement_if_needed(self) -> None:
        if self.measurement_attached and self.measurement_enabled:
            self.measurement.disable()
            self.measurement_enabled = False

    def join_monitors(self) -> None:
        if self.args.sample_instructions:
            for side in self.sides:
                side.monitor.stop()
        self.measurement.stop()

    def print_sampled_instructions(self) -> None:
        if not self.args.sample_instructions:
            return
        total1 = self.side1.instruction_count()
        total2 = self.side2.instruction_count()
        print("[sampled instructions]")
        print(f"  side1={total1}")
        print(f"  side2={total2}")
        print(f"  total={total1 + total2}")
        for side in self.sides:
            err = side.monitor.last_error_line()
            if err:
                print(f"  {side.label}_perf_note={err}", file=sys.stderr)

    def print_measurement_perf_notes(self) -> None:
        err = self.measurement.last_error_line()
        if err:
            print(f"  side1_measurement_perf_note={err}", file=sys.stderr)

    def _update_interval_state(
        self,
        state: IntervalBoundaryState,
        current: int,
        now: float,
        *,
        allow_end: bool,
    ) -> None:
        if not state.start_seen and current >= state.start_target:
            state.start_seen = True
            state.start_observed_instructions = current
            state.start_observed_monotonic_sec = now
            self._log(
                f"[interval boundary] {state.label} reached I_start at "
                f"instructions={current} threshold={state.start_target}"
            )

        if allow_end and not state.end_seen and current >= state.end_target:
            state.end_seen = True
            state.end_observed_instructions = current
            state.end_observed_monotonic_sec = now
            self._log(
                f"[interval boundary] {state.label} reached I_end at "
                f"instructions={current} threshold={state.end_target}"
            )
        elif (
            not allow_end
            and state.start_seen
            and not state.pre_sync_end_crossed
            and current >= state.end_target
        ):
            state.pre_sync_end_crossed = True
            state.pre_sync_end_observed_instructions = current
            state.pre_sync_end_observed_monotonic_sec = now
            self._log(
                f"[interval sync] WARNING: {state.label} crossed I_end before "
                f"synchronization at instructions={current} threshold={state.end_target}"
            )

    def _pause_first_side_to_reach_start(self) -> None:
        assert self.interval1 is not None and self.interval2 is not None
        if self.sync_paused_side is not None:
            return
        if self.interval1.start_seen and not self.interval2.start_seen:
            self.side1.signal_benchmark(signal.SIGSTOP)
            self.sync_paused_side = "side1"
            self._log("[interval sync] side1 reached I_start first; STOP side1")
        elif self.interval2.start_seen and not self.interval1.start_seen:
            self.side2.signal_benchmark(signal.SIGSTOP)
            self.sync_paused_side = "side2"
            self._log("[interval sync] side2 reached I_start first; STOP side2")

    def _choose_first_completed_side(self) -> str:
        assert self.interval1 is not None and self.interval2 is not None
        if self.interval1.end_seen and not self.interval2.end_seen:
            return "side1_end"
        if self.interval2.end_seen and not self.interval1.end_seen:
            return "side2_end"
        t1 = self.interval1.end_observed_monotonic_sec
        t2 = self.interval2.end_observed_monotonic_sec
        return "side1_end" if t2 is None or (t1 is not None and t1 <= t2) else "side2_end"

    def _sync_interval_control(self, current1: int, current2: int, now: float) -> bool:
        assert self.interval1 is not None and self.interval2 is not None

        if not self.sync_started:
            self._pause_first_side_to_reach_start()
            if self.interval1.start_seen and self.interval2.start_seen:
                self.side1.signal_benchmark(signal.SIGSTOP)
                self.side2.signal_benchmark(signal.SIGSTOP)
                current1 = self.side1.instruction_count()
                current2 = self.side2.instruction_count()
                now = time.monotonic()
                self._enable_measurement_if_needed()
                if not self.side1.signal_benchmark(signal.SIGCONT):
                    raise RuntimeError("failed to resume side1 at synchronized I_start")
                if not self.side2.signal_benchmark(signal.SIGCONT):
                    raise RuntimeError("failed to resume side2 at synchronized I_start")
                self.sync_started = True
                self.sync_started_monotonic_sec = now
                self.sync_started_side1_instructions = current1
                self.sync_started_side2_instructions = current2
                self._log(
                    "[interval sync] both sides reached I_start; measurement enabled "
                    "and both sides resumed"
                )
                self._update_interval_state(self.interval1, current1, now, allow_end=True)
                self._update_interval_state(self.interval2, current2, now, allow_end=True)

        if self.sync_started and not self.sync_completed:
            if self.interval1.end_seen or self.interval2.end_seen:
                self.sync_completed = True
                self.sync_completed_reason = self._choose_first_completed_side()
                self._disable_measurement_if_needed()
                self._log(
                    f"[interval sync] {self.sync_completed_reason} reached; "
                    "measurement disabled and both sides will terminate"
                )
                return True
        return False

    def wait_with_sampled_instruction_control(self) -> IntervalControlResult:
        if self.side1.proc is None or self.side2.proc is None:
            return IntervalControlResult(1, None, False)

        while True:
            rc1 = self.side1.proc.proc.poll()
            rc2 = self.side2.proc.proc.poll()
            current1 = self.side1.instruction_count()
            current2 = self.side2.instruction_count()

            if self.interval_mode:
                assert self.interval1 is not None and self.interval2 is not None
                now = time.monotonic()
                allow_end = not self.sync_interval_mode or self.sync_started
                self._update_interval_state(self.interval1, current1, now, allow_end=allow_end)
                self._update_interval_state(self.interval2, current2, now, allow_end=allow_end)

                if self.sync_interval_mode:
                    if self._sync_interval_control(current1, current2, now):
                        return IntervalControlResult(rc1, rc2, True)
                elif self.interval1.end_seen and self.interval2.end_seen:
                    self._disable_measurement_if_needed()
                    return IntervalControlResult(rc1, rc2, True)

                if rc1 is not None or rc2 is not None:
                    return IntervalControlResult(rc1, rc2, False)
            elif rc1 is not None and rc2 is not None:
                return IntervalControlResult(rc1, rc2, False)

            time.sleep(0.05)

    def print_interval_boundaries(self) -> None:
        if not self.interval_mode:
            return
        print("[interval boundaries]")
        for state in (self.interval1, self.interval2):
            assert state is not None
            print(
                f"  {state.label}: start_target={state.start_target}, "
                f"start_observed={state.start_observed_instructions}, "
                f"end_target={state.end_target}, "
                f"end_observed={state.end_observed_instructions}, "
                f"pre_sync_end_crossed={state.pre_sync_end_crossed}"
            )
        if self.sync_interval_mode:
            print("[interval sync summary]")
            print(f"  started={self.sync_started}")
            print(f"  paused_first={self.sync_paused_side}")
            print(f"  started_side1_instructions={self.sync_started_side1_instructions}")
            print(f"  started_side2_instructions={self.sync_started_side2_instructions}")
            print(f"  completed={self.sync_completed}")
            print(f"  completed_reason={self.sync_completed_reason}")

    def write_interval_boundaries_summary(self) -> None:
        if not self.interval_mode:
            return
        summary_path = Path(self.args.side1_output_dir) / "interval_boundaries.json"
        ensure_parent(summary_path)
        summary_path.write_text(
            json.dumps(
                {
                    "interval_completed": self.sync_completed
                    if self.sync_interval_mode
                    else bool(self.interval1 and self.interval2 and self.interval1.end_seen and self.interval2.end_seen),
                    "sync_interval_mode": self.sync_interval_mode,
                    "sync_started": self.sync_started,
                    "sync_paused_side": self.sync_paused_side,
                    "sync_started_monotonic_sec": self.sync_started_monotonic_sec,
                    "sync_started_side1_instructions": self.sync_started_side1_instructions,
                    "sync_started_side2_instructions": self.sync_started_side2_instructions,
                    "sync_completed": self.sync_completed,
                    "sync_completed_reason": self.sync_completed_reason,
                    "side1": asdict(self.interval1) if self.interval1 else None,
                    "side2": asdict(self.interval2) if self.interval2 else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _finalize_side1(self, *, do_postrun: bool) -> None:
        if self.side1.restored:
            return
        finalize_run(
            self.runs.run1,
            do_postrun=do_postrun,
            clean_threshold=self.args.clean_threshold,
            exclude_files=self.args.exclude_files,
        )

    def _finalize_side2(self, *, do_postrun: bool) -> None:
        if self.side2.restored:
            if not self.args.keep_side2_output:
                remove_dir_if_exists(Path(self.args.side2_output_dir))
            return
        if self.args.keep_side2_output:
            finalize_run(
                self.runs.run2,
                do_postrun=do_postrun,
                clean_threshold=self.args.clean_threshold,
                exclude_files=self.args.exclude_files,
            )
        else:
            remove_dir_if_exists(Path(self.args.side2_output_dir))

    def touch_output_target(self) -> None:
        output_target = Path(self.args.output_target)
        ensure_parent(output_target)
        output_target.touch(exist_ok=True)

    def run(self) -> int:
        self.prerun_both()
        self.launch_both()

        interval_completed = False
        if self.args.sample_instructions:
            result = self.wait_with_sampled_instruction_control()
            interval_completed = result.interval_completed
            rc1, rc2 = result.rc1, result.rc2
            if interval_completed:
                self._disable_measurement_if_needed()
                term1, term2 = self.terminate_both()
                rc1 = term1 if rc1 is None else rc1
                rc2 = term2 if rc2 is None else rc2
            else:
                if rc1 is None and self.side1.proc is not None:
                    rc1 = self.side1.proc.proc.wait()
                if rc2 is None and self.side2.proc is not None:
                    rc2 = self.side2.proc.proc.wait()
        else:
            rc1 = self.side1.proc.proc.wait() if self.side1.proc is not None else 1
            _term1, rc2 = self.terminate_both()

        self._disable_measurement_if_needed()
        self.join_monitors()
        self._remove_criu_cgroups()
        print(f"[run rc] side1={rc1} side2={rc2}")
        self.print_interval_boundaries()
        self.print_sampled_instructions()
        self.print_measurement_perf_notes()
        self.write_interval_boundaries_summary()

        loop1 = self.args.loop_until1 is not None and self.args.loop_until1 > 0
        loop2 = self.args.loop_until2 is not None and self.args.loop_until2 > 0
        if interval_completed:
            success1 = success2 = True
        else:
            success1 = (rc1 == 0) if not loop1 else (rc1 in (0, 124))
            success2 = (rc2 == 0) if not loop2 else (rc2 in (0, 124, None))

        if success1:
            self._finalize_side1(do_postrun=False if interval_completed else not loop1)
            self.touch_output_target()
        elif not self.side1.restored:
            self.runs.run1.move_files_to_output_dir()

        if self.args.keep_side2_output:
            if success2:
                self._finalize_side2(do_postrun=False if interval_completed else not loop2)
            elif not self.side2.restored:
                self.runs.run2.move_files_to_output_dir()
        else:
            remove_dir_if_exists(Path(self.args.side2_output_dir))

        return 0 if success1 else (rc1 if rc1 is not None else 1)


def build_pair_runs(args, benchmarks_root: Path) -> PairRuns:
    base = Path(args.run_dir)
    run1 = BenchmarkRun(args.benchmark1, str(base / "1"), str(Path(args.side1_output_dir)))
    run2 = BenchmarkRun(args.benchmark2, str(base / "2"), str(Path(args.side2_output_dir)))
    return PairRuns(
        run1=run1,
        run2=run2,
        cmd1=compose_submit_command(args.prefix1, args.submit1, args.loop_until1, benchmarks_root),
        cmd2=compose_submit_command(args.prefix2, args.submit2, args.loop_until2, benchmarks_root),
    )
