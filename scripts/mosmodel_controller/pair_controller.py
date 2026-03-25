from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..benchmarkCore import BenchmarkRun
from .finalize import ensure_parent, finalize_run, remove_dir_if_exists
from .launcher import (
    LaunchedSide,
    benchmark_group_pids,
    benchmark_group_pid_details,
    compose_submit_command,
    launch_run,
    launch_run_with_start_barrier,
    release_benchmark,
    signal_benchmark_group,
    signal_side,
    terminate_and_wait,
)
from .progress import (
    DetachedMeasurementPerfSession,
    LiveProgressConfig,
    MeasurementPerfConfig,
    WrappedPerfInstructionsMonitor,
)


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
    def __init__(self, args, benchmarks_root: Path, runs: PairRuns):
        self.args = args
        self.benchmarks_root = benchmarks_root
        self.runs = runs
        self.proc1: Optional[LaunchedSide] = None
        self.proc2: Optional[LaunchedSide] = None
        self.monitor1 = WrappedPerfInstructionsMonitor(
            LiveProgressConfig(
                perf_binary=self.args.progress_perf_binary,
                interval_ms=self.args.progress_interval_ms,
                label="side1",
            )
        )
        self.monitor2 = WrappedPerfInstructionsMonitor(
            LiveProgressConfig(
                perf_binary=self.args.progress_perf_binary,
                interval_ms=self.args.progress_interval_ms,
                label="side2",
            )
        )
        self.measurement1 = DetachedMeasurementPerfSession(
            MeasurementPerfConfig(
                perf_binary=self.args.progress_perf_binary,
                interval_ms=1000,
                label="side1",
                output_path=Path(self.args.output_target),
            )
        )
        self.measurement_attached = False
        self.measurement_enabled = False
        self._cleanup_started = False
        self.interval_mode = all(
            value is not None
            for value in (
                self.args.i_start_side1,
                self.args.i_end_side1,
                self.args.i_start_side2,
                self.args.i_end_side2,
            )
        )
        self.sync_interval_mode = bool(self.interval_mode and self.args.sync_interval_windows)
        self.side1_interval = (
            IntervalBoundaryState(
                label="side1",
                start_target=self.args.i_start_side1,
                end_target=self.args.i_end_side1,
            )
            if self.interval_mode
            else None
        )
        self.side2_interval = (
            IntervalBoundaryState(
                label="side2",
                start_target=self.args.i_start_side2,
                end_target=self.args.i_end_side2,
            )
            if self.interval_mode
            else None
        )

        self.sync_started = False
        self.sync_completed = False
        self.sync_paused_side: Optional[str] = None
        self.sync_completed_reason: Optional[str] = None
        self.sync_started_monotonic_sec: Optional[float] = None
        self.sync_started_side1_instructions: Optional[int] = None
        self.sync_started_side2_instructions: Optional[int] = None

        self.debug_stop_cont_enabled = os.environ.get(
            "MOSMODEL_CONTROLLER_DEBUG_STOP_CONT", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.debug_stop_cont_side = os.environ.get(
            "MOSMODEL_CONTROLLER_DEBUG_STOP_CONT_SIDE", "side1"
        ).strip().lower() or "side1"
        try:
            self.debug_stop_cont_sleep_sec = max(
                0.0,
                float(
                    os.environ.get(
                        "MOSMODEL_CONTROLLER_DEBUG_STOP_CONT_SLEEP_MS",
                        "500",
                    )
                ) / 1000.0,
            )
        except ValueError:
            self.debug_stop_cont_sleep_sec = 0.5
        try:
            self.sync_measurement_bootstrap_sec = max(
                0.0,
                float(
                    os.environ.get(
                        "MOSMODEL_CONTROLLER_SYNC_MEAS_BOOTSTRAP_MS",
                        "100",
                    )
                ) / 1000.0,
            )
        except ValueError:
            self.sync_measurement_bootstrap_sec = 0.1

        if self.debug_stop_cont_enabled:
            had_interval_flags = any(
                value is not None
                for value in (
                    self.args.i_start_side1,
                    self.args.i_end_side1,
                    self.args.i_start_side2,
                    self.args.i_end_side2,
                )
            ) or bool(getattr(self.args, "sync_interval_windows", False) or getattr(self.args, "debug_sync_ps", False))
            if had_interval_flags:
                print(
                    "[debug stop/cont] disabling interval-control flags during STOP/CONT validation"
                )
            self.args.i_start_side1 = None
            self.args.i_end_side1 = None
            self.args.i_start_side2 = None
            self.args.i_end_side2 = None
            self.args.sync_interval_windows = False
            self.args.debug_sync_ps = False
            self.interval_mode = False
            self.sync_interval_mode = False
            self.side1_interval = None
            self.side2_interval = None

    def install_signal_handlers(self) -> None:
        def cleanup_on_signal(signum, frame) -> None:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print(
                f"received {signal.Signals(signum).name}, terminating both sides...",
                file=sys.stderr,
            )
            self.terminate_both(grace_sec=1.0)
            self.join_monitors()
            self.print_interval_boundaries()
            self.print_sampled_instructions()
            self.print_measurement_perf_notes()
            self.write_interval_boundaries_summary()
            os._exit(128 + int(signum))

        signal.signal(signal.SIGINT, cleanup_on_signal)
        signal.signal(signal.SIGTERM, cleanup_on_signal)

    def prerun_both(self) -> None:
        print("running side2 prerun")
        self.runs.run2.prerun()
        print("running side1 prerun")
        self.runs.run1.prerun()

    def _require_benchmark_pid(self, launched: Optional[LaunchedSide], label: str) -> int:
        if launched is None or launched.benchmark_pid is None:
            raise RuntimeError(f"missing benchmark pid for {label}")
        return launched.benchmark_pid

    def launch_both(self) -> None:
        try:
            if self.args.sample_instructions:
                self.proc2 = launch_run_with_start_barrier(
                    self.runs.run2,
                    self.args.num_threads,
                    self.runs.cmd2,
                )
                self.monitor2.attach_to_pid(self._require_benchmark_pid(self.proc2, "side2"))
                self.monitor2.enable()
                release_benchmark(self.proc2)

                self.proc1 = launch_run_with_start_barrier(
                    self.runs.run1,
                    self.args.num_threads,
                    self.runs.cmd1,
                )
                side1_pid = self._require_benchmark_pid(self.proc1, "side1")
                self.monitor1.attach_to_pid(side1_pid)
                self.monitor1.enable()
                if not self.sync_interval_mode:
                    self.measurement1.attach_to_pid(side1_pid)
                    self.measurement_attached = True
                    self.measurement1.enable()
                    self.measurement_enabled = True
                release_benchmark(self.proc1)
                self._maybe_run_debug_stop_cont_validation()
            else:
                self.proc2 = launch_run(self.runs.run2, self.args.num_threads, self.runs.cmd2)
                self.proc1 = launch_run_with_start_barrier(
                    self.runs.run1,
                    self.args.num_threads,
                    self.runs.cmd1,
                )
                side1_pid = self._require_benchmark_pid(self.proc1, "side1")
                self.measurement1.attach_to_pid(side1_pid)
                self.measurement_attached = True
                self.measurement1.enable()
                self.measurement_enabled = True
                release_benchmark(self.proc1)
                self._maybe_run_debug_stop_cont_validation()
        except Exception:
            self.terminate_both(grace_sec=1.0)
            self.join_monitors()
            raise

    def terminate_both(self, grace_sec: float = 2.0) -> tuple[Optional[int], Optional[int]]:
        rc1 = terminate_and_wait(self.proc1, grace_sec=grace_sec)
        rc2 = terminate_and_wait(self.proc2, grace_sec=grace_sec)
        return rc1, rc2

    def _attach_sync_measurement_if_needed(self) -> None:
        if self.measurement_attached:
            return
        if not self.sync_interval_mode:
            return
        benchmark_pid = getattr(self.proc1, "benchmark_pid", None)
        benchmark_pgid = getattr(self.proc1, "benchmark_pgid", None)
        details = benchmark_group_pid_details(self.proc1)
        target_pids = details["final_target_pids"]
        print(
            f"[measurement perf] sync discovery side1: "
            f"benchmark_pgid={benchmark_pgid} benchmark_pid={benchmark_pid} "
            f"pgid_members={details['pgid_members']} descendants={details['descendants']} "
            f"final_target_pids={target_pids}"
        )
        target_rows = details.get("target_process_rows", [])
        if target_rows:
            print("[measurement perf] sync target process rows for side1:")
            for row in target_rows:
                print(
                    "  pid={pid} ppid={ppid} pgid={pgid} sid={sid} stat={stat} cmd={cmd}".format(**row)
                )
        else:
            print("[measurement perf] sync target process rows for side1: <none>")
        if not target_pids:
            raise RuntimeError(
                "no side1 benchmark-group pids available for sync measurement perf "
                f"(benchmark_pgid={benchmark_pgid}, benchmark_pid={benchmark_pid})"
            )
        print(
            f"[measurement perf] sync attach target group for side1: "
            f"benchmark_pgid={benchmark_pgid} pids={target_pids}"
        )
        self.measurement1.attach_to_pids(target_pids)
        self.measurement_attached = True

    def _enable_measurement_if_needed(self) -> None:
        if self.measurement_enabled:
            return
        if self.sync_interval_mode and not self.measurement_attached:
            self._attach_sync_measurement_if_needed()
        if not self.measurement_attached:
            return
        self.measurement1.enable()
        self.measurement_enabled = True

    def _disable_measurement_if_needed(self) -> None:
        if not self.measurement_attached or not self.measurement_enabled:
            return
        self.measurement1.disable()
        self.measurement_enabled = False

    def join_monitors(self) -> None:
        if self.args.sample_instructions:
            self.monitor1.stop(timeout=5.0)
            self.monitor2.stop(timeout=5.0)
        self.measurement1.stop(timeout=5.0)

    def print_sampled_instructions(self) -> None:
        if not self.args.sample_instructions:
            return
        total1 = self.monitor1.total_instructions()
        total2 = self.monitor2.total_instructions()
        grand_total = total1 + total2
        print("[sampled instructions]")
        print(f"  side1={total1}")
        print(f"  side2={total2}")
        print(f"  total={grand_total}")
        err1 = self.monitor1.last_error_line()
        err2 = self.monitor2.last_error_line()
        if err1:
            print(f"  side1_perf_note={err1}", file=sys.stderr)
        if err2:
            print(f"  side2_perf_note={err2}", file=sys.stderr)

    def print_measurement_perf_notes(self) -> None:
        err = self.measurement1.last_error_line()
        if err:
            print(f"  side1_measurement_perf_note={err}", file=sys.stderr)


    def _update_interval_state(
        self,
        state: Optional[IntervalBoundaryState],
        current_instructions: int,
        now_monotonic: float,
        *,
        allow_end: bool = True,
    ) -> None:
        if state is None:
            return
        if (not state.start_seen) and current_instructions >= state.start_target:
            state.start_seen = True
            state.start_observed_instructions = current_instructions
            state.start_observed_monotonic_sec = now_monotonic
            print(
                f"[interval boundary] {state.label} reached I_start "
                f"at instructions={current_instructions} "
                f"(threshold={state.start_target})"
            )
        if allow_end:
            if (not state.end_seen) and current_instructions >= state.end_target:
                state.end_seen = True
                state.end_observed_instructions = current_instructions
                state.end_observed_monotonic_sec = now_monotonic
                print(
                    f"[interval boundary] {state.label} reached I_end "
                    f"at instructions={current_instructions} "
                    f"(threshold={state.end_target})"
                )
        elif (
            state.start_seen
            and (not state.pre_sync_end_crossed)
            and current_instructions >= state.end_target
        ):
            state.pre_sync_end_crossed = True
            state.pre_sync_end_observed_instructions = current_instructions
            state.pre_sync_end_observed_monotonic_sec = now_monotonic
            print(
                f"[interval sync] warning: {state.label} crossed I_end before synchronization started "
                f"at instructions={current_instructions} (threshold={state.end_target})"
            )

    def _interval_boundaries_complete(self) -> bool:
        return bool(
            self.side1_interval is not None
            and self.side2_interval is not None
            and self.side1_interval.end_seen
            and self.side2_interval.end_seen
        )

    def _interval_completion_status(self) -> bool:
        if not self.interval_mode:
            return False
        if self.sync_interval_mode:
            return self.sync_completed
        return self._interval_boundaries_complete()

    def _debug_print_ps_snapshot(self, label: str, launched: Optional[LaunchedSide]) -> None:
        if launched is None:
            print(f"[debug stop/cont] {label}: side not launched")
            return
        print(
            f"[debug stop/cont] {label}: ps snapshot for sid={launched.sid}, "
            f"benchmark_pgid={launched.benchmark_pgid}"
        )
        result = subprocess.run(
            ["ps", "-o", "pid,ppid,pgid,sid,stat,cmd", "-s", str(launched.sid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = result.stdout.rstrip()
        if output:
            print(output)
        else:
            print("  <no processes matched>")

    def _maybe_run_debug_stop_cont_validation(self) -> None:
        if not self.debug_stop_cont_enabled:
            return
        if self.proc1 is None or self.proc2 is None:
            return

        target_label = self.debug_stop_cont_side
        if target_label not in {"side1", "side2"}:
            target_label = "side1"
        launched = self.proc1 if target_label == "side1" else self.proc2

        print(
            "[debug stop/cont] validating benchmark-group STOP/CONT "
            f"on {target_label} with sleep_sec={self.debug_stop_cont_sleep_sec}"
        )
        time.sleep(0.2)
        self._debug_print_ps_snapshot(f"before STOP {target_label}", launched)
        if not signal_benchmark_group(launched, signal.SIGSTOP):
            print(f"[debug stop/cont] failed to STOP {target_label}")
            return
        self._debug_print_ps_snapshot(f"after STOP {target_label}", launched)
        if self.debug_stop_cont_sleep_sec > 0:
            time.sleep(self.debug_stop_cont_sleep_sec)
        if not signal_benchmark_group(launched, signal.SIGCONT):
            print(f"[debug stop/cont] failed to CONT {target_label}")
            return
        self._debug_print_ps_snapshot(f"after CONT {target_label}", launched)

    def _print_sync_ps_snapshot(self, label: str, launched: Optional[LaunchedSide]) -> None:
        if not getattr(self.args, "debug_sync_ps", False):
            return
        self._debug_print_ps_snapshot(f"[interval sync debug] {label}", launched)

    def _pause_first_side_to_reach_start(self) -> None:
        if self.side1_interval is None or self.side2_interval is None:
            return
        if self.sync_paused_side is not None:
            return
        if self.side1_interval.start_seen and not self.side2_interval.start_seen:
            self._print_sync_ps_snapshot("before STOP side1", self.proc1)
            signal_benchmark_group(self.proc1, signal.SIGSTOP)
            self._print_sync_ps_snapshot("after STOP side1", self.proc1)
            self.sync_paused_side = "side1"
            print("[interval sync] side1 reached I_start first; STOP side1 and wait for side2")
        elif self.side2_interval.start_seen and not self.side1_interval.start_seen:
            self._print_sync_ps_snapshot("before STOP side2", self.proc2)
            signal_benchmark_group(self.proc2, signal.SIGSTOP)
            self._print_sync_ps_snapshot("after STOP side2", self.proc2)
            self.sync_paused_side = "side2"
            print("[interval sync] side2 reached I_start first; STOP side2 and wait for side1")

    def _choose_first_completed_side(self) -> str:
        assert self.side1_interval is not None and self.side2_interval is not None
        if self.side1_interval.end_seen and not self.side2_interval.end_seen:
            return "side1_end"
        if self.side2_interval.end_seen and not self.side1_interval.end_seen:
            return "side2_end"
        # Both seen in the same poll cycle. Pick the one whose observed timestamp is first.
        t1 = self.side1_interval.end_observed_monotonic_sec
        t2 = self.side2_interval.end_observed_monotonic_sec
        if t2 is None or (t1 is not None and t1 <= t2):
            return "side1_end"
        return "side2_end"

    def _maybe_apply_sync_interval_control(
        self,
        side1_instructions: int,
        side2_instructions: int,
        now_monotonic: float,
    ) -> bool:
        if not self.sync_interval_mode:
            return False
        assert self.side1_interval is not None and self.side2_interval is not None

        if not self.sync_started:
            self._pause_first_side_to_reach_start()
            if self.side1_interval.start_seen and self.side2_interval.start_seen:
                self._print_sync_ps_snapshot("before sync-start STOP side1", self.proc1)
                self._print_sync_ps_snapshot("before sync-start STOP side2", self.proc2)
                signal_benchmark_group(self.proc1, signal.SIGSTOP)
                signal_benchmark_group(self.proc2, signal.SIGSTOP)
                self._print_sync_ps_snapshot("after sync-start STOP side1", self.proc1)
                self._print_sync_ps_snapshot("after sync-start STOP side2", self.proc2)
                if self.sync_measurement_bootstrap_sec > 0.0:
                    print(
                        "[interval sync] bootstrap run before measurement attach: CONT both sides for "
                        f"{self.sync_measurement_bootstrap_sec:.3f}s so the real worker subtree can appear"
                    )
                    signal_benchmark_group(self.proc1, signal.SIGCONT)
                    signal_benchmark_group(self.proc2, signal.SIGCONT)
                    time.sleep(self.sync_measurement_bootstrap_sec)
                    signal_benchmark_group(self.proc1, signal.SIGSTOP)
                    signal_benchmark_group(self.proc2, signal.SIGSTOP)
                    time.sleep(0.02)
                    self._print_sync_ps_snapshot("after bootstrap STOP side1", self.proc1)
                    self._print_sync_ps_snapshot("after bootstrap STOP side2", self.proc2)
                side1_instructions = self.monitor1.total_instructions()
                side2_instructions = self.monitor2.total_instructions()
                now_monotonic = time.monotonic()
                self._enable_measurement_if_needed()
                self._print_sync_ps_snapshot("before CONT side1", self.proc1)
                self._print_sync_ps_snapshot("before CONT side2", self.proc2)
                signal_benchmark_group(self.proc1, signal.SIGCONT)
                signal_benchmark_group(self.proc2, signal.SIGCONT)
                self._print_sync_ps_snapshot("after CONT side1", self.proc1)
                self._print_sync_ps_snapshot("after CONT side2", self.proc2)
                self.sync_started = True
                self.sync_started_monotonic_sec = now_monotonic
                self.sync_started_side1_instructions = side1_instructions
                self.sync_started_side2_instructions = side2_instructions
                print(
                    "[interval sync] both sides reached I_start; stopped both benchmark groups, "
                    "enabled measurement perf, CONT both sides and begin shared interval"
                )
                self._update_interval_state(
                    self.side1_interval,
                    side1_instructions,
                    now_monotonic,
                    allow_end=True,
                )
                self._update_interval_state(
                    self.side2_interval,
                    side2_instructions,
                    now_monotonic,
                    allow_end=True,
                )

        if self.sync_started and not self.sync_completed:
            if self.side1_interval.end_seen or self.side2_interval.end_seen:
                self.sync_completed = True
                self.sync_completed_reason = self._choose_first_completed_side()
                reason_label = "side1" if self.sync_completed_reason == "side1_end" else "side2"
                self._disable_measurement_if_needed()
                print(
                    f"[interval sync] {reason_label} reached I_end after synchronization; "
                    "disabled measurement perf and terminating both sides"
                )
                return True
        return False

    def wait_with_sampled_instruction_control(self) -> IntervalControlResult:
        if self.proc1 is None:
            return IntervalControlResult(rc1=1, rc2=None, interval_completed=False)
        if self.proc2 is None:
            return IntervalControlResult(rc1=1, rc2=None, interval_completed=False)

        while True:
            rc1 = self.proc1.proc.poll()
            rc2 = self.proc2.proc.poll()
            side1_instructions = self.monitor1.total_instructions()
            side2_instructions = self.monitor2.total_instructions()
            if self.interval_mode:
                now_monotonic = time.monotonic()
                allow_end = (not self.sync_interval_mode) or self.sync_started
                self._update_interval_state(
                    self.side1_interval,
                    side1_instructions,
                    now_monotonic,
                    allow_end=allow_end,
                )
                self._update_interval_state(
                    self.side2_interval,
                    side2_instructions,
                    now_monotonic,
                    allow_end=allow_end,
                )

                if self.sync_interval_mode:
                    if self._maybe_apply_sync_interval_control(
                        side1_instructions,
                        side2_instructions,
                        now_monotonic,
                    ):
                        return IntervalControlResult(rc1=rc1, rc2=rc2, interval_completed=True)
                elif self._interval_boundaries_complete():
                    self._disable_measurement_if_needed()
                    print("[interval boundary] both sides reached I_end; disabled measurement perf and terminating both sides")
                    return IntervalControlResult(rc1=rc1, rc2=rc2, interval_completed=True)

            if self.interval_mode:
                if rc1 is not None or rc2 is not None:
                    return IntervalControlResult(rc1=rc1, rc2=rc2, interval_completed=False)
            else:
                if rc1 is not None and rc2 is not None:
                    return IntervalControlResult(rc1=rc1, rc2=rc2, interval_completed=False)

            time.sleep(0.05)

    def print_interval_boundaries(self) -> None:
        if not self.interval_mode:
            return
        print("[interval boundaries]")
        for state in (self.side1_interval, self.side2_interval):
            if state is None:
                continue
            print(
                f"  {state.label}: "
                f"start_target={state.start_target}, "
                f"start_observed={state.start_observed_instructions}, "
                f"end_target={state.end_target}, "
                f"end_observed={state.end_observed_instructions}, "
                f"pre_sync_end_crossed={state.pre_sync_end_crossed}, "
                f"pre_sync_end_observed={state.pre_sync_end_observed_instructions}"
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
        payload = {
            "interval_completed": self._interval_completion_status(),
            "sync_interval_mode": self.sync_interval_mode,
            "sync_started": self.sync_started,
            "sync_paused_side": self.sync_paused_side,
            "sync_started_monotonic_sec": self.sync_started_monotonic_sec,
            "sync_started_side1_instructions": self.sync_started_side1_instructions,
            "sync_started_side2_instructions": self.sync_started_side2_instructions,
            "sync_completed": self.sync_completed,
            "sync_completed_reason": self.sync_completed_reason,
            "side1": asdict(self.side1_interval) if self.side1_interval is not None else None,
            "side2": asdict(self.side2_interval) if self.side2_interval is not None else None,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def finalize_side1(self, *, do_postrun: bool) -> None:
        finalize_run(
            self.runs.run1,
            do_postrun=do_postrun,
            clean_threshold=self.args.clean_threshold,
            exclude_files=self.args.exclude_files,
        )

    def finalize_side2(self, *, do_postrun: bool) -> None:
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
        rc1: Optional[int]
        rc2: Optional[int]

        if self.args.sample_instructions:
            control_result = self.wait_with_sampled_instruction_control()
            interval_completed = control_result.interval_completed
            rc1 = control_result.rc1
            rc2 = control_result.rc2
            if interval_completed:
                self._disable_measurement_if_needed()
                term_rc1, term_rc2 = self.terminate_both(grace_sec=2.0)
                if rc1 is None:
                    rc1 = term_rc1
                if rc2 is None:
                    rc2 = term_rc2
            else:
                if rc1 is None and self.proc1 is not None:
                    rc1 = self.proc1.proc.wait()
                if rc2 is None and self.proc2 is not None:
                    rc2 = self.proc2.proc.wait()
        else:
            rc1 = self.proc1.proc.wait() if self.proc1 is not None else 1
            rc2 = terminate_and_wait(self.proc2, grace_sec=2.0)

        self._disable_measurement_if_needed()
        self.join_monitors()
        print(f"[run rc] side1={rc1} side2={rc2}")
        self.print_interval_boundaries()
        self.print_sampled_instructions()
        self.print_measurement_perf_notes()
        self.write_interval_boundaries_summary()

        side1_loop_mode = self.args.loop_until1 is not None and self.args.loop_until1 > 0
        side2_loop_mode = self.args.loop_until2 is not None and self.args.loop_until2 > 0

        if interval_completed:
            side1_success = True
            side2_success = True
        else:
            side1_success = (rc1 == 0) if not side1_loop_mode else (rc1 in (0, 124))
            side2_success = (rc2 == 0) if not side2_loop_mode else (rc2 in (0, 124, None))

        if side1_success:
            self.finalize_side1(do_postrun=(False if interval_completed else (not side1_loop_mode)))
            self.touch_output_target()
        else:
            self.runs.run1.move_files_to_output_dir()

        if self.args.keep_side2_output:
            if side2_success:
                self.finalize_side2(do_postrun=(False if interval_completed else (not side2_loop_mode)))
            else:
                self.runs.run2.move_files_to_output_dir()
        else:
            remove_dir_if_exists(Path(self.args.side2_output_dir))

        return 0 if side1_success else (rc1 if rc1 is not None else 1)



def build_pair_runs(args, benchmarks_root: Path) -> PairRuns:
    base_run_dir = Path(args.run_dir)
    side1_run_dir = base_run_dir / "1"
    side2_run_dir = base_run_dir / "2"

    side1_output_dir = Path(args.side1_output_dir)
    side2_output_dir = Path(args.side2_output_dir)

    run1 = BenchmarkRun(args.benchmark1, str(side1_run_dir), str(side1_output_dir))
    run2 = BenchmarkRun(args.benchmark2, str(side2_run_dir), str(side2_output_dir))

    cmd1 = compose_submit_command(
        args.prefix1,
        args.submit1,
        args.loop_until1,
        benchmarks_root,
    )
    cmd2 = compose_submit_command(
        args.prefix2,
        args.submit2,
        args.loop_until2,
        benchmarks_root,
    )

    return PairRuns(run1=run1, run2=run2, cmd1=cmd1, cmd2=cmd2)
