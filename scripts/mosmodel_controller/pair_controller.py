from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..benchmarkCore import BenchmarkRun
from .cgroup import CgroupV2
from .criu_restore import restore_stopped
from .finalize import ensure_parent, finalize_run, remove_dir_if_exists
from .launcher import (
    LaunchedSide,
    benchmark_group_pids,
    benchmark_group_pid_details,
    compose_submit_command,
    launch_run,
    launch_run_with_start_barrier,
    release_benchmark,
    terminate_and_wait,
    terminate_many_and_wait,
    signal_benchmark_group,
    signal_side,
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
        self.criu_run = bool(getattr(self.args, "criu_run", False))
        self.restore_side1 = bool(getattr(self.args, "checkpoint_dir1", None))
        self.restore_side2 = bool(getattr(self.args, "checkpoint_dir2", None))
        self.proc1: Optional[LaunchedSide] = None
        self.proc2: Optional[LaunchedSide] = None
        self.criu_cgroup1: Optional[CgroupV2] = None
        self.criu_cgroup2: Optional[CgroupV2] = None
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
        self.side1_interval = self._make_interval_state(1) if self.interval_mode else None
        self.side2_interval = self._make_interval_state(2) if self.interval_mode else None

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

        self.external_resume_gate_dir: Optional[Path] = None
        self.external_ready_file: Optional[Path] = None
        self.external_resume_file: Optional[Path] = None
        self.external_state_file: Optional[Path] = None
        self.external_resume_socket_path: str = getattr(self.args, "external_resume_socket_path", "") or ""
        self.external_resume_token: str = getattr(self.args, "external_resume_token", "") or ""
        if getattr(self.args, "external_resume_gate_dir", ""):
            self.external_resume_gate_dir = Path(self.args.external_resume_gate_dir)
            self.external_ready_file = self.external_resume_gate_dir / "READY"
            self.external_resume_file = self.external_resume_gate_dir / "RESUME"
            self.external_state_file = self.external_resume_gate_dir / "STATE.json"
            self.external_resume_gate_dir.mkdir(parents=True, exist_ok=True)
            for stale in (self.external_ready_file, self.external_resume_file, self.external_state_file):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass

        self.debug_log_path = Path(self.args.run_dir) / 'controller_debug.log'
        ensure_parent(self.debug_log_path)
        try:
            self.debug_log_path.unlink()
        except FileNotFoundError:
            pass
        self._debug_log('pair controller initialized')

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

    def _make_interval_state(self, side: int) -> IntervalBoundaryState:
        start = self.args.i_start_side1 if side == 1 else self.args.i_start_side2
        end = self.args.i_end_side1 if side == 1 else self.args.i_end_side2
        restored = self.restore_side1 if side == 1 else self.restore_side2
        label = f"side{side}"

        if restored:
            return IntervalBoundaryState(
                label=label,
                start_target=0,
                end_target=end - start,
                start_seen=True,
                start_observed_instructions=0,
                start_observed_monotonic_sec=time.monotonic(),
            )
        return IntervalBoundaryState(label=label, start_target=start, end_target=end)

    def _external_gate_enabled(self) -> bool:
        return self.external_resume_gate_dir is not None or bool(self.external_resume_socket_path)

    def _external_socket_enabled(self) -> bool:
        return bool(self.external_resume_socket_path)


    def _debug_log(self, message: str) -> None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        line = f'[{ts}] {message}'
        print(line)
        try:
            with self.debug_log_path.open('a', encoding='utf-8') as fh:
                fh.write(line + '\n')
        except Exception:
            pass

    def _ps_snapshot(self, label: str, launched: Optional[LaunchedSide]) -> None:
        if launched is None:
            self._debug_log(f'{label}: side not launched')
            return
        result = subprocess.run(
            ['ps', '-o', 'pid,ppid,pgid,sid,psr,stat,pcpu,time,cmd', '-s', str(launched.sid)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        output = result.stdout.rstrip()
        self._debug_log(
            f'{label}: sid={launched.sid} benchmark_pgid={launched.benchmark_pgid} benchmark_pid={launched.benchmark_pid}\n'
            + (output if output else '<no processes matched>')
        )

    def _write_external_gate_state(self, state: str, **extra) -> None:
        if not self._external_gate_enabled() or self.external_state_file is None:
            return
        payload = {
            "state": state,
            "run_dir": str(self.args.run_dir),
            "output_target": str(self.args.output_target),
            "side1_output_dir": str(self.args.side1_output_dir),
            "side2_output_dir": str(self.args.side2_output_dir),
            "side1_benchmark_pgid": getattr(self.proc1, "benchmark_pgid", None),
            "side2_benchmark_pgid": getattr(self.proc2, "benchmark_pgid", None),
            "side1_instructions": self.monitor1.total_instructions() if self.args.sample_instructions else None,
            "side2_instructions": self.monitor2.total_instructions() if self.args.sample_instructions else None,
            "timestamp_monotonic_sec": time.monotonic(),
        }
        payload.update(extra)
        self.external_state_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _wait_for_external_resume_if_needed(self, side1_instructions: int, side2_instructions: int) -> None:
        if not self._external_gate_enabled():
            return
        self._write_external_gate_state(
            "ready_waiting_for_resume",
            side1_instructions=side1_instructions,
            side2_instructions=side2_instructions,
            sync_started=False,
        )
        if self._external_socket_enabled():
            payload = {
                "event": "READY",
                "token": self.external_resume_token,
                "run_dir": str(self.args.run_dir),
                "output_target": str(self.args.output_target),
                "side1_output_dir": str(self.args.side1_output_dir),
                "side2_output_dir": str(self.args.side2_output_dir),
                "side1_benchmark_pgid": getattr(self.proc1, "benchmark_pgid", None),
                "side2_benchmark_pgid": getattr(self.proc2, "benchmark_pgid", None),
                "side1_instructions": side1_instructions,
                "side2_instructions": side2_instructions,
                "state_path": str(self.external_state_file) if self.external_state_file is not None else None,
            }
            self._debug_log(f'[external resume socket] both sides are aligned and STOPped; waiting for RESUME from {self.external_resume_socket_path}')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(self.external_resume_socket_path)
                sock.sendall((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
                buffer = bytearray()
                while True:
                    rc1 = self.proc1.proc.poll() if self.proc1 is not None else None
                    rc2 = self.proc2.proc.poll() if self.proc2 is not None else None
                    if rc1 is not None or rc2 is not None:
                        raise RuntimeError(
                            "controller was waiting for external resume on socket, but one side exited early: "
                            f"side1_rc={rc1} side2_rc={rc2}"
                        )
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise RuntimeError("external resume socket closed before RESUME")
                    buffer.extend(chunk)
                    if b"\n" not in buffer:
                        continue
                    line, _rest = buffer.split(b"\n", 1)
                    response = line.decode("utf-8", errors="replace").strip()
                    if response == "RESUME":
                        self._write_external_gate_state(
                            "resumed",
                            side1_instructions=self.monitor1.total_instructions() if self.args.sample_instructions else side1_instructions,
                            side2_instructions=self.monitor2.total_instructions() if self.args.sample_instructions else side2_instructions,
                            sync_started=False,
                        )
                        self._debug_log('[external resume socket] RESUME observed; continuing synchronized interval startup')
                        return
                    raise RuntimeError(f"unexpected external resume socket response: {response!r}")

        assert self.external_ready_file is not None and self.external_resume_file is not None
        self.external_ready_file.write_text("READY\n")
        self._debug_log(f'[external resume gate] both sides are aligned and STOPped; waiting for RESUME at {self.external_resume_file}')
        while True:
            if self.external_resume_file.exists():
                try:
                    self.external_resume_file.unlink()
                except FileNotFoundError:
                    pass
                try:
                    self.external_ready_file.unlink()
                except FileNotFoundError:
                    pass
                self._write_external_gate_state(
                    "resumed",
                    side1_instructions=self.monitor1.total_instructions() if self.args.sample_instructions else side1_instructions,
                    side2_instructions=self.monitor2.total_instructions() if self.args.sample_instructions else side2_instructions,
                    sync_started=False,
                )
                self._debug_log('[external resume gate] RESUME observed; continuing synchronized interval startup')
                return
            rc1 = self.proc1.proc.poll() if self.proc1 is not None else None
            rc2 = self.proc2.proc.poll() if self.proc2 is not None else None
            if rc1 is not None or rc2 is not None:
                raise RuntimeError(
                    "controller was waiting for external resume, but one side exited early: "
                    f"side1_rc={rc1} side2_rc={rc2}"
                )
            time.sleep(0.05)

    def install_signal_handlers(self) -> None:
        def cleanup_on_signal(signum, frame) -> None:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            self._debug_log(f'received {signal.Signals(signum).name}, terminating both sides')
            self._ps_snapshot('signal-handler side1', self.proc1)
            self._ps_snapshot('signal-handler side2', self.proc2)
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
        if not self.restore_side2:
            print("running side2 prerun")
            self.runs.run2.prerun()
        if not self.restore_side1:
            print("running side1 prerun")
            self.runs.run1.prerun()

    def _require_benchmark_pid(self, launched: Optional[LaunchedSide], label: str) -> int:
        if launched is None or launched.benchmark_pid is None:
            raise RuntimeError(f"missing benchmark pid for {label}")
        return launched.benchmark_pid

    def _side_interval(self, side: int) -> IntervalBoundaryState:
        state = self.side1_interval if side == 1 else self.side2_interval
        if state is None:
            raise RuntimeError(f"side{side} has no interval state")
        return state

    def _side_monitor(self, side: int) -> WrappedPerfInstructionsMonitor:
        return self.monitor1 if side == 1 else self.monitor2

    def _side_proc(self, side: int) -> Optional[LaunchedSide]:
        return self.proc1 if side == 1 else self.proc2

    def _set_side_proc(self, side: int, launched: LaunchedSide) -> None:
        if side == 1:
            self.proc1 = launched
        else:
            self.proc2 = launched

    def _side_is_restored(self, side: int) -> bool:
        return self.restore_side1 if side == 1 else self.restore_side2

    def _side_cgroup(self, side: int) -> Optional[CgroupV2]:
        return self.criu_cgroup1 if side == 1 else self.criu_cgroup2

    def _set_side_cgroup(self, side: int, cgroup: CgroupV2) -> None:
        if side == 1:
            self.criu_cgroup1 = cgroup
        else:
            self.criu_cgroup2 = cgroup

    def _side_checkpoint_dir(self, side: int) -> Path:
        value = self.args.checkpoint_dir1 if side == 1 else self.args.checkpoint_dir2
        if not value:
            raise RuntimeError(f"side{side} has no checkpoint directory")
        return Path(value)

    def _side_checkpoint_archive_dir(self, side: int) -> Path:
        value = (
            self.args.checkpoint_archive_dir1
            if side == 1
            else self.args.checkpoint_archive_dir2
        )
        if not value:
            raise RuntimeError(f"side{side} has no checkpoint archive directory")
        return Path(value)

    def _prepare_restored_side(self, side: int) -> None:
        restored = restore_stopped(
            checkpoint_dir=self._side_checkpoint_dir(side),
            checkpoint_archive_dir=self._side_checkpoint_archive_dir(side),
            output_dir=Path(
                self.args.side1_output_dir if side == 1 else self.args.side2_output_dir
            ),
            prefix=self._side_prefix(side),
        )
        self._set_side_proc(side, restored.as_launched_side())
        self._set_side_cgroup(side, restored.cgroup)
        monitor = self._side_monitor(side)
        if self.args.sample_instructions:
            monitor.attach_to_cgroup(restored.cgroup.perf_name)
            monitor.enable()
        self._debug_log(
            f"prepared restored side{side}: cgroup={restored.cgroup.perf_name} "
            f"root_pid={restored.root_pid} benchmark_pid={restored.benchmark_pid}"
        )

    def _prepare_native_side(
        self,
        side: int,
        *,
        cmd: Optional[str] = None,
        attach_progress: bool = True,
    ) -> None:
        launched = launch_run_with_start_barrier(
            self._side_run(side),
            self.args.num_threads,
            self._side_cmd(side) if cmd is None else cmd,
        )
        self._set_side_proc(side, launched)
        if attach_progress and self.args.sample_instructions:
            monitor = self._side_monitor(side)
            monitor.attach_to_pid(self._require_benchmark_pid(launched, f"side{side}"))
            monitor.enable()
        self._debug_log(
            f"prepared native side{side} behind start barrier: "
            f"benchmark_pid={getattr(launched, 'benchmark_pid', None)}"
        )

    def _prepare_side(
        self,
        side: int,
        *,
        cmd: Optional[str] = None,
        attach_progress: bool = True,
    ) -> None:
        if self._side_is_restored(side):
            if cmd is not None and cmd != self._side_cmd(side):
                raise RuntimeError(f"cannot override command for restored side{side}")
            self._prepare_restored_side(side)
        else:
            self._prepare_native_side(side, cmd=cmd, attach_progress=attach_progress)

    def _resume_prepared_side(self, side: int) -> None:
        launched = self._side_proc(side)
        if self._side_is_restored(side):
            if not signal_benchmark_group(launched, signal.SIGCONT):
                raise RuntimeError(f"failed to resume restored side{side}")
        elif not release_benchmark(launched):
            raise RuntimeError(f"failed to release native side{side} start barrier")

    def _mark_prepared_side_at_start(self, side: int) -> None:
        state = self._side_interval(side)
        if self._side_is_restored(side) or state.start_target == 0:
            state.start_seen = True
            state.start_observed_instructions = 0
            state.start_observed_monotonic_sec = time.monotonic()

    def _side_run(self, side: int) -> BenchmarkRun:
        return self.runs.run1 if side == 1 else self.runs.run2

    def _side_cmd(self, side: int) -> str:
        return self.runs.cmd1 if side == 1 else self.runs.cmd2

    def _side_prefix(self, side: int) -> str:
        return self.args.prefix1 if side == 1 else self.args.prefix2

    def _side_submit(self, side: int) -> str:
        return self.args.submit1 if side == 1 else self.args.submit2

    def _signal_benchmark_side(self, side: int, sig: int) -> None:
        # Use the full side session plus the inner benchmark process group. Some
        # benchmark wrappers can create children outside the original PGID.
        launched = self._side_proc(side)
        signal_side(launched, sig)
        signal_benchmark_group(launched, sig)

    def _launch_criu_pair(self) -> None:
        if not self.sync_interval_mode:
            raise RuntimeError("SMT CRIU mode requires synchronized instruction intervals")
        try:
            # Prepare both sides without allowing either benchmark to execute.
            # Restored sides are left stopped by CRIU; native I_start=0 sides
            # remain behind the start barrier.
            self._prepare_side(2)
            self._prepare_side(1)
            self._mark_prepared_side_at_start(1)
            self._mark_prepared_side_at_start(2)

            assert self.side1_interval is not None and self.side2_interval is not None
            if not (self.side1_interval.start_seen and self.side2_interval.start_seen):
                raise RuntimeError(
                    "SMT CRIU preparation did not place both sides at their interval starts"
                )

            self._enable_measurement_if_needed()
            self._resume_prepared_side(2)
            self._resume_prepared_side(1)

            now = time.monotonic()
            self.sync_started = True
            self.sync_started_monotonic_sec = now
            self.sync_started_side1_instructions = self.monitor1.total_instructions()
            self.sync_started_side2_instructions = self.monitor2.total_instructions()
            self._debug_log(
                "[interval sync] SMT CRIU pair prepared at I_start; enabled "
                "measurement and resumed both sides"
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
                self._debug_log(f'launch_both sample path side1_pid={side1_pid} side1_benchmark_pgid={getattr(self.proc1, "benchmark_pgid", None)} side2_benchmark_pgid={getattr(self.proc2, "benchmark_pgid", None)}')
                self._ps_snapshot('launch_both after release side1', self.proc1)
                self._ps_snapshot('launch_both after release side2', self.proc2)
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
                self._debug_log(f'launch_both nonsample path side1_pid={side1_pid} side1_benchmark_pgid={getattr(self.proc1, "benchmark_pgid", None)} side2_benchmark_pgid={getattr(self.proc2, "benchmark_pgid", None)}')
                self._ps_snapshot('launch_both after release side1', self.proc1)
                self._ps_snapshot('launch_both after release side2', self.proc2)
                self._maybe_run_debug_stop_cont_validation()
        except Exception:
            self.terminate_both()
            self.join_monitors()
            raise

    def terminate_both(self) -> tuple[Optional[int], Optional[int]]:
        for side in (1, 2):
            cgroup = self._side_cgroup(side)
            if cgroup is None:
                continue
            try:
                cgroup.kill()
            except Exception as exc:
                self._debug_log(f"WARNING: failed to kill side{side} CRIU cgroup: {exc}")
        self._debug_log("sending SIGTERM to both side sessions for graceful cleanup")
        rc1, rc2 = terminate_many_and_wait((self.proc1, self.proc2))
        return rc1, rc2

    def _remove_criu_cgroups(self) -> None:
        for side in (1, 2):
            cgroup = self._side_cgroup(side)
            if cgroup is None:
                continue
            try:
                if cgroup.is_populated():
                    cgroup.kill()
                cgroup.remove()
                self._debug_log(f"removed side{side} CRIU cgroup {cgroup.perf_name}")
            except Exception as exc:
                self._debug_log(f"WARNING: failed to remove side{side} CRIU cgroup: {exc}")
            finally:
                if side == 1:
                    self.criu_cgroup1 = None
                else:
                    self.criu_cgroup2 = None

    def _attach_sync_measurement_if_needed(self) -> None:
        if self.measurement_attached:
            return
        if not self.sync_interval_mode:
            return
        if self.restore_side1:
            cgroup = self.criu_cgroup1
            if cgroup is None:
                raise RuntimeError("missing side1 CRIU cgroup for measurement perf")
            print(f"[measurement perf] sync attach side1 CRIU cgroup={cgroup.perf_name}")
            self.measurement1.attach_to_cgroup(cgroup.perf_name)
            self.measurement_attached = True
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
        if len(target_pids) == 1 and target_pids[0] == benchmark_pid:
            print(
                "[measurement perf] side1 worker subtree has not expanded; "
                "attach to the stable benchmark leader so perf inheritance "
                "covers descendants created after resume"
            )
        else:
            print(
                f"[measurement perf] side1 worker subtree is ready: "
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
            self.monitor1.stop()
            self.monitor2.stop()
        self.measurement1.stop()

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
            self._debug_log(f'[interval boundary] {state.label} reached I_start at instructions={current_instructions} (threshold={state.start_target})')
        if allow_end:
            if (not state.end_seen) and current_instructions >= state.end_target:
                state.end_seen = True
                state.end_observed_instructions = current_instructions
                state.end_observed_monotonic_sec = now_monotonic
                self._debug_log(f'[interval boundary] {state.label} reached I_end at instructions={current_instructions} (threshold={state.end_target})')
        elif (
            state.start_seen
            and (not state.pre_sync_end_crossed)
            and current_instructions >= state.end_target
        ):
            state.pre_sync_end_crossed = True
            state.pre_sync_end_observed_instructions = current_instructions
            state.pre_sync_end_observed_monotonic_sec = now_monotonic
            self._debug_log(f'[interval sync] warning: {state.label} crossed I_end before synchronization started at instructions={current_instructions} (threshold={state.end_target})')

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
        self._ps_snapshot(label, launched)

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
            self._debug_log('[interval sync] side1 reached I_start first; STOP side1 and wait for side2')
        elif self.side2_interval.start_seen and not self.side1_interval.start_seen:
            self._print_sync_ps_snapshot("before STOP side2", self.proc2)
            signal_benchmark_group(self.proc2, signal.SIGSTOP)
            self._print_sync_ps_snapshot("after STOP side2", self.proc2)
            self.sync_paused_side = "side2"
            self._debug_log('[interval sync] side2 reached I_start first; STOP side2 and wait for side1')

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
                self._ps_snapshot('sync-start both STOPped side1', self.proc1)
                self._ps_snapshot('sync-start both STOPped side2', self.proc2)
                self._wait_for_external_resume_if_needed(side1_instructions, side2_instructions)
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
                self._write_external_gate_state(
                    "sync_started",
                    sync_started=True,
                    side1_instructions=side1_instructions,
                    side2_instructions=side2_instructions,
                )
                self._debug_log('[interval sync] both sides reached I_start; stopped both benchmark groups, enabled measurement perf, CONT both sides and begin shared interval')
                self._ps_snapshot('sync-start after CONT side1', self.proc1)
                self._ps_snapshot('sync-start after CONT side2', self.proc2)
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
                self._debug_log(f'[interval sync] {reason_label} reached I_end after synchronization; disabled measurement perf and terminating both sides')
                self._ps_snapshot('sync-complete side1', self.proc1)
                self._ps_snapshot('sync-complete side2', self.proc2)
                self._write_external_gate_state(
                    "sync_completed",
                    sync_started=True,
                    completed_reason=self.sync_completed_reason,
                    side1_instructions=side1_instructions,
                    side2_instructions=side2_instructions,
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
                    self._debug_log('[interval boundary] both sides reached I_end; disabled measurement perf and terminating both sides')
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
        if self.restore_side1:
            return
        finalize_run(
            self.runs.run1,
            do_postrun=do_postrun,
            clean_threshold=self.args.clean_threshold,
            exclude_files=self.args.exclude_files,
        )

    def finalize_side2(self, *, do_postrun: bool) -> None:
        if self.restore_side2:
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

        interval_completed = False
        rc1: Optional[int]
        rc2: Optional[int]

        if self.args.sample_instructions:
            self.launch_both()
            control_result = self.wait_with_sampled_instruction_control()
            interval_completed = control_result.interval_completed
            rc1 = control_result.rc1
            rc2 = control_result.rc2
            if interval_completed:
                self._disable_measurement_if_needed()
                term_rc1, term_rc2 = self.terminate_both()
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
            self.launch_both()
            rc1 = self.proc1.proc.wait() if self.proc1 is not None else 1
            rc2 = terminate_and_wait(self.proc2)

        self._disable_measurement_if_needed()
        self.join_monitors()
        self._remove_criu_cgroups()
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
        elif not self.restore_side1:
            self.runs.run1.move_files_to_output_dir()

        if self.args.keep_side2_output:
            if side2_success:
                self.finalize_side2(do_postrun=(False if interval_completed else (not side2_loop_mode)))
            elif not self.restore_side2:
                self.runs.run2.move_files_to_output_dir()
        else:
            remove_dir_if_exists(Path(self.args.side2_output_dir))

        self._write_external_gate_state(
            "finished",
            sync_started=self.sync_started,
            interval_completed=interval_completed,
            rc1=rc1,
            rc2=rc2,
            side1_success=side1_success,
            side2_success=side2_success,
        )
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
