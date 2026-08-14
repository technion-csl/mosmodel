from __future__ import annotations

import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..benchmarkCore import BenchmarkRun
from .cgroup import CgroupV2
from .criu_restore import restore_stopped
from .launcher import (
    LaunchedSide,
    benchmark_group_pids,
    launch_run_at_benchmark_start,
    launch_run_with_start_barrier,
    release_benchmark,
    resume_benchmark_start_gate,
    signal_benchmark_group,
)
from .progress import (
    DetachedMeasurementPerfSession,
    LiveProgressConfig,
    WrappedPerfInstructionsMonitor,
)


@dataclass(frozen=True)
class SideConfig:
    label: str
    run: BenchmarkRun
    cmd: str
    prefix: str
    output_dir: Path
    i_start: Optional[int] = None
    checkpoint_dir: Optional[Path] = None
    checkpoint_archive_dir: Optional[Path] = None

    @property
    def restored(self) -> bool:
        return self.checkpoint_dir is not None

    def validate(self) -> None:
        if (self.checkpoint_dir is None) != (self.checkpoint_archive_dir is None):
            raise ValueError(
                f"{self.label}: checkpoint_dir and checkpoint_archive_dir must be provided together"
            )
        if self.restored and (self.i_start is None or self.i_start <= 0):
            raise ValueError(
                f"{self.label}: checkpoints are only valid when I_start > 0"
            )


class ControllerSide:
    """One benchmark side, independent of single-vs-pair orchestration."""

    def __init__(
        self,
        config: SideConfig,
        *,
        perf_binary: str,
        progress_interval_ms: int,
    ) -> None:
        config.validate()
        self.config = config
        self.proc: Optional[LaunchedSide] = None
        self.cgroup: Optional[CgroupV2] = None
        self.cgroup_frozen = False
        self.monitor = WrappedPerfInstructionsMonitor(
            LiveProgressConfig(
                perf_binary=perf_binary,
                interval_ms=progress_interval_ms,
                label=config.label,
            )
        )

    @property
    def label(self) -> str:
        return self.config.label

    @property
    def restored(self) -> bool:
        return self.config.restored

    @property
    def run(self) -> BenchmarkRun:
        return self.config.run

    @property
    def benchmark_pid(self) -> int:
        if self.proc is None or self.proc.benchmark_pid is None:
            raise RuntimeError(f"missing benchmark pid for {self.label}")
        return self.proc.benchmark_pid

    def prerun(self) -> None:
        if not self.restored:
            self.run.prerun()

    def prepare(
        self,
        *,
        num_threads: int,
        sample_instructions: bool,
        log: Callable[[str], None],
    ) -> None:
        if self.proc is not None:
            raise RuntimeError(f"{self.label} is already prepared")

        if self.restored:
            assert self.config.checkpoint_dir is not None
            assert self.config.checkpoint_archive_dir is not None
            restored = restore_stopped(
                checkpoint_dir=self.config.checkpoint_dir,
                checkpoint_archive_dir=self.config.checkpoint_archive_dir,
                output_dir=self.config.output_dir,
                prefix=self.config.prefix,
            )
            self.proc = restored.as_launched_side()
            self.cgroup = restored.cgroup
            if sample_instructions:
                self.monitor.attach_to_cgroup(restored.cgroup.perf_name)
            log(
                f"prepared restored {self.label}: cgroup={restored.cgroup.perf_name} "
                f"root_pid={restored.root_pid} benchmark_pid={restored.benchmark_pid}"
            )
        else:
            if self.config.i_start == 0:
                self.proc = launch_run_at_benchmark_start(
                    self.run,
                    num_threads,
                    self.config.cmd,
                )
                gate = "run.sh start gate"
            else:
                self.proc = launch_run_with_start_barrier(
                    self.run,
                    num_threads,
                    self.config.cmd,
                )
                gate = "launcher start barrier"

            if sample_instructions:
                self.monitor.attach_to_pid(self.benchmark_pid)
            log(
                f"prepared native {self.label} at {gate}: "
                f"benchmark_pid={self.benchmark_pid}"
            )

        if sample_instructions:
            self.monitor.enable()

    def resume(self) -> None:
        if self.proc is None:
            raise RuntimeError(f"{self.label} is not prepared")
        if self.restored:
            if not signal_benchmark_group(self.proc, signal.SIGCONT):
                raise RuntimeError(f"failed to resume restored {self.label}")
        elif self.config.i_start == 0:
            if not resume_benchmark_start_gate(self.proc):
                raise RuntimeError(f"failed to resume native {self.label} at run.sh gate")
        elif not release_benchmark(self.proc):
            raise RuntimeError(f"failed to release native {self.label} start barrier")

    def signal_benchmark(self, sig: int) -> bool:
        return signal_benchmark_group(self.proc, sig)

    def instruction_count(self) -> int:
        return self.monitor.total_instructions()

    def attach_measurement(self, measurement: DetachedMeasurementPerfSession) -> None:
        if self.restored:
            if self.cgroup is None:
                raise RuntimeError(f"missing CRIU cgroup for {self.label}")
            measurement.attach_to_cgroup(self.cgroup.perf_name)
            return

        if self.config.i_start == 0:
            # The gated run.sh PID is the benchmark lineage. perf inheritance
            # covers its descendants without counting affinity/mosalloc parents.
            measurement.attach_to_pid(self.benchmark_pid)
            return

        pids = benchmark_group_pids(self.proc)
        if not pids:
            raise RuntimeError(f"no benchmark pids available for {self.label}")
        measurement.attach_to_pids(pids)

    def kill_cgroup(self) -> None:
        if self.cgroup is not None and self.cgroup.is_populated():
            self.cgroup.kill()

    def remove_cgroup(self, log: Callable[[str], None]) -> None:
        if self.cgroup is None:
            return
        cgroup = self.cgroup
        try:
            if cgroup.is_populated():
                cgroup.kill()
            cgroup.remove()
            log(f"removed {self.label} CRIU cgroup {cgroup.perf_name}")
        except Exception as exc:
            log(f"WARNING: failed to remove {self.label} CRIU cgroup: {exc}")
        finally:
            self.cgroup = None
            self.cgroup_frozen = False
