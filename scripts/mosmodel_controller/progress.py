from __future__ import annotations

import os
import select
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TextIO


@dataclass
class LiveProgressConfig:
    perf_binary: str
    interval_ms: int
    label: str


@dataclass
class MeasurementPerfConfig:
    perf_binary: str
    interval_ms: int
    label: str
    output_path: Path


class WrappedPerfInstructionsMonitor:
    def __init__(self, config: LiveProgressConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._total_instructions = 0
        self._last_error_line: Optional[str] = None
        self._ctl_write_fd: Optional[int] = None
        self._ack_read_fd: Optional[int] = None

    def attach_to_pid(self, target_pid: int) -> None:
        self._attach(["-p", str(int(target_pid))])

    def attach_to_cgroup(self, cgroup_name: str) -> None:
        if not cgroup_name.strip():
            raise RuntimeError(f"empty cgroup name for {self.config.label}")
        self._attach(["-a", "-G", cgroup_name])

    def _attach(self, target_args: list[str]) -> None:
        if self._proc is not None:
            raise RuntimeError(f"progress monitor already attached for {self.config.label}")
        if self.config.interval_ms <= 0:
            raise RuntimeError(
                f"invalid progress interval for {self.config.label}: {self.config.interval_ms}"
            )

        ctl_read_fd, ctl_write_fd = os.pipe()
        ack_read_fd, ack_write_fd = os.pipe()

        cmd = [
            self.config.perf_binary,
            "stat",
            "-e",
            "instructions",
            "-I",
            str(self.config.interval_ms),
            "-x",
            ";",
            "--no-big-num",
            "--delay=-1",
            f"--control=fd:{ctl_read_fd},{ack_write_fd}",
            *target_args,
        ]

        print(f"[progress perf] spawn {self.config.label}: {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                pass_fds=(ctl_read_fd, ack_write_fd),
            )
        finally:
            os.close(ctl_read_fd)
            os.close(ack_write_fd)

        self._ctl_write_fd = ctl_write_fd
        self._ack_read_fd = ack_read_fd

        if self._proc.stderr is None:
            raise RuntimeError(f"no perf stream available for {self.config.label}")

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stderr,),
            name=f"perf-progress-{self.config.label}",
            daemon=True,
        )
        self._reader_thread.start()

    def enable(self) -> None:
        if self._proc is None or self._ctl_write_fd is None or self._ack_read_fd is None:
            raise RuntimeError(f"progress monitor not attached for {self.config.label}")
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"perf exited before enable for {self.config.label} rc={self._proc.returncode}"
            )

        os.write(self._ctl_write_fd, b"enable\n")
        chunks: list[bytes] = []
        while True:
            select.select([self._ack_read_fd], [], [])
            chunk = os.read(self._ack_read_fd, 4096)
            if not chunk:
                returncode = self._proc.wait()
                note = self.last_error_line()
                if note:
                    raise RuntimeError(f"failed to enable perf for {self.config.label}: {note}")
                raise RuntimeError(
                    f"perf exited while waiting for enable ack for {self.config.label} "
                    f"rc={returncode}"
                )
            chunks.append(chunk)
            data = b"".join(chunks)
            if b"\n" not in data:
                continue
            line, _, _rest = data.partition(b"\n")
            if line.strip() != b"ack":
                raise RuntimeError(
                    f"unexpected perf ack for {self.config.label}: {line.decode(errors='replace')!r}"
                )
            return

    def _reader_loop(self, perf_stream: TextIO) -> None:
        debug_path = f"/tmp/progress_perf_{self.config.label}.log"
        with open(debug_path, "w") as dbg:
            for raw_line in perf_stream:
                dbg.write(raw_line)
                dbg.flush()

                line = raw_line.strip()
                if not line:
                    continue

                delta = self._parse_instructions_delta(line)
                if delta is not None:
                    with self._lock:
                        self._total_instructions += delta
                    continue

                lowered = line.lower()
                if "failed" in lowered or "error:" in lowered or "permission denied" in lowered:
                    self._last_error_line = line

    @staticmethod
    def _parse_instructions_delta(line: str) -> Optional[int]:
        fields = [field.strip() for field in line.split(";")]
        if not fields:
            return None

        event_idx = None
        for i, field in enumerate(fields):
            if field.startswith("instructions"):
                event_idx = i
                break

        if event_idx is None:
            return None

        for j in range(event_idx - 1, -1, -1):
            count_str = fields[j]
            lowered = count_str.lower()
            if lowered in {"<not counted>", "<not supported>", "nan"}:
                return None

            normalized = count_str.replace(",", "")
            try:
                value = float(normalized)
            except ValueError:
                continue

            if value < 0:
                return None
            return int(round(value))

        return None

    def total_instructions(self) -> int:
        with self._lock:
            return self._total_instructions

    def last_error_line(self) -> Optional[str]:
        return self._last_error_line

    def stop(self) -> None:
        if self._ctl_write_fd is not None:
            try:
                os.close(self._ctl_write_fd)
            except OSError:
                pass
            self._ctl_write_fd = None

        if self._ack_read_fd is not None:
            try:
                os.close(self._ack_read_fd)
            except OSError:
                pass
            self._ack_read_fd = None

        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()

        if self._reader_thread is not None:
            self._reader_thread.join()


class DetachedMeasurementPerfSession:
    EVENTS = (
        "cpu-cycles,"
        "instructions,"
        "dtlb_load_misses.walk_completed,"
        "dtlb_store_misses.walk_completed"
    )

    def __init__(self, config: MeasurementPerfConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._ctl_write_fd: Optional[int] = None
        self._ack_read_fd: Optional[int] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._last_error_line: Optional[str] = None

    def attach_to_pid(self, target_pid: int) -> None:
        self.attach_to_pids([target_pid])

    def attach_to_pids(self, target_pids: Sequence[int]) -> None:
        pid_list = [int(pid) for pid in target_pids if int(pid) > 0]
        if not pid_list:
            raise RuntimeError(f"no target pids supplied for measurement perf {self.config.label}")
        pid_arg = ",".join(str(pid) for pid in pid_list)
        self._attach(["-p", pid_arg], f"target_pids={pid_arg}")

    def attach_to_cgroup(self, cgroup_name: str) -> None:
        if not cgroup_name.strip():
            raise RuntimeError(f"empty cgroup name for measurement perf {self.config.label}")
        self._attach(["-a", "-G", cgroup_name], f"cgroup={cgroup_name}")

    def _attach(self, target_args: list[str], target_description: str) -> None:
        if self._proc is not None:
            raise RuntimeError(f"measurement perf already attached for {self.config.label}")
        if self.config.interval_ms <= 0:
            raise RuntimeError(
                f"invalid measurement interval for {self.config.label}: {self.config.interval_ms}"
            )

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.config.output_path.unlink()
        except FileNotFoundError:
            pass

        print(
            f"[measurement perf] attach {self.config.label}: "
            f"{target_description} interval_ms={self.config.interval_ms} "
            f"output={self.config.output_path}"
        )

        ctl_read_fd, ctl_write_fd = os.pipe()
        ack_read_fd, ack_write_fd = os.pipe()
        cmd = [
            self.config.perf_binary,
            "stat",
            f"--interval-print={self.config.interval_ms}",
            "--field-separator=,",
            f"--output={self.config.output_path}",
            f"--event={self.EVENTS}",
            "--delay=-1",
            f"--control=fd:{ctl_read_fd},{ack_write_fd}",
            *target_args,
        ]
        print(f"[measurement perf] spawn {self.config.label}: {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                pass_fds=(ctl_read_fd, ack_write_fd),
            )
        finally:
            os.close(ctl_read_fd)
            os.close(ack_write_fd)

        self._ctl_write_fd = ctl_write_fd
        self._ack_read_fd = ack_read_fd

        if self._proc is not None:
            print(
                f"[measurement perf] spawned {self.config.label}: "
                f"perf_pid={self._proc.pid} ctl_fd={self._ctl_write_fd} ack_fd={self._ack_read_fd}"
            )

        if self._proc.stderr is None:
            raise RuntimeError(f"no measurement perf stream available for {self.config.label}")

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stderr,),
            name=f"perf-measurement-{self.config.label}",
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self, perf_stream: TextIO) -> None:
        debug_path = f"/tmp/measurement_perf_{self.config.label}.log"
        with open(debug_path, "w") as dbg:
            for raw_line in perf_stream:
                dbg.write(raw_line)
                dbg.flush()
                line = raw_line.strip()
                lowered = line.lower()
                if "failed" in lowered or "error:" in lowered or "permission denied" in lowered:
                    self._last_error_line = line

    def _command(self, verb: str) -> None:
        if self._proc is None or self._ctl_write_fd is None or self._ack_read_fd is None:
            raise RuntimeError(f"measurement perf not attached for {self.config.label}")

        proc_rc = self._proc.poll()
        if proc_rc is not None:
            if verb == "disable" and proc_rc == 0:
                # Benign case: the attached benchmark already exited naturally, so
                # perf terminated on its own before we got a chance to send disable.
                print(
                    f"[measurement perf] {self.config.label}: perf already exited cleanly "
                    f"before {verb} rc={proc_rc} perf_pid={self._proc.pid}"
                )
                return
            raise RuntimeError(
                f"measurement perf exited before {verb} for {self.config.label} "
                f"rc={proc_rc} perf_pid={self._proc.pid}"
            )

        print(
            f"[measurement perf] {self.config.label}: send {verb} to perf_pid={self._proc.pid} "
            f"via ctl_fd={self._ctl_write_fd} awaiting ack_fd={self._ack_read_fd}"
        )
        os.write(self._ctl_write_fd, f"{verb}\n".encode())
        chunks: list[bytes] = []
        while True:
            select.select([self._ack_read_fd], [], [])
            chunk = os.read(self._ack_read_fd, 4096)
            if not chunk:
                alive_rc = self._proc.wait()
                if verb == "disable" and alive_rc == 0:
                    print(
                        f"[measurement perf] {self.config.label}: perf exited cleanly while "
                        f"waiting for {verb} ack rc={alive_rc} perf_pid={self._proc.pid}"
                    )
                    return
                note = self.last_error_line()
                if note:
                    raise RuntimeError(
                        f"failed to {verb} measurement perf for {self.config.label}: {note}"
                    )
                raise RuntimeError(
                    f"measurement perf exited while waiting for {verb} ack for "
                    f"{self.config.label} rc={alive_rc} perf_pid={self._proc.pid}"
                )
            chunks.append(chunk)
            data = b"".join(chunks)
            print(
                f"[measurement perf] {self.config.label}: raw ack chunk for {verb} "
                f"from ack_fd={self._ack_read_fd}: {chunk!r}"
            )
            if b"\n" not in data:
                continue
            line, _, _rest = data.partition(b"\n")
            print(
                f"[measurement perf] {self.config.label}: raw ack line for {verb} "
                f"from perf_pid={self._proc.pid}: {line!r}"
            )
            if line.strip() != b"ack":
                raise RuntimeError(
                    "unexpected measurement perf ack for "
                    f"{self.config.label}: {line.decode(errors='replace')!r}"
                )
            alive_rc = self._proc.poll()
            if alive_rc is None:
                print(
                    f"[measurement perf] {self.config.label}: received {verb} ack "
                    f"from perf_pid={self._proc.pid}; perf still alive"
                )
            else:
                print(
                    f"[measurement perf] {self.config.label}: received {verb} ack "
                    f"from perf_pid={self._proc.pid}; perf already exited rc={alive_rc}"
                )
            return

    def enable(self) -> None:
        self._command("enable")

    def disable(self) -> None:
        self._command("disable")

    def last_error_line(self) -> Optional[str]:
        return self._last_error_line

    def stop(self) -> None:
        if self._ctl_write_fd is not None:
            try:
                os.close(self._ctl_write_fd)
            except OSError:
                pass
            self._ctl_write_fd = None

        if self._ack_read_fd is not None:
            try:
                os.close(self._ack_read_fd)
            except OSError:
                pass
            self._ack_read_fd = None

        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()

        if self._reader_thread is not None:
            self._reader_thread.join()
