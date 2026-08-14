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


def _error_line(line: str) -> bool:
    lowered = line.lower()
    return 'failed' in lowered or 'error:' in lowered or 'permission denied' in lowered


def _close_fd(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _control(
    proc: subprocess.Popen,
    ctl_fd: int,
    ack_fd: int,
    verb: str,
    label: str,
    last_error: Optional[str],
) -> None:
    rc = proc.poll()
    if rc is not None:
        if verb == 'disable' and rc == 0:
            return
        raise RuntimeError(f'perf exited before {verb} for {label}: rc={rc}')

    os.write(ctl_fd, f'{verb}\n'.encode())
    data = b''
    while b'\n' not in data:
        select.select([ack_fd], [], [])
        chunk = os.read(ack_fd, 4096)
        if not chunk:
            rc = proc.wait()
            if verb == 'disable' and rc == 0:
                return
            detail = f': {last_error}' if last_error else ''
            raise RuntimeError(f'perf exited waiting for {verb} ack for {label}: rc={rc}{detail}')
        data += chunk

    if data.partition(b'\n')[0].strip() != b'ack':
        raise RuntimeError(f'unexpected perf {verb} ack for {label}: {data!r}')


def _stop(proc: Optional[subprocess.Popen], thread: Optional[threading.Thread]) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait()
    if thread is not None:
        thread.join()


class WrappedPerfInstructionsMonitor:
    def __init__(self, config: LiveProgressConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._total = 0
        self._last_error: Optional[str] = None
        self._ctl_fd: Optional[int] = None
        self._ack_fd: Optional[int] = None

    def attach_to_pid(self, pid: int) -> None:
        self._attach(['-p', str(pid)])

    def attach_to_cgroup(self, name: str) -> None:
        if not name.strip():
            raise RuntimeError(f'empty cgroup for {self.config.label}')
        self._attach(['-a', '-G', name])

    def _attach(self, target: list[str]) -> None:
        if self._proc is not None:
            raise RuntimeError(f'progress perf already attached for {self.config.label}')
        if self.config.interval_ms <= 0:
            raise ValueError('progress interval must be positive')

        ctl_read, ctl_write = os.pipe()
        ack_read, ack_write = os.pipe()
        cmd = [
            self.config.perf_binary, 'stat', '-e', 'instructions', '-I',
            str(self.config.interval_ms), '-x', ';', '--no-big-num', '--delay=-1',
            f'--control=fd:{ctl_read},{ack_write}', *target,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                pass_fds=(ctl_read, ack_write),
            )
        finally:
            os.close(ctl_read)
            os.close(ack_write)

        self._ctl_fd, self._ack_fd = ctl_write, ack_read
        if self._proc.stderr is None:
            raise RuntimeError(f'no perf stream for {self.config.label}')
        self._thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stderr,),
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self, stream: TextIO) -> None:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            delta = self._parse_delta(line)
            if delta is not None:
                with self._lock:
                    self._total += delta
            elif _error_line(line):
                self._last_error = line

    @staticmethod
    def _parse_delta(line: str) -> Optional[int]:
        fields = [field.strip() for field in line.split(';')]
        event = next((i for i, value in enumerate(fields) if value.startswith('instructions')), None)
        if event is None:
            return None
        for value in reversed(fields[:event]):
            if value.lower() in {'<not counted>', '<not supported>', 'nan'}:
                return None
            try:
                parsed = float(value.replace(',', ''))
                return int(round(parsed)) if parsed >= 0 else None
            except ValueError:
                pass
        return None

    def enable(self) -> None:
        if self._proc is None or self._ctl_fd is None or self._ack_fd is None:
            raise RuntimeError(f'progress perf not attached for {self.config.label}')
        _control(self._proc, self._ctl_fd, self._ack_fd, 'enable', self.config.label, self._last_error)

    def total_instructions(self) -> int:
        with self._lock:
            return self._total

    def last_error_line(self) -> Optional[str]:
        return self._last_error

    def stop(self) -> None:
        _close_fd(self._ctl_fd)
        _close_fd(self._ack_fd)
        self._ctl_fd = self._ack_fd = None
        _stop(self._proc, self._thread)


class DetachedMeasurementPerfSession:
    EVENTS = 'cpu-cycles,instructions,dtlb_load_misses.walk_completed,dtlb_store_misses.walk_completed'

    def __init__(self, config: MeasurementPerfConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._ctl_fd: Optional[int] = None
        self._ack_fd: Optional[int] = None

    def attach_to_pid(self, pid: int) -> None:
        self.attach_to_pids([pid])

    def attach_to_pids(self, pids: Sequence[int]) -> None:
        targets = [str(int(pid)) for pid in pids if int(pid) > 0]
        if not targets:
            raise RuntimeError(f'no measurement pids for {self.config.label}')
        self._attach(['-p', ','.join(targets)])

    def attach_to_cgroup(self, name: str) -> None:
        if not name.strip():
            raise RuntimeError(f'empty measurement cgroup for {self.config.label}')
        self._attach(['-a', '-G', name])

    def _attach(self, target: list[str]) -> None:
        if self._proc is not None:
            raise RuntimeError(f'measurement perf already attached for {self.config.label}')
        if self.config.interval_ms <= 0:
            raise ValueError('measurement interval must be positive')

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.unlink(missing_ok=True)
        ctl_read, ctl_write = os.pipe()
        ack_read, ack_write = os.pipe()
        cmd = [
            self.config.perf_binary, 'stat',
            f'--interval-print={self.config.interval_ms}', '--field-separator=,',
            f'--output={self.config.output_path}', f'--event={self.EVENTS}', '--delay=-1',
            f'--control=fd:{ctl_read},{ack_write}', *target,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                pass_fds=(ctl_read, ack_write),
            )
        finally:
            os.close(ctl_read)
            os.close(ack_write)

        self._ctl_fd, self._ack_fd = ctl_write, ack_read
        if self._proc.stderr is None:
            raise RuntimeError(f'no measurement perf stream for {self.config.label}')
        self._thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stderr,),
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self, stream: TextIO) -> None:
        for raw in stream:
            line = raw.strip()
            if _error_line(line):
                self._last_error = line

    def _command(self, verb: str) -> None:
        if self._proc is None or self._ctl_fd is None or self._ack_fd is None:
            raise RuntimeError(f'measurement perf not attached for {self.config.label}')
        _control(self._proc, self._ctl_fd, self._ack_fd, verb, self.config.label, self._last_error)

    def enable(self) -> None:
        self._command('enable')

    def disable(self) -> None:
        self._command('disable')

    def last_error_line(self) -> Optional[str]:
        return self._last_error

    def stop(self) -> None:
        _close_fd(self._ctl_fd)
        _close_fd(self._ack_fd)
        self._ctl_fd = self._ack_fd = None
        _stop(self._proc, self._thread)
