from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _privileged_prefix() -> list[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    return ["sudo"]


def _run_privileged_checked(command: list[str]) -> None:
    result = subprocess.run(
        [*_privileged_prefix(), *command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def _write_control(path: Path, value: str) -> None:
    data = value if value.endswith("\n") else value + "\n"
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        path.write_text(data, encoding="utf-8")
        return

    result = subprocess.run(
        ["sudo", "tee", str(path)],
        input=data,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to write {path}")


def _find_cgroup2_mount() -> Path:
    mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        after_fields = after.split()
        if not after_fields or after_fields[0] != "cgroup2":
            continue
        before_fields = before.split()
        if len(before_fields) < 5:
            continue
        return Path(before_fields[4]).resolve()
    raise RuntimeError("cgroup v2 filesystem is not mounted")


def _pid_cgroup_relative_path(pid: int) -> Path:
    text = (Path("/proc") / str(pid) / "cgroup").read_text(encoding="utf-8")
    for line in text.splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            return Path(relative.lstrip("/"))
    raise RuntimeError(f"pid {pid} is not attached to a cgroup v2 hierarchy")


def process_tree_pids(root_pid: int) -> list[int]:
    """Return root_pid and all currently visible descendants."""
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to inspect restored process tree")

    by_parent: dict[int, list[int]] = {}
    visible: set[int] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        visible.add(pid)
        by_parent.setdefault(ppid, []).append(pid)

    root_pid = int(root_pid)
    if root_pid not in visible:
        return []

    tree: list[int] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        tree.append(pid)
        stack.extend(by_parent.get(pid, []))
    return tree



@dataclass
class CgroupV2:
    mount_point: Path
    path: Path

    @classmethod
    def create_for_pid(cls, pid: int, label: str) -> "CgroupV2":
        mount_point = _find_cgroup2_mount()
        parent_relative = _pid_cgroup_relative_path(pid)
        parent = mount_point / parent_relative

        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
        name = f"mosmodel-criu-{os.getuid()}-{os.getpid()}-{digest}"
        path = parent / name
        _run_privileged_checked(["mkdir", str(path)])
        return cls(mount_point=mount_point, path=path)

    @property
    def perf_name(self) -> str:
        return self.path.relative_to(self.mount_point).as_posix()

    def add_pid(self, pid: int) -> None:
        _write_control(self.path / "cgroup.procs", str(int(pid)))

    def add_pids(self, pids: Iterable[int]) -> None:
        moved: list[int] = []
        for pid in sorted({int(pid) for pid in pids if int(pid) > 0}):
            if not (Path("/proc") / str(pid)).exists():
                continue
            self.add_pid(pid)
            moved.append(pid)
        if not moved:
            raise RuntimeError(f"no restored processes were moved into {self.path}")

    def pids(self) -> list[int]:
        try:
            text = (self.path / "cgroup.procs").read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return sorted(int(line) for line in text.splitlines() if line.strip())

    def _events(self) -> dict[str, int]:
        try:
            text = (self.path / "cgroup.events").read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        result: dict[str, int] = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                continue
        return result

    def is_populated(self) -> bool:
        events = self._events()
        if "populated" in events:
            return events["populated"] == 1
        return bool(self.pids())

    def freeze(self) -> None:
        freeze_path = self.path / "cgroup.freeze"
        if not freeze_path.exists():
            raise RuntimeError(f"cgroup freezer is unavailable: {freeze_path}")
        _write_control(freeze_path, "1")

        while True:
            events = self._events()
            if events.get("frozen") == 1 or not self.is_populated():
                return
            time.sleep(0.005)

    def unfreeze(self) -> None:
        freeze_path = self.path / "cgroup.freeze"
        if freeze_path.exists():
            _write_control(freeze_path, "0")

    def kill(self) -> None:
        if not self.path.exists():
            return

        kill_path = self.path / "cgroup.kill"
        if kill_path.exists():
            _write_control(kill_path, "1")
        else:
            # Older cgroup v2 kernels may not expose cgroup.kill.
            self.unfreeze()
            for pid in self.pids():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    _run_privileged_checked(["kill", "-KILL", str(pid)])

        while self.is_populated():
            time.sleep(0.01)

    def remove(self) -> None:
        if not self.path.exists():
            return
        if self.is_populated():
            raise RuntimeError(f"cannot remove populated cgroup {self.path}")
        self.unfreeze()
        _run_privileged_checked(["rmdir", str(self.path)])

    def kill_and_remove(self) -> None:
        if not self.path.exists():
            return
        try:
            self.kill()
        finally:
            if self.path.exists() and not self.is_populated():
                self.remove()
