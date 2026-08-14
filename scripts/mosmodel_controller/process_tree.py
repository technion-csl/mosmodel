from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

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



def children(pid: int) -> list[int]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    text = path.read_text(encoding="utf-8").strip()
    return [int(value) for value in text.split()] if text else []


def single_child(pid: int) -> int:
    child_pids = children(pid)
    if len(child_pids) != 1:
        raise RuntimeError(f"expected one child of pid {pid}, found {child_pids}")
    return child_pids[0]


def deepest_single_child(pid: int) -> int:
    current = pid
    while True:
        child_pids = children(current)
        if not child_pids:
            return current
        if len(child_pids) != 1:
            raise RuntimeError(
                f"expected one child of pid {current}, found {child_pids}"
            )
        current = child_pids[0]


def wait_until_stopped(pid: int) -> None:
    stat_path = Path("/proc") / str(pid) / "stat"
    while True:
        text = stat_path.read_text(encoding="utf-8")
        state = text[text.rfind(")") + 2 :].split()[0]
        if state in {"T", "t", "Z", "X", "x"}:
            return
        time.sleep(0.001)


def read_process_group(pid: int) -> tuple[int, int]:
    text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[2]), int(fields[3])


def namespace_pids(pid: int) -> list[int]:
    """Return the NSpid chain for a host-visible process."""
    status_path = Path("/proc") / str(pid) / "status"
    text = status_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("NSpid:"):
            return [int(value) for value in line.split(":", 1)[1].split()]
    return []


def host_pid_for_namespace_pid(
    wrapper_pid: int,
    namespace_pid: int,
) -> Optional[int]:
    """Map one restore-namespace PID to its host PID.

    CRIU writes --pidfile from inside the private restore PID namespace, while
    the outer controller performs /proc, cgroup, signal, and perf operations in
    the host PID namespace.  Restricting the lookup to descendants of this
    restore wrapper prevents ambiguity between concurrent restores that reuse
    the same namespace-local PIDs.
    """
    matches: list[int] = []
    for pid in process_tree_pids(wrapper_pid):
        if pid == wrapper_pid:
            continue
        try:
            pid_chain = namespace_pids(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue

        if len(pid_chain) >= 2 and pid_chain[-1] == namespace_pid:
            matches.append(pid)

    if len(matches) > 1:
        raise RuntimeError(
            "multiple descendants match restored namespace PID "
            f"{namespace_pid}: {matches}"
        )
    return matches[0] if matches else None
