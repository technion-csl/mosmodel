from __future__ import annotations

from pathlib import Path

from .process_tree import process_tree_pids

TCP_ESTABLISHED_STATE = '01'


def _socket_inode(path: Path) -> int | None:
    try:
        target = path.readlink()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None

    text = str(target)
    prefix = 'socket:['
    if not text.startswith(prefix) or not text.endswith(']'):
        return None
    try:
        return int(text[len(prefix):-1])
    except ValueError:
        return None


def _process_tree_socket_inodes(root_pid: int) -> set[int]:
    inodes: set[int] = set()
    for pid in process_tree_pids(root_pid):
        fd_dir = Path('/proc') / str(pid) / 'fd'
        try:
            fds = list(fd_dir.iterdir())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for fd in fds:
            inode = _socket_inode(fd)
            if inode is not None:
                inodes.add(inode)
    return inodes


def _established_tcp_inodes(pid: int) -> set[int]:
    inodes: set[int] = set()
    for table_name in ('tcp', 'tcp6'):
        path = Path('/proc') / str(pid) / 'net' / table_name
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue

        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != TCP_ESTABLISHED_STATE:
                continue
            try:
                inodes.add(int(fields[9]))
            except ValueError:
                continue
    return inodes


def tree_has_established_tcp(root_pid: int) -> bool:
    """Return whether this process tree owns an ESTABLISHED TCP/TCP6 socket.

    /proc/<pid>/net/tcp* describes the network namespace rather than only one
    process, so intersect it with socket inodes actually referenced by the
    target process tree before deciding that CRIU needs --tcp-established.
    """
    socket_inodes = _process_tree_socket_inodes(root_pid)
    if not socket_inodes:
        return False
    return bool(socket_inodes & _established_tcp_inodes(root_pid))
