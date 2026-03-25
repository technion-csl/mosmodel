from __future__ import annotations

import os
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> tuple[int | None, Path, list[str]]:
    release_fd: int | None = None
    args = list(argv)

    if len(args) >= 2 and args[0] == "--release-fd":
        try:
            release_fd = int(args[1])
        except ValueError as exc:
            raise SystemExit(f"invalid --release-fd value: {args[1]!r}") from exc
        args = args[2:]

    if len(args) < 3 or args[1] != "--":
        raise SystemExit(
            "usage: bench_group_exec.py [--release-fd <fd>] <pgid-file> -- <command> [args...]"
        )

    pgid_file = Path(args[0])
    cmd = args[2:]
    return release_fd, pgid_file, cmd


def _wait_for_release_token(release_fd: int) -> None:
    try:
        token = os.read(release_fd, 1)
    finally:
        os.close(release_fd)

    if not token:
        print(
            "bench_group_exec.py: release pipe closed before benchmark was released",
            file=sys.stderr,
        )
        raise SystemExit(125)


def main() -> int:
    release_fd, pgid_file, cmd = _parse_args(sys.argv[1:])

    # Create a new process group inside the existing side session so STOP/CONT can
    # target only the benchmark subtree while the detached sampler stays outside it.
    try:
        os.setpgid(0, 0)
    except OSError as exc:
        print(f"bench_group_exec.py: os.setpgid(0, 0) failed: {exc}", file=sys.stderr)
        raise SystemExit(125)

    leader_pid = os.getpid()
    try:
        pgid_file.write_text(f"{leader_pid}\n")
    except OSError as exc:
        print(f"bench_group_exec.py: failed to write {pgid_file}: {exc}", file=sys.stderr)
        raise SystemExit(125)

    if release_fd is not None:
        _wait_for_release_token(release_fd)

    os.execvp(cmd[0], cmd)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
