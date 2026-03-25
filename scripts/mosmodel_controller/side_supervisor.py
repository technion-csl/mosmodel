from __future__ import annotations

import subprocess
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
            "usage: side_supervisor.py [--release-fd <fd>] <pgid-file> -- <command> [args...]"
        )

    pgid_file = Path(args[0])
    cmd = args[2:]
    return release_fd, pgid_file, cmd


def main() -> int:
    release_fd, pgid_file, cmd = _parse_args(sys.argv[1:])

    helper_script = Path(__file__).with_name("bench_group_exec.py").resolve()
    child_cmd = [sys.executable, str(helper_script)]
    if release_fd is not None:
        child_cmd.extend(["--release-fd", str(release_fd)])
    child_cmd.extend([str(pgid_file), "--", *cmd])

    pass_fds: tuple[int, ...] = () if release_fd is None else (release_fd,)
    proc = subprocess.Popen(child_cmd, pass_fds=pass_fds)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
