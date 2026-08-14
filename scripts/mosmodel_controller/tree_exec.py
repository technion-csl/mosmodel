from __future__ import annotations

import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != '--':
        raise SystemExit('usage: tree_exec.py -- <command> [args...]')

    os.setsid()
    if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
        raise RuntimeError('checkpoint tree failed to create its own session')

    fd = os.open('/dev/null', os.O_RDONLY)
    os.dup2(fd, 0)
    if fd > 2:
        os.close(fd)

    for name in os.listdir('/proc/self/fd'):
        try:
            open_fd = int(name)
        except ValueError:
            continue
        if open_fd <= 2:
            continue
        try:
            os.close(open_fd)
        except OSError:
            pass

    command = argv[1:]
    os.execvp(command[0], command)
    return 127


if __name__ == '__main__':
    raise SystemExit(main())
