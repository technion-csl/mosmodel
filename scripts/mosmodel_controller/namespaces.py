from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .checkpoint import RUNTIME_MOUNT, WORK_MOUNT

CHECKPOINT_MARKER = 'MOSMODEL_CHECKPOINT_NAMESPACE'


def _sudo() -> list[str]:
    return [] if os.geteuid() == 0 else ['sudo']


def _bind(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['mount', '--bind', str(source), str(target)], check=True)


def in_checkpoint_namespace() -> bool:
    return os.environ.get(CHECKPOINT_MARKER) == '1'


def run_checkpoint_namespace(
    work_dir: Path,
    runtime_dir: Path,
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> int:
    if in_checkpoint_namespace():
        raise RuntimeError('already inside checkpoint namespace')

    work_dir = work_dir.resolve()
    runtime_dir = runtime_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    env_file = work_dir.parent / f'.checkpoint_env.{os.getpid()}.json'
    env_file.write_text(json.dumps(dict(environment or os.environ)))
    env_file.chmod(0o600)

    argv = [
        *_sudo(), 'unshare', '--mount', '--pid', '--fork', '--mount-proc',
        '--propagation', 'private', sys.executable, '-m',
        'scripts.mosmodel_controller.namespaces', 'checkpoint',
        '--uid', str(os.getuid()), '--gid', str(os.getgid()),
        '--groups', ','.join(map(str, os.getgroups())),
        '--work-dir', str(work_dir), '--runtime-dir', str(runtime_dir),
        '--env-file', str(env_file), '--', *command,
    ]
    try:
        return subprocess.run(argv, cwd=runtime_dir, check=False).returncode
    finally:
        env_file.unlink(missing_ok=True)


def restore_namespace_command(
    work_dir: Path,
    runtime_dir: Path,
    command: Sequence[str],
) -> list[str]:
    return [
        *_sudo(), 'unshare', '--mount', '--pid', '--fork', '--mount-proc',
        '--kill-child=SIGKILL', '--propagation', 'private', sys.executable, '-m',
        'scripts.mosmodel_controller.namespaces', 'restore',
        '--work-dir', str(work_dir.resolve()), '--runtime-dir', str(runtime_dir.resolve()),
        '--', *command,
    ]


def _checkpoint(args: argparse.Namespace) -> int:
    if os.geteuid() != 0 or os.getpid() != 1:
        raise RuntimeError('checkpoint namespace helper must run as root PID 1')

    work = Path(args.work_dir).resolve()
    runtime = Path(args.runtime_dir).resolve()
    if not work.is_dir() or not runtime.is_dir():
        raise FileNotFoundError('checkpoint work/runtime directory is missing')

    environment = json.loads(Path(args.env_file).read_text())
    Path(args.env_file).unlink(missing_ok=True)
    _bind(work, WORK_MOUNT)
    _bind(runtime, RUNTIME_MOUNT)

    os.setgroups([int(value) for value in args.groups.split(',') if value])
    os.setgid(args.gid)
    os.setuid(args.uid)
    environment[CHECKPOINT_MARKER] = '1'
    os.chdir(RUNTIME_MOUNT)
    os.execvpe(args.command[0], args.command, environment)
    return 127


def _restore(args: argparse.Namespace) -> int:
    if os.geteuid() != 0 or os.getpid() != 1:
        raise RuntimeError('restore namespace helper must run as root PID 1')

    work = Path(args.work_dir).resolve()
    runtime = Path(args.runtime_dir).resolve()
    if not work.is_dir() or not runtime.is_dir():
        raise FileNotFoundError('restore work/runtime directory is missing')

    _bind(work, WORK_MOUNT)
    _bind(runtime, RUNTIME_MOUNT)
    os.chdir(WORK_MOUNT)

    command_pid = os.fork()
    if command_pid == 0:
        os.execvp(args.command[0], args.command)
        os._exit(127)

    command_status = None
    while True:
        try:
            child, status = os.wait()
        except ChildProcessError:
            return 0 if command_status is None else os.waitstatus_to_exitcode(command_status)
        if child == command_pid:
            command_status = status
            rc = os.waitstatus_to_exitcode(status)
            if rc != 0:
                return rc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    checkpoint = sub.add_parser('checkpoint')
    checkpoint.add_argument('--uid', type=int, required=True)
    checkpoint.add_argument('--gid', type=int, required=True)
    checkpoint.add_argument('--groups', default='')
    checkpoint.add_argument('--work-dir', required=True)
    checkpoint.add_argument('--runtime-dir', required=True)
    checkpoint.add_argument('--env-file', required=True)
    checkpoint.add_argument('command', nargs=argparse.REMAINDER)

    restore = sub.add_parser('restore')
    restore.add_argument('--work-dir', required=True)
    restore.add_argument('--runtime-dir', required=True)
    restore.add_argument('command', nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit('missing command after --')
    return _checkpoint(args) if args.mode == 'checkpoint' else _restore(args)


if __name__ == '__main__':
    raise SystemExit(main())
