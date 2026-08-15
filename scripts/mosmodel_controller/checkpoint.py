from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

WORK_MOUNT = Path('/work')
RUNTIME_MOUNT = Path('/runtime')
LAYOUT_PATH = WORK_MOUNT / 'layout.csv'
BENCHMARK_LOG_PATH = WORK_MOUNT / 'benchmark.log'
PERF_PROGRESS_PATH = WORK_MOUNT / 'perf_progress.out'
TREE_EXEC_PATH = RUNTIME_MOUNT / 'scripts' / 'mosmodel_controller' / 'tree_exec.py'

IMAGES_DIR = 'images'
WORK_SNAPSHOT_DIR = 'work.snapshot'
RUNTIME_SNAPSHOT_DIR = 'runtime.snapshot'
MOSALLOC_LIBRARY_RELATIVE = Path('mosalloc/build/src/libmosalloc.so')
RESTORE_WORK_DIR = 'work'
METADATA_FILE = 'metadata.json'
DONE_FILE = 'checkpoint.done'


def runtime_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sudo() -> list[str]:
    return [] if os.geteuid() == 0 else ['sudo']


def run_root(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_sudo(), *command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def checkpoint_is_complete(path: Path) -> bool:
    return (
        (path / DONE_FILE).is_file()
        and (path / METADATA_FILE).is_file()
        and (path / IMAGES_DIR).is_dir()
        and (path / WORK_SNAPSHOT_DIR).is_dir()
    )


def snapshot_work(destination: Path) -> None:
    temporary = destination.with_name(f'{destination.name}.tmp.{os.getpid()}')
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(WORK_MOUNT, temporary, symlinks=True)
    shutil.rmtree(destination, ignore_errors=True)
    temporary.replace(destination)


def write_metadata(
    checkpoint_dir: Path,
    benchmark: Path,
    layout: Optional[Path],
    i_start: int,
    observed: int,
    num_threads: int,
    runtime_artifacts: list[str] | None = None,
    tcp_established: bool = False,
) -> None:
    metadata = {
        'schema_version': 3,
        'benchmark': str(benchmark.resolve()),
        'layout': None if layout is None else layout.name,
        'i_start': i_start,
        'observed_instructions': observed,
        'num_threads': num_threads,
        'work_path': str(WORK_MOUNT),
        'runtime_path': str(RUNTIME_MOUNT),
        'benchmark_log': str(BENCHMARK_LOG_PATH),
        'runtime_artifacts': runtime_artifacts or [],
        'tcp_established': tcp_established,
    }
    path = checkpoint_dir / METADATA_FILE
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def mark_complete(checkpoint_dir: Path) -> None:
    (checkpoint_dir / DONE_FILE).write_text('ok\n')


def reset_restore_work(archive_dir: Path, restore_dir: Path) -> None:
    archive_dir = archive_dir.resolve()
    restore_dir = restore_dir.resolve()
    if archive_dir == restore_dir:
        raise RuntimeError('checkpoint archive and restore workspace must differ')
    if not checkpoint_is_complete(archive_dir):
        raise FileNotFoundError(f'incomplete checkpoint archive: {archive_dir}')

    source = archive_dir / WORK_SNAPSHOT_DIR
    destination = restore_dir / RESTORE_WORK_DIR
    restore_dir.mkdir(parents=True, exist_ok=True)
    result = run_root(['rm', '-rf', str(destination)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    result = run_root(['cp', '-a', str(source), str(destination)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def virtualize_command(command: str, runtime_dir: Path, use_layout: bool) -> str:
    if not command.strip():
        return ''

    runtime_dir = runtime_dir.resolve()
    argv = shlex.split(command)
    rewritten: list[str] = []
    replace_layout = False

    def rewrite_path(value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            return str(RUNTIME_MOUNT / path.relative_to(runtime_dir))
        except ValueError:
            return value

    for token in argv:
        if replace_layout:
            rewritten.append(str(LAYOUT_PATH))
            replace_layout = False
            continue
        if '=' in token and not token.startswith('='):
            name, value = token.split('=', 1)
            mapped = rewrite_path(value)
            token = f'{name}={mapped}' if mapped != value else token
        else:
            token = rewrite_path(token)
        rewritten.append(token)
        if use_layout and token == '-cpf':
            replace_layout = True

    if replace_layout:
        raise ValueError('-cpf is missing its layout argument')
    return shlex.join(rewritten)


def read_metadata(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / METADATA_FILE).read_text())


def mosalloc_library_from_submit(submit: str) -> Optional[Path]:
    argv = shlex.split(submit)
    if not any(Path(token).name == 'runMosalloc.py' for token in argv):
        return None
    try:
        index = argv.index('--library')
        return Path(argv[index + 1])
    except (ValueError, IndexError):
        raise ValueError('runMosalloc.py submit command is missing --library')


def snapshot_runtime_artifact(checkpoint_dir: Path, source: Path) -> str:
    try:
        relative = source.relative_to(RUNTIME_MOUNT)
    except ValueError as error:
        raise ValueError(f'runtime artifact is outside {RUNTIME_MOUNT}: {source}') from error
    destination = checkpoint_dir / RUNTIME_SNAPSHOT_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(relative)


def memory_node(prefix: str) -> int:
    argv = shlex.split(prefix)
    for index, token in enumerate(argv[:-1]):
        if Path(token).name == 'setCpuMemoryAffinity.sh':
            return int(argv[index + 1])
    raise ValueError('expected setCpuMemoryAffinity.sh <node> in --prefix')


def hugepage_owner(checkpoint_dir: Path) -> str:
    digest = hashlib.sha256(str(checkpoint_dir).encode()).hexdigest()[:16]
    return f'criu_checkpoint_{digest}'


def release_hugepages(submit: str, prefix: str, owner: str) -> None:
    script: Optional[Path] = None
    for token in shlex.split(submit):
        if Path(token).name == 'runMosalloc.py':
            candidate = Path(token).resolve().parent / 'reserveHugePages.sh'
            if candidate.is_file():
                script = candidate
                break
    if script is None:
        raise RuntimeError('could not find reserveHugePages.sh from --submit')

    env = os.environ.copy()
    env['MOSALLOC_KEEP_HUGEPAGE_POOL'] = '1'
    subprocess.run(
        [str(script), 'release', f'--owner={owner}', f'--node={memory_node(prefix)}'],
        env=env,
        check=True,
    )


def read_progress(path: Path) -> int:
    try:
        lines = path.read_text(errors='replace').splitlines()
    except FileNotFoundError:
        return 0

    total = 0
    for line in lines:
        fields = [field.strip() for field in line.split(';')]
        event = next((i for i, field in enumerate(fields) if field.startswith('instructions')), None)
        if event is None:
            continue
        for field in reversed(fields[:event]):
            try:
                total += int(round(float(field.replace(',', ''))))
                break
            except ValueError:
                pass
    return total


def dump(
    root_pid: int, images_dir: Path, dump_log: Path, *, tcp_established: bool = False,
) -> None:
    images_dir.mkdir(parents=True)
    command = [*_sudo(), 'criu', 'dump', '-t', str(root_pid), '-D', str(images_dir)]
    if tcp_established:
        command.append('--tcp-established')
    command.extend(['-v4', '-o', str(dump_log)])
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f'CRIU dump failed rc={result.returncode}: {message}; see {dump_log}'
        )
