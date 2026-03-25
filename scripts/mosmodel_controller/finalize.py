from __future__ import annotations

import shutil
from pathlib import Path

from ..benchmarkCore import BenchmarkRun



def finalize_run(
    run: BenchmarkRun,
    *,
    do_postrun: bool,
    clean_threshold: int,
    exclude_files: list[str],
) -> None:
    if do_postrun:
        print(f"{run._benchmark_dir}: post-running")
        run.postrun()

    run.move_files_to_output_dir()
    run.clean_output_dir(clean_threshold, exclude_files)



def remove_dir_if_exists(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
