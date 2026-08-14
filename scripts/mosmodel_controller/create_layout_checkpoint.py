#!/usr/bin/env python3
"""Backward-compatible CLI for the canonical portable checkpoint creator.

Historically this module contained a second checkpoint implementation with a
legacy archive layout. It now only translates the old CLI to
`create_checkpoint.py`, so there is one implementation and one archive format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .checkpoint import runtime_root
from .create_checkpoint import main as create_checkpoint_main


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a layout checkpoint using the canonical portable creator."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--i-start", type=int, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--submit", default="")
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional disposable build directory. If omitted, a hidden sibling "
            "of --checkpoint-dir is used for compatibility with old callers."
        ),
    )
    parser.add_argument("--runtime-dir", default=str(runtime_root()))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else checkpoint_dir.parent / f".{checkpoint_dir.name}.checkpoint_build"
    )

    forwarded = [
        "--benchmark",
        args.benchmark,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--run-dir",
        str(run_dir),
        "--runtime-dir",
        str(Path(args.runtime_dir).resolve()),
        "--i-start",
        str(args.i_start),
        "--num-threads",
        str(args.num_threads),
    ]
    if args.layout:
        forwarded.extend(["--layout", args.layout])
    if args.prefix:
        forwarded.extend(["--prefix", args.prefix])
    if args.submit:
        forwarded.extend(["--submit", args.submit])
    if args.force:
        forwarded.append("--force")

    return create_checkpoint_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
