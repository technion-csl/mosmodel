#! /usr/bin/env python3
from __future__ import annotations

from .cli import find_benchmarks_root, parse_args
from .pair_controller import PairController, build_pair_runs


def main() -> int:
    args = parse_args()
    benchmarks_root = find_benchmarks_root()
    runs = build_pair_runs(args, benchmarks_root)

    controller = PairController(args, benchmarks_root, runs)
    controller.install_signal_handlers()
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
