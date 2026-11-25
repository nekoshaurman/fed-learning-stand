from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fl_benchmark.config import load_config
from fl_benchmark.task import run_local_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        out = run_local_simulation(
            config,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            synthetic_smoke=args.synthetic_smoke,
        )
    except Exception as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1
    print(f"Experiment completed: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
