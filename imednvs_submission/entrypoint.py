#!/usr/bin/env python3
"""Container entrypoint for iMED NVS submissions."""

from __future__ import annotations

import argparse
from pathlib import Path

from nvs_method import render_target_views


def is_sequence_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "endoscope2").is_dir()
        and (path / "K.txt").is_file()
        and (path / "pose.txt").is_file()
    )


def discover_sequences(input_root: Path) -> list[Path]:
    if is_sequence_dir(input_root):
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if is_sequence_dir(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iMED NVS submission entrypoint")
    parser.add_argument("--input", default="/input", help="Mounted iMED NVS input root")
    parser.add_argument("--output", default="/output", help="Mounted prediction output root")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional local-test limit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(input_root)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(f"No iMED NVS sequence directories found under {input_root}")

    for sequence in sequences:
        render_target_views(sequence, output_root / sequence.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
