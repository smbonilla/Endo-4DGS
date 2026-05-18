#!/usr/bin/env python3
"""Lightweight local output checker for iMED NVS submissions."""

from __future__ import annotations

import sys
from pathlib import Path


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


def source_frame_names(sequence: Path) -> list[str]:
    source_dir = sequence / "endoscope2"
    preferred_left_dir = source_dir / "L"
    search_root = preferred_left_dir if preferred_left_dir.is_dir() else source_dir
    frames = sorted(
        path
        for path in search_root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    return [f"{index:05d}.png" for index, _ in enumerate(frames)]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: check_outputs.py INPUT_ROOT OUTPUT_ROOT")

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    failures: list[str] = []

    for sequence in discover_sequences(input_root):
        candidate_render_dirs = [
            output_root / sequence.name / "renders",
            output_root / "input" / "renders",
            output_root / "renders",
        ]
        render_dir = next((path for path in candidate_render_dirs if path.is_dir()), candidate_render_dirs[0])
        expected = source_frame_names(sequence)
        produced = sorted(path.name for path in render_dir.glob("*.png")) if render_dir.is_dir() else []
        missing = sorted(set(expected) - set(produced))
        if missing:
            failures.append(f"{sequence.name}: missing {len(missing)} render(s), first missing: {missing[0]}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1

    print("[OK] NVS output structure check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
