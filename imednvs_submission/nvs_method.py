"""Minimal iMED NVS rendering template.

Replace `render_target_views` with your NVS training/optimization and rendering
pipeline. The starter intentionally does not read target-view `endoscope1` RGB
images during runtime; it uses source-view `endoscope2` frames only to
determine how many target-view renders to write for local format checks.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def _source_frames(sequence_dir: Path) -> list[Path]:
    source_dir = sequence_dir / "endoscope2"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source-view directory: {source_dir}")

    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    preferred_left_dir = source_dir / "L"
    search_root = preferred_left_dir if preferred_left_dir.is_dir() else source_dir
    frames = sorted(
        path
        for path in search_root.rglob("*")
        if path.suffix.lower() in image_suffixes
    )
    if not frames:
        raise FileNotFoundError(f"No source-view frames found in: {source_dir}")
    return frames


def render_target_views(sequence_dir: str | Path, output_dir: str | Path) -> None:
    """Optimize if needed and render target endoscope1-view RGB images.

    Args:
        sequence_dir: Input sequence directory. The method may read source-view
            `endoscope2` frames and metadata such as `K.txt` and `pose.txt`.
        output_dir: Writable output directory for this sequence. Rendered
            target-view PNG files must be written to `output_dir/renders`.
    """

    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    for index, src_frame in enumerate(_source_frames(sequence_dir)):
        image = Image.open(src_frame).convert("RGB")
        image.save(render_dir / f"{index:05d}.png")
