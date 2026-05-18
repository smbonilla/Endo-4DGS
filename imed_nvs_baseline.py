#!/usr/bin/env python3
"""iMED-NVS Endo-4DGS baseline entrypoint."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


IMED_CONFIG = """ModelParams = dict(
    camera_extent=10,
    use_pretrain=True
)

OptimizationParams = dict(
    coarse_iterations={coarse_iterations},
    deformation_lr_init=0.00016,
    deformation_lr_final=0.0000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.000016,
    iterations={iterations},
    percent_dense=0.01,
    render_process=True,
    densify_until_iter=3000,
    pruning_from_iter=500,
    densify_from_iter=500,
    densification_interval=100,
    pruning_interval=100,
    opacity_reset_interval=9000,
)

ModelHiddenParams = dict(
    kplanes_config={{
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 75]
    }},
    multires=[1, 2],
    defor_depth=0,
    net_width=64,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    weight_decay_iteration=0,
    bounds=1.6,
    pool_list=[2],
    multi_scale=False
)

PipelineParams = dict(
    use_depth=True,
    use_smooth=True,
    use_normal=True,
    use_confidence=True
)
"""


def run(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def sequence_name(sequence: Path) -> str:
    return sequence.resolve().name


def is_sequence_dir(path: Path) -> bool:
    return (
        (path / "pose.txt").is_file()
        and (path / "K.txt").is_file()
        and (path / "endoscope1").is_dir()
        and (path / "endoscope2").is_dir()
    )


def discover_sequences(data_root: Path) -> list[Path]:
    if is_sequence_dir(data_root):
        return [data_root]
    return sorted(path for path in data_root.rglob("*") if path.is_dir() and is_sequence_dir(path))


def write_config(output_dir: Path, iterations: int, coarse_iterations: int) -> Path:
    config_path = output_dir / "imed_runtime_config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        IMED_CONFIG.format(iterations=iterations, coarse_iterations=coarse_iterations),
        encoding="utf-8",
    )
    return config_path


def _safe_link(src: Path, dst: Path) -> None:
    """Create a symlink for immutable input data without copying large frames."""

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        raise FileExistsError(f"Refusing to overwrite existing work item: {dst}")

    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_file():
            shutil.copy2(src, dst)
            return
        raise


def prepare_writable_sequence_view(sequence: Path, output: Path) -> Path:
    """Expose a read-only iMED sequence through a writable work directory."""

    work_sequence = output / "_input_sequence"
    work_sequence.mkdir(parents=True, exist_ok=True)
    for name in ("pose.txt", "K.txt", "endoscope1", "endoscope2"):
        _safe_link(sequence / name, work_sequence / name)
    return work_sequence


def run_sequence(
    sequence: Path,
    output: Path,
    repo: Path,
    iterations: int = 1000,
    coarse_iterations: int = 300,
    port: int = 6017,
    render: bool = True,
    metrics: bool = True,
) -> Path:
    sequence = sequence.resolve()
    output = output.resolve()
    repo = repo.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = write_config(output, iterations=iterations, coarse_iterations=coarse_iterations)
    work_sequence = prepare_writable_sequence_view(sequence, output)

    train_cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(work_sequence),
        "--model_path",
        str(output),
        "--expname",
        f"imed_nvs/{sequence_name(sequence)}",
        "--port",
        str(port),
        "--configs",
        str(config),
    ]
    run(train_cmd, cwd=repo)

    if render:
        render_cmd = [
            sys.executable,
            "render.py",
            "--model_path",
            str(output),
            "--skip_train",
            "--skip_video",
            "--configs",
            str(config),
        ]
        run(render_cmd, cwd=repo)

    if metrics:
        metrics_cmd = [sys.executable, "metrics.py", "-m", str(output)]
        run(metrics_cmd, cwd=repo)

    return output / "test"


def new_view(sequence_path: str, output_dir: str, **kwargs) -> str:
    """Train on endoscope2 and render held-out endoscope1 views for one sequence."""

    repo = Path(kwargs.pop("repo", os.environ.get("ENDO4DGS_REPO", "/workspace/Endo-4DGS")))
    test_dir = run_sequence(Path(sequence_path), Path(output_dir), repo=repo, **kwargs)
    return str(test_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iMED-NVS Endo-4DGS baseline")
    parser.add_argument("--repo", default=os.environ.get("ENDO4DGS_REPO", "/workspace/Endo-4DGS"))
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("run-sequence", help="Train/render/evaluate one iMED-NVS sequence")
    one.add_argument("--sequence", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--iterations", type=int, default=1000)
    one.add_argument("--coarse-iterations", type=int, default=300)
    one.add_argument("--port", type=int, default=6017)
    one.add_argument("--no-render", action="store_true")
    one.add_argument("--no-metrics", action="store_true")

    many = sub.add_parser("run-dataset", help="Run all detected iMED-NVS sequences under a data root")
    many.add_argument("--data-root", required=True)
    many.add_argument("--output-root", required=True)
    many.add_argument("--iterations", type=int, default=1000)
    many.add_argument("--coarse-iterations", type=int, default=300)
    many.add_argument("--max-sequences", type=int, default=None)
    many.add_argument("--no-render", action="store_true")
    many.add_argument("--no-metrics", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo)

    if args.command == "run-sequence":
        run_sequence(
            Path(args.sequence),
            Path(args.output),
            repo=repo,
            iterations=args.iterations,
            coarse_iterations=args.coarse_iterations,
            port=args.port,
            render=not args.no_render,
            metrics=not args.no_metrics,
        )
        return 0

    sequences = discover_sequences(Path(args.data_root))
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(f"No iMED-NVS sequence folders found under {args.data_root}")

    for index, sequence in enumerate(sequences):
        output = Path(args.output_root) / sequence_name(sequence)
        run_sequence(
            sequence,
            output,
            repo=repo,
            iterations=args.iterations,
            coarse_iterations=args.coarse_iterations,
            port=6017 + index,
            render=not args.no_render,
            metrics=not args.no_metrics,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
