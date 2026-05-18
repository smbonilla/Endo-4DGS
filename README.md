# Endo-4DGS for iMED Static Two-Camera Evaluation

This repository supports the iMED 2026 challenge subtask on deformable novel view synthesis, part of EndoVis 2026 at MICCAI 2026 (Strasbourg, France).

[[Challenge Website](https://imed-challenge.github.io/)] [[Participate](https://www.synapse.org/Synapse:syn74277461/wiki/639538)] [[Parent Challenge Hub](https://opencas.dkfz.de/endovis/challenges/2026/)]

<p align="center">
  <img src="assets/comparison.gif" width="900" alt="Ground truth (left), trained endoscope render (middle), and evaluated endoscope render (right) on session_004_scene_2_tool_1"/>
</p>

## Huge Thanks to the Original Repository

This project is a fork of the original Endo-4DGS repository by Huang et al.

- Original codebase: [lastbasket/Endo-4DGS](https://github.com/lastbasket/Endo-4DGS)
- This fork: [smbonilla/Endo-4DGS](https://github.com/smbonilla/Endo-4DGS)

Please cite the original Endo-4DGS work:

```bibtex
@inproceedings{huang2024endo,
  title={Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting},
  author={Huang, Yiming and Cui, Beilei and Bai, Long and Guo, Ziqi and Xu, Mengya and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={197--207},
  year={2024},
  organization={Springer}
}
```

## What This Fork Adds

This fork adds iMED dataset support for a static two-camera protocol:

- Training view: `endoscope2/L`
- Test view: `endoscope1/L`
- Depths: `depthL` (mm, used directly)
- Masks: `toolL`, loaded and converted as `mask = 1 - raw_mask/255`
- Poses: `pose.txt` with exactly 2 rows (`k tx ty tz qx qy qz qw`)
- Intrinsics: `K.txt`

Expected dataset layout:

```text
./data/imed/session_004_scene_2_tool_1
├── K.txt
├── pose.txt
├── endoscope1
│   ├── L
│   ├── depthL
│   └── toolL
└── endoscope2
    ├── L
    ├── depthL
    └── toolL
```

## Method: Train on One Camera, Test on the Other

For each static iMED session:

1. Train Gaussian splats only on `endoscope2` images.
2. Keep both cameras static and use the provided two-camera pose relation.
3. Render novel views from the held-out `endoscope1` camera.
4. Evaluate rendered test images against `endoscope1` ground truth.

This setup measures cross-camera generalization rather than interpolation within one camera stream.

## Metrics and Timing

- **PSNR / SSIM:** computed in `metrics.py`.
- **LPIPS:** also computed in `metrics.py`.
- **Inference speed:** printed as `FPS` in `render.py` during rendering.

For iMED two-camera evaluation, PSNR/SSIM are computed on the valid reprojection region from `endoscope2` into `endoscope1` (single global mask per sequence).

## Setup

```bash
git clone https://github.com/smbonilla/Endo-4DGS.git
cd Endo-4DGS
git submodule update --init --recursive
conda create -n ED4DGS python=3.8
conda activate ED4DGS
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics
```

## Run iMED Training

```bash
sh train_imed.sh
```

## Run Rendering + Evaluation

```bash
python render.py --model_path <OUTPUT_PATH> --pc --skip_video --skip_train --configs arguments/imed.py
python metrics.py --model_path <OUTPUT_PATH>
```

## Docker-Compatible Branch

This branch packages the iMED NVS baseline and a participant-facing
Docker-compatible submission scaffold.

- The repository root builds the Endo-4DGS baseline image.
- `imednvs_submission/` is a CUDA/3DGS-ready Docker submission scaffold that
  participants can copy and replace with their own per-sequence optimization
  and rendering method.
- Challenge data are not included in either Docker image. Mount iMED NVS data
  at runtime as `/input:ro` and write predictions or model outputs to `/output`.

The published baseline image is:

```bash
docker pull docker.synapse.org/syn74277461/imed-nvs-baseline:v1
```

### Build Baseline Image

```bash
docker build -t imed-nvs-baseline:dev .
```

### Run One Sequence

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/iMED_NVS/session_004_scene_2_tool_1:/input:ro \
  -v /path/to/outputs:/output \
  imed-nvs-baseline:dev \
  run-sequence \
  --sequence /input \
  --output /output/session_004_scene_2_tool_1
```

### Run All Detected Sequences

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/iMED_NVS:/input:ro \
  -v /path/to/outputs:/output \
  imed-nvs-baseline:dev \
  run-dataset \
  --data-root /input \
  --output-root /output
```

The root Docker entrypoint is `imed_nvs_baseline.py`. It trains on
`endoscope2`, renders held-out `endoscope1`, and uses a writable
`_input_sequence` view inside the output directory so `/input` can remain
read-only.

### Submission Scaffold

To build and test the participant submission scaffold:

```bash
cd imednvs_submission
docker build -t my-nvs-submission:dev .
./scripts/local_test.sh my-nvs-submission:dev /path/to/iMED_NVS ./my_test_output
```

The template follows the NVS submission contract directly: the container reads
mounted sequence data from `/input` and writes rendered target-view RGB PNGs to
`/output/<sequence_name>/renders/`.

Unlike the iMED pose-estimation task, many NVS methods, especially 3DGS-style
methods, perform per-sequence optimization on the hidden test sequence before
rendering. The submitted Docker image should therefore include all code,
compiled CUDA extensions, pretrained weights, and other assets needed for
training/optimization and rendering at runtime.
