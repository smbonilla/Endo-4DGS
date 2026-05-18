# iMED NVS Submission Template

This directory is a Docker-compatible scaffold for the iMED Novel View
Synthesis task. It is separate from the Endo-4DGS baseline in the repository
root, so participants can start from a clean submission image and replace the
per-sequence optimization and rendering method with their own approach.

Unlike the iMED pose-estimation task, many NVS methods are not pure feed-forward
inference methods. 3DGS-style methods commonly optimize a scene representation
from the source-view frames of each hidden test sequence, then render the target
view. Your submitted Docker image should therefore contain everything needed for
runtime training/optimization and rendering.

## Runtime Contract

The challenge evaluator runs your container with the iMED NVS data mounted
read-only at `/input` and a writable output directory mounted at `/output`:

```bash
docker run --rm --gpus all --ipc=host \
  -v <hidden_nvs_data>:/input:ro \
  -v <prediction_dir>:/output \
  <your_image>
```

The required interface is the Docker input/output behavior: read from `/input`,
optionally train or optimize on the provided source-view sequence, render the
requested target-view images, and write them under `/output`.

## Input Layout

Each sequence contains source-view `endoscope2` frames and camera metadata:

```text
<sequence_name>/
    K.txt
    pose.txt
    endoscope2/
        L/
            frame_*.png
```

Public development data may also contain target-view `endoscope1` images for
local evaluation. Do not read `endoscope1` target RGB images during inference;
hidden-test target images are not available to submitted containers.

## Output Layout

For each sequence, write RGB PNG predictions under:

```text
/output/<sequence_name>/renders/
    00000.png
    00001.png
    00002.png
```

Use one output image per requested target frame. The starter implementation
uses the number of source-view frames to create sequential output names.

## Build

```bash
docker build -t my-nvs-submission:dev .
```

The starter Dockerfile installs a CUDA 11.8/PyTorch environment and common
3DGS dependencies:

```text
diff-gaussian-rasterization-depth @ 03f0b7d00383d6e96c22b37325ac9e5450947bf5
Depth-Anything-ONNX @ d6cc8e1e6713a2129683496b328bde675a66370d
simple-knn
```

If your method uses different CUDA extensions, submodule revisions, or model
weights, install or copy them into the image at build time. The evaluator may
run submitted containers without network access.

## Local Test

```bash
./scripts/local_test.sh my-nvs-submission:dev \
  /path/to/iMED_NVS \
  ./my_test_output
```

The starter method only copies source-view images into the expected output
structure. Replace `render_target_views` in `nvs_method.py` with your actual
NVS training/optimization and rendering pipeline before submission.

## Synapse Submission

```bash
docker tag my-nvs-submission:dev \
  docker.synapse.org/syn74277461/<team-or-method-name>:v1

docker login docker.synapse.org
docker push docker.synapse.org/syn74277461/<team-or-method-name>:v1
```

Submit the uploaded Docker repository through the Synapse Docker tab using
"Submit Docker Repository to Challenge".
