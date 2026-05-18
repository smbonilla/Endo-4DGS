#!/usr/bin/env sh
set -eu

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 IMAGE_NAME /path/to/iMED_NVS /path/to/output [extra container args...]" >&2
    exit 2
fi

IMAGE_NAME="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"
shift 3

mkdir -p "$OUTPUT_DIR"

docker run --rm ${IMED_NVS_DOCKER_GPU_ARGS:---gpus all} --ipc=host \
    -v "$INPUT_DIR:/input:ro" \
    -v "$OUTPUT_DIR:/output" \
    "$IMAGE_NAME" "$@"

python3 "$(dirname "$0")/check_outputs.py" "$INPUT_DIR" "$OUTPUT_DIR"
