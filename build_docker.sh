#!/usr/bin/env bash
# Build and optionally test the Docker image for TIRA submission.
# Run from the solution/ directory.
set -euo pipefail

IMAGE_NAME="${1:-pan26-style-change}"
INPUT_DIR="${2:-}"
OUTPUT_DIR="${3:-./tira-output}"

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .
echo ""
echo "Build successful: $IMAGE_NAME"

if [[ -n "$INPUT_DIR" ]]; then
    echo ""
    echo "Running local test on: $INPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    docker run --rm \
        --gpus=all \
        -v "$(realpath "$INPUT_DIR")":/input \
        -v "$(realpath "$OUTPUT_DIR")":/output \
        "$IMAGE_NAME" -i /input -o /output
    echo ""
    echo "Test outputs written to: $OUTPUT_DIR"
fi

echo ""
echo "TIRA submission command:"
echo "  tira-run \\"
echo "    --input-dataset multi-author-writing-style-analysis-2026/smoketest-20260330-training \\"
echo "    --image $IMAGE_NAME \\"
echo "    --command 'python /app/predict.py -i \$inputDataset -o \$outputDir' \\"
echo "    --push true"