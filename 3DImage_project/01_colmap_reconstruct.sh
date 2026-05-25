#!/bin/bash
# STEP 1: COLMAP - Feature Extraction, Matching, and Sparse Reconstruction
# Pipeline: Images → COLMAP (pose + intrinsics)


set -e

COLMAP_BIN=${COLMAP_BIN:-colmap}
IMAGE_PATH=${1:-"./images"}
OUTPUT_PATH=${2:-"./colmap_output"}

echo "[COLMAP] Starting reconstruction pipeline..."
echo "  Image path  : $IMAGE_PATH"
echo "  Output path : $OUTPUT_PATH"

mkdir -p "$OUTPUT_PATH/sparse"

# --- Feature Extraction ---
echo "[COLMAP] Extracting features..."
$COLMAP_BIN feature_extractor \
    --database_path "$OUTPUT_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

# --- Feature Matching ---
echo "[COLMAP] Matching features (exhaustive)..."
$COLMAP_BIN exhaustive_matcher \
    --database_path "$OUTPUT_PATH/database.db" \
    --SiftMatching.use_gpu 1

# --- Sparse Reconstruction (Mapper) ---
echo "[COLMAP] Running mapper..."
$COLMAP_BIN mapper \
    --database_path "$OUTPUT_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH/sparse"

# --- Convert to TXT format for easy parsing ---
echo "[COLMAP] Converting model to TXT..."
$COLMAP_BIN model_converter \
    --input_path "$OUTPUT_PATH/sparse/0" \
    --output_path "$OUTPUT_PATH/sparse/0" \
    --output_type TXT

echo "[COLMAP] Done. Results saved to: $OUTPUT_PATH/sparse/0"
echo "  cameras.txt  - intrinsics per camera"
echo "  images.txt   - poses (R, t) per image"
echo "  points3D.txt - sparse 3D points"
