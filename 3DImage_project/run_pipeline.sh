#!/bin/bash
# =============================================================================
# MASTER PIPELINE RUNNER (SAM SEGMENTATION VERSIYONU)
# Images → COLMAP → DepthAnything → ScaleNorm → IncrementalFusion → Poisson → Texture
# =============================================================================

set -e

IMAGE_DIR=${IMAGE_DIR:-"./images"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./output"}
DA_ENCODER=${DA_ENCODER:-"vitl"}
DA_CHECKPOINT=${DA_CHECKPOINT:-""}
DA_INPUT_SIZE=${DA_INPUT_SIZE:-518}
DEDUP_CELL=${DEDUP_CELL:-0.008}        # dedup voxel cell in meters
DEPTH_MIN=${DEPTH_MIN:-0.1}
DEPTH_MAX=${DEPTH_MAX:-10.0}
SCALE_FACTOR=${SCALE_FACTOR:-0.01}
POISSON_DEPTH=${POISSON_DEPTH:-9}
ATLAS_SIZE=${ATLAS_SIZE:-4096}
MASK_EROSION=${MASK_EROSION:-2}
DEBUG_PER_IMAGE=${DEBUG_PER_IMAGE:-0}

# SAM Segmentation parametreleri
SAM_MODEL=${SAM_MODEL:-"vit_b"}         # vit_h (en iyi), vit_l, vit_b (hızlı), mobile_sam
SAM_DEVICE=${SAM_DEVICE:-"cuda"}        # cuda or cpu
USE_OTSU=${USE_OTSU:-0}                 # SAM yüklü değilse 1 yap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo " 3D Rekonstriksiyon Pipeline (SAM Segmentation)"
echo "================================================"
echo " Görüntüler      : $IMAGE_DIR"
echo " Çıktı           : $OUTPUT_ROOT"
echo " DA Encoder      : $DA_ENCODER"
echo " Dedup cell      : ${DEDUP_CELL}m"
echo " Derinlik aralığı: [${DEPTH_MIN}, ${DEPTH_MAX}]m"
echo " SAM Model       : $SAM_MODEL"
echo " SAM Device      : $SAM_DEVICE"
echo "================================================"

COLMAP_OUT="$OUTPUT_ROOT/colmap"
POSES_OUT="$OUTPUT_ROOT/poses"
DEPTH_OUT="$OUTPUT_ROOT/depth_relative"
DEPTH_METRIC_OUT="$OUTPUT_ROOT/depth_metric"
MASKS_OUT="$OUTPUT_ROOT/masks"
FUSION_OUT="$OUTPUT_ROOT/tsdf"
POISSON_OUT="$OUTPUT_ROOT/poisson"
TEXTURE_OUT="$OUTPUT_ROOT/textured"

mkdir -p "$COLMAP_OUT" "$POSES_OUT" "$DEPTH_OUT" "$DEPTH_METRIC_OUT" \
         "$MASKS_OUT" "$FUSION_OUT" "$POISSON_OUT" "$TEXTURE_OUT"

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 1/7] COLMAP: Feature extraction + matching + reconstruction"
python "$SCRIPT_DIR/01_colmap.py" "$IMAGE_DIR" "$COLMAP_OUT"

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 2/7] Parse COLMAP output → poses.npz"
python "$SCRIPT_DIR/02_parse_colmap.py" \
    --colmap_dir "$COLMAP_OUT/sparse/0" \
    --output_dir "$POSES_OUT"

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 3/7] DepthAnything: per-image depth estimation"
DEPTH_ARGS="--image_dir $IMAGE_DIR --output_dir $DEPTH_OUT --encoder $DA_ENCODER --input_size $DA_INPUT_SIZE"
[ -n "$DA_CHECKPOINT" ] && DEPTH_ARGS="$DEPTH_ARGS --checkpoint $DA_CHECKPOINT"
python "$SCRIPT_DIR/03_depth_inference.py" $DEPTH_ARGS

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 3.5/7] Segmentation: SAM ile ön plan tespiti (değişken arka plan)"
SAM_ARGS="--image_dir $IMAGE_DIR --output_dir $MASKS_OUT --model_type $SAM_MODEL --device $SAM_DEVICE"
if [ "$USE_OTSU" = "1" ]; then
    SAM_ARGS="$SAM_ARGS --use_otsu"
fi
python "$SCRIPT_DIR/03.5_segmentation_SAM.py" $SAM_ARGS

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 4/7] Scale normalization: relative → metric depth"
python "$SCRIPT_DIR/04_scale_normalization.py" \
    --colmap_dir "$COLMAP_OUT/sparse/0" \
    --depth_dir  "$DEPTH_OUT" \
    --output_dir "$DEPTH_METRIC_OUT"

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 5/7] Incremental point cloud fusion (TSDF)"
FUSION_ARGS=(
    --poses_npz   "$POSES_OUT/poses.npz"
    --image_dir   "$IMAGE_DIR"
    --depth_dir   "$DEPTH_METRIC_OUT"
    --mask_dir    "$MASKS_OUT"
    --output_dir  "$FUSION_OUT"
    --dedup_cell  "$DEDUP_CELL"
    --depth_min   "$DEPTH_MIN"
    --depth_max   "$DEPTH_MAX"
    --scale       "$SCALE_FACTOR"
    --mask_erosion "$MASK_EROSION"
)
if [ "$DEBUG_PER_IMAGE" = "1" ]; then
    FUSION_ARGS+=(--debug_per_image)
fi
python "$SCRIPT_DIR/05_incremental_fusion.py" "${FUSION_ARGS[@]}"

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 6/7] Poisson surface reconstruction"
python "$SCRIPT_DIR/06_poisson_mesh.py" \
    --input       "$FUSION_OUT/tsdf_mesh.ply" \
    --output_dir  "$POISSON_OUT" \
    --depth       "$POISSON_DEPTH"

# --------------------------------------------------------------------------
echo ""
echo "[ADIM 7/7] Texture mapping"
python "$SCRIPT_DIR/07_texture_mesh.py" \
    --mesh_ply    "$POISSON_OUT/poisson_mesh.ply" \
    --poses_npz   "$POSES_OUT/poses.npz" \
    --image_dir   "$IMAGE_DIR" \
    --output_dir  "$TEXTURE_OUT" \
    --atlas_size  "$ATLAS_SIZE"

# --------------------------------------------------------------------------
echo ""
echo "================================================"
echo " Pipeline Tamamlandı ✓"
echo "================================================"
echo " COLMAP sparse  : $COLMAP_OUT/sparse/0/"
echo " Poses          : $POSES_OUT/poses.npz"
echo " Depth (rel)    : $DEPTH_OUT/"
echo " Depth (metric) : $DEPTH_METRIC_OUT/"
echo " Maskeler       : $MASKS_OUT/ (SAM ile)"
echo " Nokta bulutu   : $FUSION_OUT/tsdf_mesh.ply"
echo " Poisson mesh   : $POISSON_OUT/poisson_mesh.ply"
echo " Dokulu mesh    : $TEXTURE_OUT/textured_mesh.obj"
echo "================================================"