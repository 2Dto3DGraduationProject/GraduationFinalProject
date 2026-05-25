"""
STEP 5: Incremental Point Cloud Fusion
--------------------------------------
Strategy:
  - Frame 0 defines world origin and base point cloud
  - For each subsequent frame, unproject depth -> 3D points in world space
  - Only points that are NOT already in the base cloud are added (spatial hash dedup)
  - No voxel grid RAM explosion, no TSDF marching cubes artifacts

Output: tsdf_mesh.ply  (named for pipeline compatibility, actually a colored point cloud PLY)
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def load_mask(mask_dir: Path, stem: str, h: int, w: int) -> np.ndarray:
    """Return boolean mask (H, W). True = foreground pixel."""
    p = mask_dir / f"{stem}_mask.png"
    if p.exists():
        m = np.array(Image.open(p).convert("L")) > 127
        return m
    # fallback: all pixels valid
    return np.ones((h, w), dtype=bool)


# ---------------------------------------------------------------------------
# Unprojection: depth map -> 3D points in world space
# ---------------------------------------------------------------------------

def unproject(depth_im: np.ndarray,
              K: np.ndarray,
              c2w: np.ndarray,
              mask: np.ndarray,
              depth_min: float,
              depth_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject valid depth pixels to world-space XYZ.
    Returns:
        pts_world : (N, 3) float32
        pix_yx    : (N, 2) int32  — row, col indices for color sampling
    """
    h, w = depth_im.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')  # (H, W)

    valid = mask & (depth_im > depth_min) & (depth_im < depth_max)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    z  = depth_im[valid].astype(np.float32)
    xv = xs[valid].astype(np.float32)
    yv = ys[valid].astype(np.float32)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (xv - cx) / fx * z
    Y = (yv - cy) / fy * z
    Z = z

    pts_cam = np.stack([X, Y, Z], axis=1)          # (N, 3)
    ones    = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
    pts_h   = np.hstack([pts_cam, ones])            # (N, 4)
    pts_world = (c2w.astype(np.float32) @ pts_h.T).T[:, :3]  # (N, 3)

    pix_yx = np.stack([yv.astype(np.int32), xv.astype(np.int32)], axis=1)
    return pts_world, pix_yx


# ---------------------------------------------------------------------------
# Spatial hash deduplication
# ---------------------------------------------------------------------------

def voxel_key(pts: np.ndarray, cell_size: float) -> np.ndarray:
    """Map (N, 3) float -> (N, 3) int voxel indices."""
    return np.floor(pts / cell_size).astype(np.int64)


def build_hash_set(pts: np.ndarray, cell_size: float) -> set:
    keys = voxel_key(pts, cell_size)
    return set(map(tuple, keys))


def filter_new_points(pts: np.ndarray,
                      existing_set: set,
                      cell_size: float) -> tuple[np.ndarray, set]:
    """
    Return only pts whose voxel cell is NOT in existing_set.
    Also returns the updated set including the new cells.
    """
    if pts.shape[0] == 0:
        return pts, existing_set

    keys = voxel_key(pts, cell_size)
    key_tuples = list(map(tuple, keys))

    new_mask = np.array([k not in existing_set for k in key_tuples], dtype=bool)
    new_pts  = pts[new_mask]
    new_keys = [k for k, m in zip(key_tuples, new_mask) if m]

    updated_set = existing_set | set(new_keys)
    return new_pts, updated_set, new_mask


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def remove_statistical_outliers(pcd: o3d.geometry.PointCloud,
                                 nb_neighbors: int = 20,
                                 std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    return cl


def orient_upright(pcd: o3d.geometry.PointCloud,
                   poses_c2w: np.ndarray) -> o3d.geometry.PointCloud:
    """Rotate cloud so dominant camera-orbit axis aligns with +Y."""
    cam_pos  = poses_c2w[:, :3, 3]
    centered = cam_pos - cam_pos.mean(axis=0)
    cov      = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    up = evecs[:, 0]
    if np.dot(up, cam_pos.mean(axis=0)) < 0:
        up = -up

    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, target)
    s = np.linalg.norm(v)
    c = float(np.dot(up, target))
    if s < 1e-6:
        return pcd
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    R  = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    pcd.rotate(R, center=pcd.get_center())
    print(f"[Fusion] Upright rotation applied. Up vector: {up.round(4)}")
    return pcd


# ---------------------------------------------------------------------------
# Save as PLY (point cloud — named tsdf_mesh.ply for pipeline compat)
# ---------------------------------------------------------------------------

def save_point_cloud_ply(pts: np.ndarray,
                          cols: np.ndarray,
                          path: Path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"[Fusion] Saved point cloud -> {path}  ({len(pts):,} pts)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_incremental_fusion(poses_npz: str,
                            image_dir: str,
                            depth_dir: str,
                            mask_dir: str,
                            output_dir: str,
                            dedup_cell: float = 0.008,
                            depth_min: float = 0.1,
                            depth_max: float = 10.0,
                            scale_factor: float = 0.01):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir  = Path(image_dir)
    depth_dir  = Path(depth_dir)
    mask_dir   = Path(mask_dir)

    data        = np.load(poses_npz, allow_pickle=True)
    image_names = data["image_names"].tolist()
    poses_c2w   = data["poses_c2w"].astype(np.float64)   # (N, 4, 4)
    intrinsics  = data["intrinsics"].astype(np.float64)   # (N, 3, 3)
    N           = len(image_names)

    print(f"[Fusion] {N} frames | dedup_cell={dedup_cell}m | depth [{depth_min}, {depth_max}]m")

    all_pts  : list[np.ndarray] = []
    all_cols : list[np.ndarray] = []
    occupied : set              = set()   # spatial hash of occupied cells

    for i, name in enumerate(image_names):
        stem       = Path(name).stem
        depth_path = depth_dir  / f"{stem}_depth_metric.npy"
        color_path = image_dir  / name

        if not depth_path.exists() or not color_path.exists():
            print(f"[Fusion] [{i+1}/{N}] SKIP (missing file): {name}")
            continue

        depth_im = np.load(depth_path).astype(np.float32)
        color_im = np.array(Image.open(color_path).convert("RGB"), dtype=np.uint8)
        h, w     = depth_im.shape

        mask = load_mask(mask_dir, stem, h, w)

        pts_world, pix_yx = unproject(
            depth_im, intrinsics[i], poses_c2w[i],
            mask, depth_min, depth_max
        )

        if pts_world.shape[0] == 0:
            print(f"[Fusion] [{i+1}/{N}] {name} | 0 valid points, skipped")
            continue

        # Frame 0: all points go in as base cloud
        if i == 0:
            new_pts  = pts_world
            new_mask_bool = np.ones(pts_world.shape[0], dtype=bool)
            keys     = voxel_key(pts_world, dedup_cell)
            occupied = set(map(tuple, keys))
            print(f"[Fusion] [{i+1}/{N}] {name} | BASE: {len(new_pts):,} pts -> occupied cells: {len(occupied):,}")
        else:
            keys      = voxel_key(pts_world, dedup_cell)
            key_tuples = list(map(tuple, keys))
            new_mask_bool = np.array([k not in occupied for k in key_tuples], dtype=bool)
            new_pts   = pts_world[new_mask_bool]
            new_keys  = {k for k, m in zip(key_tuples, new_mask_bool) if m}
            occupied |= new_keys
            print(f"[Fusion] [{i+1}/{N}] {name} | "
                  f"total={pts_world.shape[0]:,}  new={new_pts.shape[0]:,}  "
                  f"skipped={pts_world.shape[0]-new_pts.shape[0]:,}  "
                  f"occupied={len(occupied):,}")

        if new_pts.shape[0] == 0:
            continue

        new_cols = color_im[pix_yx[new_mask_bool, 0], pix_yx[new_mask_bool, 1]]  # (M, 3)

        all_pts.append(new_pts)
        all_cols.append(new_cols)

    if not all_pts:
        print("[Fusion] ERROR: No points accumulated. Check depth maps and masks.")
        return

    pts_full  = np.vstack(all_pts).astype(np.float64)
    cols_full = np.vstack(all_cols).astype(np.float64)
    print(f"\n[Fusion] Total before post-proc: {len(pts_full):,} points")

    # --- Build Open3D PCD ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_full)
    pcd.colors = o3d.utility.Vector3dVector(cols_full / 255.0)

    # --- Statistical outlier removal ---
    pcd = remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    print(f"[Fusion] After outlier removal: {len(pcd.points):,} points")

    # --- Upright rotation ---
    pcd = orient_upright(pcd, poses_c2w)

    # --- Scale ---
    pcd.scale(scale_factor, center=(0, 0, 0))
    print(f"[Fusion] Scale factor applied: {scale_factor}")

    # --- Save ---
    out_path = output_dir / "tsdf_mesh.ply"   # name kept for pipeline compat
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[Fusion] Done. Points: {len(pcd.points):,}")
    print(f"[Fusion] Output -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental Point Cloud Fusion")
    parser.add_argument("--poses_npz",   required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--depth_dir",   required=True)
    parser.add_argument("--mask_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--dedup_cell",  type=float, default=0.008,
                        help="Dedup voxel cell size in meters (default: 0.008 = 8mm)")
    parser.add_argument("--depth_min",   type=float, default=0.1)
    parser.add_argument("--depth_max",   type=float, default=10.0)
    parser.add_argument("--scale",       type=float, default=0.01,
                        help="Final scale factor (default: 0.01 for game engine export)")
    args = parser.parse_args()

    run_incremental_fusion(
        poses_npz=args.poses_npz,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        dedup_cell=args.dedup_cell,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        scale_factor=args.scale,
    )