"""
STEP 5: Incremental Point Cloud Fusion
--------------------------------------
Strategy:
  - Frame 0 defines world origin and base point cloud
  - For each subsequent frame, unproject depth -> 3D points in world space
  - Only points that are NOT already in the base cloud are added 
  - No voxel grid RAM explosion, no TSDF marching cubes artifacts

Output: tsdf_mesh.ply  
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
from scipy.ndimage import binary_dilation, binary_erosion


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def load_mask(mask_dir: Path, stem: str, h: int, w: int) -> np.ndarray:
    """Return boolean mask (H, W). True = foreground pixel."""
    p = mask_dir / f"{stem}_mask.png"
    if p.exists():
        m = np.array(Image.open(p).convert("L")) > 127
        m = binary_erosion(m, iterations=15)
        return m
    # fallback: all pixels valid
    return np.ones((h, w), dtype=bool)


# ---------------------------------------------------------------------------
# Unprojection: depth map -> 3D points in world space
# ---------------------------------------------------------------------------

def unproject_optimized(depth_im: np.ndarray,
                        K: np.ndarray,
                        c2w: np.ndarray,
                        mask: np.ndarray,
                        depth_min: float,
                        depth_max: float) -> tuple[np.ndarray, np.ndarray]:
    
    # 1. Meshgrid yerine doğrudan maske ve derinlik koşullarını kontrol et
    valid = mask & (depth_im > depth_min) & (depth_im < depth_max)
    
    # 2. Sadece geçerli piksellerin koordinatlarını al 
    yv, xv = np.nonzero(valid)
    
    if len(yv) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    z = depth_im[yv, xv].astype(np.float32)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # 3. 3D Kamera koordinatları
    X = (xv.astype(np.float32) - cx) / fx * z
    Y = (yv.astype(np.float32) - cy) / fy * z
    pts_cam = np.stack([X, Y, z], axis=-1)  # (N, 3)

    # 4. Homojen matris (ones) KULLANMADAN doğrudan Rotasyon ve Çeviri (Çok daha hızlı)
    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    
    pts_world = pts_cam @ R.T + t  # (N, 3)

    # 5. Renk örneklemesi için orijinal piksel koordinatları
    pix_yx = np.stack([yv.astype(np.int32), xv.astype(np.int32)], axis=-1)
    
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

def run_incremental_fusion(poses_npz: str, image_dir: str, depth_dir: str, mask_dir: str, output_dir: str, dedup_cell: float = 0.008, depth_min: float = 0.1, depth_max: float = 10.0, scale_factor: float = 0.01):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir, depth_dir, mask_dir = Path(image_dir), Path(depth_dir), Path(mask_dir)

    data = np.load(poses_npz, allow_pickle=True)
    image_names = data["image_names"].tolist()
    poses_c2w = data["poses_c2w"].astype(np.float64)
    intrinsics = data["intrinsics"].astype(np.float64)
    N = len(image_names)

    print(f"[Fusion] TSDF Hacimsel Birleştirme Başlatılıyor... (Hassasiyet: {dedup_cell}m)")
    
    # TSDF Motoru: Üst üste binen katmanları matematiksel olarak ortalar ve tek bir yüzeye çeker.
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=dedup_cell,
        sdf_trunc=dedup_cell * 25,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i, name in enumerate(image_names):
        stem = Path(name).stem
        depth_path = depth_dir / f"{stem}_depth_metric.npy"
        color_path = image_dir / name

        if not depth_path.exists() or not color_path.exists():
            continue

        depth_im = np.load(depth_path).astype(np.float32)
        color_im = np.array(Image.open(color_path).convert("RGB"))
        h, w = depth_im.shape

        # Sadece arkaplanı silinmiş maskeli alanı al
        mask = load_mask(mask_dir, stem, h, w)
        depth_im[~mask] = 0.0  
        
        # Kamera arkasına uzanan hatalı derinlikleri uzay boşluğuna at (1.5m sınırı)
        depth_im[depth_im > depth_max] = 0.0

        color_o3d = o3d.geometry.Image(color_im)
        depth_o3d = o3d.geometry.Image(depth_im)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, 
            depth_scale=1.0, 
            depth_trunc=depth_max, 
            convert_rgb_to_intensity=False
        )

        K = intrinsics[i]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h,
            fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2]
        )
        
        extrinsic = np.linalg.inv(poses_c2w[i]) # w2c
        volume.integrate(rgbd, intrinsic_o3d, extrinsic)
        print(f"[Fusion] [{i+1}/{N}] {name} hacme eritildi.")

    print("\n[Fusion] Katmanlar eritilip TEK KATMANLI nokta bulutu çıkarılıyor...")
    
    # İŞTE BURASI: Mesh değil, TSDF'nin erittiği yapıdan sadece nokta bulutunu çekiyoruz.
    pcd = volume.extract_point_cloud()

    if len(pcd.points) == 0:
        print("[Fusion] HATA: Nokta bulutu oluşturulamadı.")
        return

    print(f"[Fusion] Katmanlardan arındırılmış net nokta sayısı: {len(pcd.points):,}")

    # Uçuşan son gürültüleri tıraşla
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.03)

    # Hizalama ve Oyun Motoru Ölçeği
    pcd = orient_upright(pcd, poses_c2w)
    pcd.scale(scale_factor, center=(0, 0, 0))

    out_path = output_dir / "tsdf_mesh.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[Fusion] Çıktı Kaydedildi -> {out_path}")

    print("[Görselleştirme] Üst üste binmeleri eritilmiş Nokta Bulutu Önizlemesi...")
    o3d.visualization.draw_geometries([pcd], window_name="TSDF Nokta Bulutu (Kaynaşmış)", width=1280, height=720)

    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental Point Cloud Fusion")
    parser.add_argument("--poses_npz",   required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--depth_dir",   required=True)
    parser.add_argument("--mask_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--dedup_cell",  type=float, default=0.015,
                        help="Dedup voxel cell size in meters (default: 0.015 = 15mm)")
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