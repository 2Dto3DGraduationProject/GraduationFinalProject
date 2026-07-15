"""
STEP 4: Scale Normalization — relative depth -> metric depth
-------------------------------------------------------------
Aligns DepthAnything relative disparity maps to COLMAP metric scale
using least-squares fitting on per-image 3D point observations.
"""

import argparse
import numpy as np
from pathlib import Path


def load_points3d_txt(path):
    points3d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            pid   = int(parts[0])
            xyz   = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            track = parts[8:]
            image_ids = [int(track[i]) for i in range(0, len(track), 2)]
            points3d[pid] = {"xyz": xyz, "image_ids": image_ids}
    return points3d


def load_images_txt(path):
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        parts     = lines[i].split()
        image_id  = int(parts[0])
        qvec      = np.array(list(map(float, parts[1:5])))
        tvec      = np.array(list(map(float, parts[5:8])))
        camera_id = int(parts[8])
        name      = parts[9]

        qw, qx, qy, qz = qvec
        R = np.array([
            [1-2*qy**2-2*qz**2,   2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
            [  2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2,   2*qy*qz-2*qx*qw],
            [  2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2],
        ], dtype=np.float64)
        w2c          = np.eye(4)
        w2c[:3, :3]  = R
        w2c[:3,  3]  = tvec
        images[image_id] = {"name": name, "w2c": w2c, "camera_id": camera_id}
    return images


def load_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts  = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            params = list(map(float, parts[4:]))
            if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            elif model in ("PINHOLE", "OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                raise ValueError(f"Unsupported model: {model}")
            cameras[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return cameras


def align_least_squares(depth_rel, depths_metric):
    disp = 1.0 / (depths_metric + 1e-8)
    A    = np.stack([depth_rel, np.ones_like(depth_rel)], axis=1)
    res, _, _, _ = np.linalg.lstsq(A, disp, rcond=None)
    return res[0], res[1]  # scale, shift


def normalize_depths(colmap_dir, depth_dir, output_dir,
                     min_track_len=3, min_obs=5):
    colmap_dir = Path(colmap_dir)
    depth_dir  = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras  = load_cameras_txt(colmap_dir  / "cameras.txt")
    images   = load_images_txt(colmap_dir   / "images.txt")
    points3d = load_points3d_txt(colmap_dir / "points3D.txt")

    # Build per-image COLMAP depth observations
    img_id_to_depths = {iid: {} for iid in images}
    for pid, pt in points3d.items():
        if len(pt["image_ids"]) < min_track_len:
            continue
        for iid in pt["image_ids"]:
            if iid not in images:
                continue
            w2c  = images[iid]["w2c"]
            z    = float((w2c[:3, :3] @ pt["xyz"] + w2c[:3, 3])[2])
            if z > 0:
                img_id_to_depths[iid][pid] = z

    scaled, skipped = 0, 0

    for image_id, img in images.items():
        name  = img["name"]
        stem  = Path(name).stem
        dpath = depth_dir / f"{stem}_depth.npy"
        if not dpath.exists():
            skipped += 1
            continue

        depth_rel = np.load(dpath).astype(np.float64)
        H, W      = depth_rel.shape
        K         = cameras[img["camera_id"]]
        fx, fy    = K[0, 0], K[1, 1]
        cx, cy    = K[0, 2], K[1, 2]

        colmap_depths = img_id_to_depths.get(image_id, {})

        if len(colmap_depths) < min_obs:
            scale = np.median(depth_rel[depth_rel > 0])
            depth_metric = depth_rel / scale if scale > 0 else depth_rel
            skipped += 1
        else:
            d_rel_s, d_met_s = [], []
            for pid, z_met in colmap_depths.items():
                xyz_c = images[image_id]["w2c"][:3, :3] @ points3d[pid]["xyz"] + images[image_id]["w2c"][:3, 3]
                px = int(round(fx * xyz_c[0] / xyz_c[2] + cx))
                py = int(round(fy * xyz_c[1] / xyz_c[2] + cy))
                if 0 <= px < W and 0 <= py < H:
                    d_rel_s.append(depth_rel[py, px])
                    d_met_s.append(z_met)

            if len(d_rel_s) < min_obs:
                dm_med = float(np.median(1.0 / (np.array(d_met_s) + 1e-8)))
                dr_med = float(np.median(d_rel_s)) + 1e-8
                disp   = np.clip(depth_rel * (dm_med / dr_med), 1e-4, None)
                depth_metric = 1.0 / disp
            else:
                s, sh  = align_least_squares(np.array(d_rel_s), np.array(d_met_s))
                disp   = np.clip(depth_rel * s + sh, 1e-4, None)
                depth_metric = 1.0 / disp

            scaled += 1

        depth_metric = np.clip(depth_metric, 0, None).astype(np.float64)
        valid = depth_metric[depth_metric > 0]
        print(f"[ScaleNorm] {name} | metric depth min={valid.min():.3f} max={valid.max():.3f} m")

        np.save(output_dir / f"{stem}_depth_metric.npy", depth_metric)

    print(f"[ScaleNorm] Done. Aligned={scaled} Fallback={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir",    required=True)
    parser.add_argument("--depth_dir",     required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--min_track_len", type=int, default=5)
    parser.add_argument("--min_obs",       type=int, default=15)
    args = parser.parse_args()

    normalize_depths(args.colmap_dir, args.depth_dir, args.output_dir,
                     args.min_track_len, args.min_obs)