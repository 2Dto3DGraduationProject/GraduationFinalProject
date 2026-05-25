"""
STEP 2: Parse COLMAP Output
----------------------------
Reads cameras.txt and images.txt from COLMAP sparse reconstruction
and exports them as poses.npz for downstream stages.
"""

import argparse
import numpy as np
from pathlib import Path


def parse_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts  = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            width  = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            elif model in ("PINHOLE", "OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")

            K = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0.,  0., 1.]], dtype=np.float64)
            cameras[cam_id] = {"K": K, "width": width, "height": height, "model": model}
    return cameras


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*qy**2-2*qz**2,   2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [  2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2,   2*qy*qz-2*qx*qw],
        [  2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2],
    ], dtype=np.float64)


def parse_images_txt(path):
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

        R   = qvec2rotmat(qvec)
        c2w = np.eye(4, dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T;   c2w[:3, 3] = -R.T @ tvec
        w2c[:3, :3] = R;     w2c[:3, 3] = tvec

        images[name] = {"image_id": image_id, "camera_id": camera_id,
                        "c2w": c2w, "w2c": w2c}
    return images


def export_poses(colmap_sparse_dir, output_dir):
    colmap_sparse_dir = Path(colmap_sparse_dir)
    output_dir        = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = parse_cameras_txt(colmap_sparse_dir / "cameras.txt")
    images  = parse_images_txt(colmap_sparse_dir  / "images.txt")

    names      = sorted(images.keys())
    poses_c2w  = []
    poses_w2c  = []
    intrinsics = []

    for name in names:
        img = images[name]
        cam = cameras[img["camera_id"]]
        poses_c2w.append(img["c2w"])
        poses_w2c.append(img["w2c"])
        intrinsics.append(cam["K"])

    poses_c2w  = np.stack(poses_c2w,  axis=0)
    poses_w2c  = np.stack(poses_w2c,  axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    out_path = output_dir / "poses.npz"
    np.savez(out_path,
             image_names=np.array(names),
             poses_c2w=poses_c2w,
             poses_w2c=poses_w2c,
             intrinsics=intrinsics,
             width=cam["width"],
             height=cam["height"])

    print(f"[COLMAP Parser] {len(names)} images parsed -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir",  required=True)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()
    export_poses(args.colmap_dir, args.output_dir)