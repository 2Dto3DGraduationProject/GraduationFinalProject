"""
STEP 7: Texture Mapping
------------------------
Input : Poisson mesh PLY + COLMAP images + poses.npz
Output: textured_mesh.obj + .mtl + atlas.png

For each face, selects the best-angle image as texture source
and projects it onto UV-unwrapped face.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d

try:
    import xatlas
    HAS_XATLAS = True
except ImportError:
    HAS_XATLAS = False
    print("[Texture] xatlas not found. Falling back to per-face planar UV. "
          "Install: pip install xatlas")


def load_poses_npz(path):
    d = np.load(path, allow_pickle=True)
    return {"image_names": d["image_names"].tolist(),
            "poses_c2w":   d["poses_c2w"],
            "poses_w2c":   d["poses_w2c"],
            "intrinsics":  d["intrinsics"],
            "width":       int(d["width"]),
            "height":      int(d["height"])}


def project_point(xyz_world, w2c, K):
    xyz_cam = w2c[:3, :3] @ xyz_world + w2c[:3, 3]
    z = xyz_cam[2]
    if z <= 0:
        return None, None, z
    px = K[0, 0] * xyz_cam[0] / z + K[0, 2]
    py = K[1, 1] * xyz_cam[1] / z + K[1, 2]
    return px, py, z


def select_best_image_per_face(mesh, poses, image_dir):
    image_names = poses["image_names"]
    poses_c2w   = poses["poses_c2w"]
    poses_w2c   = poses["poses_w2c"]
    intrinsics  = poses["intrinsics"]
    W, H        = poses["width"], poses["height"]

    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8

    best_cam = np.full(len(tris), -1, dtype=np.int32)
    best_dot = np.full(len(tris), -np.inf, dtype=np.float32)

    print(f"[Texture] Selecting best image for {len(tris):,} faces from {len(image_names)} cameras...")

    for i, name in enumerate(image_names):
        if not (image_dir / name).exists():
            continue
        cam_pos  = poses_c2w[i][:3, 3]
        w2c, K   = poses_w2c[i], intrinsics[i]

        to_cam   = cam_pos[None, :] - face_centers
        to_cam  /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-8
        dot      = np.sum(face_normals * to_cam, axis=1)

        xyz_cam  = (w2c[:3, :3] @ face_centers.T).T + w2c[:3, 3]
        zc = xyz_cam[:, 2]
        pxc = K[0, 0] * xyz_cam[:, 0] / (zc + 1e-8) + K[0, 2]
        pyc = K[1, 1] * xyz_cam[:, 1] / (zc + 1e-8) + K[1, 2]

        valid  = (zc > 0) & (pxc >= 0) & (pxc < W) & (pyc >= 0) & (pyc < H) & (dot > 0)
        update = valid & (dot > best_dot)
        best_dot[update] = dot[update]
        best_cam[update] = i

    unassigned = (best_cam == -1).sum()
    if unassigned > 0:
        print(f"[Texture] {unassigned} faces unassigned -> fallback to cam 0")
        best_cam[best_cam == -1] = 0

    return best_cam


def build_atlas_and_texture(mesh, poses, image_dir, output_dir, atlas_size=4096):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris  = np.asarray(mesh.triangles, dtype=np.uint32)

    best_cam    = select_best_image_per_face(mesh, poses, image_dir)
    image_names = poses["image_names"]
    poses_w2c   = poses["poses_w2c"]
    intrinsics  = poses["intrinsics"]

    if HAS_XATLAS:
        print("[Texture] Running xatlas UV unwrap...")
        vmapping, indices, uvs = xatlas.parametrize(verts, tris)
        new_tris  = indices
        uv_coords = uvs
    else:
        print("[Texture] Per-face planar UV (fallback)...")
        uv_coords = np.zeros((len(tris) * 3, 2), dtype=np.float32)
        new_tris  = np.arange(len(tris) * 3, dtype=np.uint32).reshape(-1, 3)

    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    img_cache = {}

    def get_image(idx):
        if idx not in img_cache:
            img_cache[idx] = np.array(
                Image.open(image_dir / image_names[idx]).convert("RGB"))
        return img_cache[idx]

    n_faces = len(tris)
    cols    = int(np.ceil(np.sqrt(n_faces)))
    tile    = max(1, atlas_size // cols)

    print(f"[Texture] Atlas {atlas_size}x{atlas_size} | grid {cols}x{cols} | tile={tile}px")

    for fi in range(n_faces):
        cam_idx = best_cam[fi]
        w2c, K  = poses_w2c[cam_idx], intrinsics[cam_idx]
        img     = get_image(cam_idx)
        ih, iw  = img.shape[:2]

        row = fi // cols;  col = fi % cols
        u0  = col * tile;  v0  = row * tile

        centroid = verts[tris[fi]].mean(axis=0)
        px, py, z = project_point(centroid, w2c, K)
        if px is not None and z > 0:
            pxi = int(np.clip(px, 0, iw - 1))
            pyi = int(np.clip(py, 0, ih - 1))
            atlas[v0:v0+tile, u0:u0+tile] = img[pyi, pxi]

        fu = (u0 + tile / 2.0) / atlas_size
        fv = (v0 + tile / 2.0) / atlas_size
        if not HAS_XATLAS:
            uv_coords[fi*3:fi*3+3] = [fu, fv]

    atlas_path = output_dir / "atlas.png"
    Image.fromarray(atlas).save(atlas_path)
    print(f"[Texture] Atlas saved -> {atlas_path}")

    obj_path = output_dir / "textured_mesh.obj"
    mtl_path = output_dir / "textured_mesh.mtl"

    with open(mtl_path, "w") as f:
        f.write("newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n"
                f"map_Kd {atlas_path.name}\n")

    if HAS_XATLAS:
        face_v  = np.array([[vmapping[new_tris[fi, k]] for k in range(3)]
                             for fi in range(n_faces)])
        face_vt = new_tris
        vt_list = uv_coords
    else:
        face_v  = tris
        face_vt = new_tris
        vt_list = uv_coords

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\nusemtl material0\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in vt_list:
            f.write(f"vt {uv[0]:.6f} {1.0-uv[1]:.6f}\n")
        for fi in range(n_faces):
            vi, ti = face_v[fi], face_vt[fi]
            f.write(f"f {vi[0]+1}/{ti[0]+1} {vi[1]+1}/{ti[1]+1} {vi[2]+1}/{ti[2]+1}\n")

    print(f"[Texture] OBJ -> {obj_path}")
    return str(obj_path)


def run_texturing(mesh_ply, poses_npz, image_dir, output_dir, atlas_size=4096):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir  = Path(image_dir)

    print(f"[Texture] Loading mesh: {mesh_ply}")
    mesh = o3d.io.read_triangle_mesh(mesh_ply)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"[Texture] {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")

    poses    = load_poses_npz(poses_npz)
    obj_path = build_atlas_and_texture(mesh, poses, image_dir, output_dir, atlas_size)

    print(f"[Texture] Done -> {obj_path}")
    return obj_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_ply",   required=True)
    parser.add_argument("--poses_npz",  required=True)
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--atlas_size", type=int, default=4096)
    args = parser.parse_args()

    run_texturing(args.mesh_ply, args.poses_npz,
                  args.image_dir, args.output_dir, args.atlas_size)