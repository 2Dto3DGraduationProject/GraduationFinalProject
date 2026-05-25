"""
STEP 6: Poisson Surface Reconstruction
----------------------------------------
Input : point cloud PLY (from step 5 incremental fusion)
Output: watertight Poisson mesh PLY
"""

import argparse
import numpy as np
from pathlib import Path
import open3d as o3d
import time


def poisson_from_ply(input_ply, output_dir,
                     depth=9, scale=1.1,
                     linear_fit=False,
                     density_threshold=0.01):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Poisson] Loading: {input_ply}")
    # Try as mesh first, fall back to point cloud
    mesh = o3d.io.read_triangle_mesh(input_ply)

    if mesh.has_vertices() and len(np.asarray(mesh.triangles)) > 0:
        print(f"[Poisson] Input mesh: {len(mesh.vertices):,} vertices, "
              f"{len(mesh.triangles):,} faces")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=min(2_000_000, len(mesh.vertices) * 3),
            use_triangle_normal=True)
    else:
        # It's a point cloud PLY
        pcd = o3d.io.read_point_cloud(input_ply)
        print(f"[Poisson] Input point cloud: {len(pcd.points):,} points")

    if not pcd.has_normals():
        num_points = len(pcd.points)
        estimated_seconds = (num_points / 100000)*10
        estimated_minutes = estimated_seconds / 60.0

        if estimated_minutes > 1:
            zaman_bilgisi = f"~{estimated_minutes:.1f} dakika"
        else:
            zaman_bilgisi = f"~{estimated_seconds:.0f} saniye"
            
        print(f"[Poisson] Estimating normals... (Nokta: {num_points:,} | Beklenen süre: {zaman_bilgisi})")
        
        start_time = time.time()

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(80)

        end_time = time.time()
        gercek_sure = end_time - start_time
        
        if gercek_sure > 60:
            print(f"[Poisson] Normal hesaplaması tamamlandı! (Gerçekleşen süre: {gercek_sure / 60.0:.1f} dk)")
        else:
            print(f"[Poisson] Normal hesaplaması tamamlandı! (Gerçekleşen süre: {gercek_sure:.1f} sn)")


    print(f"[Poisson] Running reconstruction (depth={depth})...")
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)

    print(f"[Poisson] Raw mesh: {len(poisson_mesh.vertices):,} vertices, "
          f"{len(poisson_mesh.triangles):,} faces")

    if density_threshold > 0:
        dens  = np.asarray(densities)
        thresh = np.quantile(dens, density_threshold)
        poisson_mesh.remove_vertices_by_mask(dens < thresh)
        print(f"[Poisson] After density filter (q={density_threshold}): "
              f"{len(poisson_mesh.vertices):,} vertices, "
              f"{len(poisson_mesh.triangles):,} faces")

    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_non_manifold_edges()

    out_path = output_dir / "poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_path), poisson_mesh)
    print(f"[Poisson] Saved -> {out_path}")

    np.save(output_dir / "poisson_densities.npy", np.asarray(densities))
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",             required=True)
    parser.add_argument("--output_dir",        required=True)
    parser.add_argument("--depth",             type=int,   default=9)
    parser.add_argument("--scale",             type=float, default=1.1)
    parser.add_argument("--density_threshold", type=float, default=0.01)
    args = parser.parse_args()

    poisson_from_ply(args.input, args.output_dir,
                     args.depth, args.scale,
                     density_threshold=args.density_threshold)