import open3d as o3d

path = "OUTPUT_MODELS/tsdf/tsdf_mesh.ply"

print(f"[View] Loading: {path}")

# Try as point cloud first (incremental fusion output)
pcd = o3d.io.read_point_cloud(path)

if len(pcd.points) > 0:
    print(f"[View] Point cloud: {len(pcd.points):,} points")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.03)
    print(f"[View] After filtering: {len(pcd.points):,} points")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Viewer",
        width=1280, height=960,
        point_show_normal=False,
    )
else:
    # Fallback: load as mesh
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    print(f"[View] Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Mesh Viewer",
        width=1280, height=960,
    )