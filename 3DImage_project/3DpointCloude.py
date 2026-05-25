import open3d as o3d

# Oluşan .ply dosyasının tam yolunu buraya yaz
dosya_yolu = "Output_Models/tsdf/tsdf_mesh.ply" # veya poisson_mesh.ply

print("[INFO] Nokta bulutu yükleniyor...")
pcd = o3d.io.read_point_cloud(dosya_yolu)

# Ekrana 3D pencereyi bas
o3d.visualization.draw_geometries([pcd], 
                                  window_name="3D Nokta Bulutu",
                                  width=1024, 
                                  height=768)