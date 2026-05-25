print("Adım 1: Kod başlatıldı, kütüphaneler yükleniyor...")
import os
import open3d as o3d

print("Adım 2: Kütüphaneler başarıyla yüklendi!")

# Pipeline'ın dosyayı nereye kaydettiyse o yolu bulalım
yollar = [
    "output/colmap/sparse/0/points3D.txt",
    "Output_Models/colmap/sparse/0/points3D.txt",
    "Output_Models/sparse/0/points3D.txt"
]

dogru_yol = None
for yol in yollar:
    if os.path.exists(yol):
        dogru_yol = yol
        break

if dogru_yol is None:
    print("HATA: points3D.txt dosyası hiçbir klasörde bulunamadı!")
    print("Muhtemelen Adım 1 (COLMAP) düzgün çalışmadı veya çöktü.")
    exit()

print(f"Adım 3: Dosya bulundu -> {dogru_yol}")
print("Adım 4: Noktalar okunuyor, lütfen bekleyin...")

pts = []
cols = []
with open(dogru_yol, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        data = line.split()
        pts.append([float(data[1]), float(data[2]), float(data[3])])
        cols.append([float(data[4])/255, float(data[5])/255, float(data[6])/255])

print(f"Adım 5: Tam {len(pts)} adet nokta okundu. 3D Pencere açılıyor...")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(cols)

# Eksenleri ekleyelim (Kırmızı=X, Yeşil=Y, Mavi=Z)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

o3d.visualization.draw_geometries([pcd, axes], window_name="COLMAP Kontrol")
print("İşlem tamamlandı, pencere kapatıldı.")