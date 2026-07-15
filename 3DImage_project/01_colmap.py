import sys
import os
import subprocess
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


image_dir = sys.argv[1]
out_dir   = Path(sys.argv[2])
db        = out_dir / "database.db"
sparse    = out_dir / "sparse"

os.makedirs(sparse, exist_ok=True)

print("[COLMAP] Feature extraction...")
run(f"colmap feature_extractor "
    f"--database_path {db} "
    f"--image_path {image_dir} "
    f"--ImageReader.single_camera 1 "
    f"--SiftExtraction.max_num_features 8300")

print("[COLMAP] Sequential matching...")
run(f"colmap sequential_matcher "
    f"--database_path {db} "
    f"--SequentialMatching.overlap 12")

print("[COLMAP] Exhaustive matching...")
run(f"colmap exhaustive_matcher --database_path {db}")

print("[COLMAP] Mapper...")
run(f"colmap mapper --database_path {db} --image_path {image_dir} --output_path {sparse}")

print("[COLMAP] Exporting TXT model...")
run(f"colmap model_converter "
    f"--input_path {sparse}/0 "
    f"--output_path {sparse}/0 "
    f"--output_type TXT")

print("[COLMAP] Done.")