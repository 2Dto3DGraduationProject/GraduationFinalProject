import argparse
import subprocess
import sys
from pathlib import Path


def log_step(step, name):
    print(f"\n[STEP {step}] {name}\n{'='*50}", flush=True)


def run_python(script, args):
    cmd    = [sys.executable, str(script)] + [str(a) for a in args]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="3D Asset Pipeline")
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--dedup_cell", type=float, default=0.008,
                        help="Dedup voxel cell size in meters (default: 0.008)")
    parser.add_argument("--depth_min",  type=float, default=0.1)
    parser.add_argument("--depth_max",  type=float, default=10.0)
    parser.add_argument("--scale",      type=float, default=0.01)
    args, _ = parser.parse_known_args()

    out = Path(args.output_dir)
    sd  = Path(__file__).resolve().parent   # scripts dir

    cfg = argparse.Namespace(
        image_dir        = Path(args.image_dir),
        colmap_dir       = out / "colmap",
        poses_dir        = out / "poses",
        depth_dir        = out / "depth_relative",
        depth_metric_dir = out / "depth_metric",
        masks_dir        = out / "masks",
        fusion_dir       = out / "tsdf",        # output of step 5
        poisson_dir      = out / "poisson",
        texture_dir      = out / "textured",
    )

    steps = [
        (1, sd / "01_colmap.py", [
            cfg.image_dir, cfg.colmap_dir
        ]),
        (2, sd / "02_parse_colmap.py", [
            "--colmap_dir", cfg.colmap_dir / "sparse" / "0",
            "--output_dir", cfg.poses_dir
        ]),
        (3, sd / "03_depth_inference.py", [
            "--image_dir",  cfg.image_dir,
            "--output_dir", cfg.depth_dir
        ]),
        ("3.5", sd / "03.5_segmentation.py", [
            "--image_dir",  cfg.image_dir,
            "--output_dir", cfg.masks_dir
        ]),
        (4, sd / "04_scale_normalization.py", [
            "--colmap_dir", cfg.colmap_dir / "sparse" / "0",
            "--depth_dir",  cfg.depth_dir,
            "--output_dir", cfg.depth_metric_dir
        ]),
        (5, sd / "05_incremental_fusion.py", [
            "--poses_npz",  cfg.poses_dir  / "poses.npz",
            "--image_dir",  cfg.image_dir,
            "--depth_dir",  cfg.depth_metric_dir,
            "--mask_dir",   cfg.masks_dir,
            "--output_dir", cfg.fusion_dir,
            "--dedup_cell", args.dedup_cell,
            "--depth_min",  args.depth_min,
            "--depth_max",  args.depth_max,
            "--scale",      args.scale,
        ]),
        (6, sd / "06_poisson_mesh.py", [
            "--input",      cfg.fusion_dir / "tsdf_mesh.ply",
            "--output_dir", cfg.poisson_dir
        ]),
        (7, sd / "07_texture_mesh.py", [
            "--mesh_ply",   cfg.poisson_dir / "poisson_mesh.ply",
            "--poses_npz",  cfg.poses_dir   / "poses.npz",
            "--image_dir",  cfg.image_dir,
            "--output_dir", cfg.texture_dir
        ]),
    ]

    for step_num, script, script_args in steps:
        log_step(step_num, script.name)
        if not run_python(script, script_args):
            print(f"[PIPELINE] ERROR: Step {step_num} failed. Aborting.")
            break
    else:
        print("\n[PIPELINE] All steps complete.")


if __name__ == "__main__":
    main()