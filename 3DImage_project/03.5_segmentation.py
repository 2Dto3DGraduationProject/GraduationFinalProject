import argparse
from pathlib import Path
from PIL import Image
from rembg import remove, new_session


def run_rembg(image_dir, output_dir):
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"[Segmentation] {len(image_paths)} images | model=u2net")

    session = new_session("u2net")

    for i, img_path in enumerate(image_paths):
        img  = Image.open(img_path).convert("RGB")
        mask = remove(img, session=session, only_mask=True)
        out  = output_dir / f"{img_path.stem}_mask.png"
        mask.save(out)
        print(f"[Segmentation] [{i+1}/{len(image_paths)}] {img_path.name} -> {out.name}")

    print(f"[Segmentation] Done -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_rembg(args.image_dir, args.output_dir)