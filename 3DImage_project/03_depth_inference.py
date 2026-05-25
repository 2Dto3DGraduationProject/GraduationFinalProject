"""
STEP 3: DepthAnything - Per-Image Monocular Depth Estimation
-------------------------------------------------------------
Output: per-image depth maps (.npy float32 + .png uint16 visualization)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from depth_anything.dpt import DepthAnything


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(encoder, checkpoint, device):
    config = ENCODER_CONFIGS[encoder]
    model  = DepthAnything(config)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"[Depth] Checkpoint loaded: {checkpoint}")
    else:
        model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14")
        print(f"[Depth] HuggingFace model: depth_anything_{encoder}14")
    model.to(device).eval()
    return model


def build_transform(input_size=518):
    return transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def run_depth(model, image_path, transform, device, orig_h, orig_w):
    img   = Image.open(image_path).convert("RGB")
    inp   = transform(img).unsqueeze(0).to(device)
    depth = model(inp)
    if depth.dim() == 3:
        depth = depth[0]
    depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0),
                          size=(orig_h, orig_w),
                          mode="bilinear", align_corners=True).squeeze()
    return depth.cpu().numpy().astype(np.float32)


def run_inference(image_dir, output_dir, encoder, checkpoint, input_size, device_str):
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device(device_str)
    model     = load_model(encoder, checkpoint, device)
    transform = build_transform(input_size)

    image_paths = sorted(p for p in image_dir.iterdir()
                         if p.suffix.lower() in SUPPORTED_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir}")

    ref = Image.open(image_paths[0]).convert("RGB")
    orig_h, orig_w = ref.height, ref.width
    print(f"[Depth] {len(image_paths)} images | {orig_w}x{orig_h} | encoder={encoder}")

    for i, img_path in enumerate(image_paths):
        depth = run_depth(model, img_path, transform, device, orig_h, orig_w)

        stem = img_path.stem
        np.save(output_dir / f"{stem}_depth.npy", depth)

        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            vis = ((depth - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
        else:
            vis = np.zeros_like(depth, dtype=np.uint16)
        Image.fromarray(vis).save(output_dir / f"{stem}_depth.png")

        print(f"[Depth] [{i+1}/{len(image_paths)}] {img_path.name} "
              f"range=[{depth.min():.4f}, {depth.max():.4f}]")

    print(f"[Depth] Done -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder",    default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    run_inference(args.image_dir, args.output_dir, args.encoder,
                  args.checkpoint, args.input_size, args.device)