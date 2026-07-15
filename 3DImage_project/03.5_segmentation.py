"""
STEP 3.5: Segmentation - Ön plan Segmentasyonu
-----------------------------------------------
Değişken arka planlar için SAM (Segment Anything Model) kullan.
SAM yüklü değilse, Otsu threshold'a fallback yap.

Kurulum:
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install opencv-python-headless
"""

import argparse
import os
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_erosion, binary_dilation

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("[WARNING] SAM yüklü değil. Kurulum: pip install git+https://github.com/facebookresearch/segment-anything.git")


def run_sam_segmentation(image_dir, output_dir, model_type="vit_b", device="cuda"):
    """
    Segment Anything Model (SAM) ile otomatik ön plan segmentasyonu.
    
    SAM'ın avantajları:
    - Arka plan rengi ne olursa olsun çalışıyor (generalist model)
    - Prompt olmadan otomatik segmentation yapabiliyor
    - Çok yüksek kalite maskeleme
    
    Args:
        model_type: "vit_h" (en iyi, yavaş), "vit_l", "vit_b" (hızlı), "mobile_sam" (ultra hızlı)
    """
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SAM modelini yükle
    print(f"[Segmentation] SAM model yükleniyor ({model_type})...")
    print(f"             (İlk defa: ~300MB - 2.5GB indirecek)")
    
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    if model_type == "mobile_sam":
        print("[HATA] MobileSAM için ayrı bir kütüphane gerekir. vit_b'ye geçiliyor.")
        model_type = "vit_b"
        
    ckpt_path = f"sam_{model_type}.pth"
    
    # Model ağırlıkları yoksa internetten indir
    if not os.path.exists(ckpt_path):
        print(f"[Segmentation] SAM {model_type} ağırlıkları indiriliyor...")
        print(f"               Bu işlem dosya boyutuna göre sürebilir.")
        urllib.request.urlretrieve(checkpoint_urls[model_type], ckpt_path)
        print("[Segmentation] İndirme tamamlandı!")

    # Modeli indirilen dosyadan (checkpoint) yükle
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)

    if device == "cuda" and torch.cuda.is_available():
        sam.to(device=device)
        print(f"[Segmentation] Model GPU'ya yüklendi")
    else:
        sam.to(device="cpu")
        print(f"[Segmentation] Model CPU'da çalışıyor (yavaş olabilir)")
        device = "cpu"
    
    # Otomatik mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,        # Grid çözünürlüğü (yüksek = daha detaylı)
        pred_iou_thresh=0.88,      # IoU kalite threshold (yüksek = kaliteli ama az mask)
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
    )

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[Segmentation] SAM segmentasyonu başlıyor... ({len(image_paths)} görsel)")

    for idx, img_path in enumerate(image_paths):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_np.shape[:2]

        print(f"[Segmentation] [{idx+1}/{len(image_paths)}] {img_path.name} - işleniyor...")
        
        # Otomatik maskeleme (0.5 - 2 saniye per görüntü, GPU hızlı)
        masks = mask_generator.generate(img_np)
        
        if not masks:
            print(f"             UYARI: SAM mask bulunamadı, Otsu threshold'a fallback")
            # Fallback: Otsu automatic thresholding
            gray = np.array(Image.open(img_path).convert("L"))
            hist, bins = np.histogram(gray, 256)
            total = gray.size
            
            best_thresh, best_var = 0, 0
            for t in range(1, 255):
                w0 = np.sum(hist[:t]) / total
                w1 = 1 - w0
                if w0 == 0 or w1 == 0:
                    continue
                mu0 = np.sum(np.arange(t) * hist[:t]) / (w0 * total) if w0 > 0 else 0
                mu1 = np.sum(np.arange(t, 256) * hist[t:]) / (w1 * total) if w1 > 0 else 0
                var = w0 * w1 * (mu0 - mu1) ** 2
                if var > best_var:
                    best_var, best_thresh = var, t
            
            mask_np = (gray > best_thresh).astype(np.uint8) * 255
        else:
            # En büyük maskeyi seç (genellikle ana nesne)
            # Eğer birden fazla nesne varsa, hepsini birleştir
            all_masks = np.zeros((h, w), dtype=bool)
            for mask_dict in masks:
                all_masks |= mask_dict["segmentation"]
            
            mask_np = all_masks.astype(np.uint8) * 255

        # Edge artifact'larını temizle: erosion → dilation
        mask_binary = mask_np > 127
        mask_binary = binary_erosion(mask_binary, iterations=2)
        mask_binary = binary_dilation(mask_binary, iterations=1)
        mask_np = mask_binary.astype(np.uint8) * 255

        # Kaydet
        out_mask = Image.fromarray(mask_np)
        out_mask.save(output_dir / f"{img_path.stem}_mask.png")

        print(f"             ✓ Maske kaydedildi")

    print(f"\n[Segmentation] BAŞARILI - {len(image_paths)} maske oluşturuldu")
    print(f"             Çıktı: {output_dir}")


def run_otsu_threshold(image_dir, output_dir):
    """
    Fallback: Otsu Automatic Thresholding
    SAM'ın yüklü olmadığı veya çalışamadığı durumlar için.
    
    Otsu, görüntüdeki histogram'ı analiz ederek
    en iyi threshold değerini otomatik bulur.
    """
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[Segmentation] Otsu Threshold masking başlıyor... ({len(image_paths)} görsel)")

    for idx, img_path in enumerate(image_paths):
        gray = np.array(Image.open(img_path).convert("L"))
        
        # Otsu Threshold Hesaplaması
        hist, bins = np.histogram(gray, 256)
        total = gray.size
        
        best_thresh, best_var = 0, 0
        for t in range(1, 255):
            w0 = np.sum(hist[:t]) / total
            w1 = 1 - w0
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = np.sum(np.arange(t) * hist[:t]) / (w0 * total) if w0 > 0 else 0
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / (w1 * total) if w1 > 0 else 0
            var = w0 * w1 * (mu0 - mu1) ** 2
            
            if var > best_var:
                best_var, best_thresh = var, t
        
        mask_np = (gray > best_thresh).astype(np.uint8) * 255
        
        # Küçük gürültüyü temizle: erosion + dilation
        mask_binary = binary_erosion(mask_np > 127, iterations=1)
        mask_binary = binary_dilation(mask_binary, iterations=1)
        mask_np = mask_binary.astype(np.uint8) * 255
        
        out_mask = Image.fromarray(mask_np)
        out_mask.save(output_dir / f"{img_path.stem}_mask.png")

        print(f"[Segmentation] [{idx+1}/{len(image_paths)}] {img_path.name} (threshold={best_thresh})")

    print(f"\n[Segmentation] {len(image_paths)} Otsu maskesi oluşturuldu -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ön plan segmentasyonu (SAM → Otsu fallback)")
    parser.add_argument("--image_dir",  required=True, help="Girdi görüntü klasörü")
    parser.add_argument("--output_dir", required=True, help="Çıktı maske klasörü")
    parser.add_argument("--model_type", default="vit_b", 
                        choices=["vit_h", "vit_l", "vit_b", "mobile_sam"],
                        help="SAM model boyutu (default: vit_b, hızlı ve kaliteli)")
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"],
                        help="GPU/CPU (default: cuda)")
    parser.add_argument("--use_otsu",   action="store_true",
                        help="Sadece Otsu threshold kullan (SAM yerine)")
    
    args = parser.parse_args()

    if args.use_otsu or not HAS_SAM:
        if not HAS_SAM:
            print("[INFO] SAM yüklü değil → Otsu threshold kullanılıyor")
        run_otsu_threshold(args.image_dir, args.output_dir)
    else:
        run_sam_segmentation(args.image_dir, args.output_dir, args.model_type, args.device)