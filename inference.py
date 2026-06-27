"""
Turf Detection — Inference Script
MobileSAM Fine-tuned on Aerial Imagery

Usage:
    python inference.py --image input_image.tif
    python inference.py --image input_image.tif --output ./results --conf 0.35
    python inference.py --input ./images/                     # batch mode
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from shapely.geometry import shape as shp_shape
from rasterio.features import shapes as rio_shapes


# ── CONFIG ────────────────────────────────────────────────────────────
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE   = 1024
TILE_INF   = 640
STRIDE_INF = 320
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
SAM_CKPT    = os.path.join(WEIGHTS_DIR, 'mobile_sam.pt')
BEST_CKPT   = os.path.join(WEIGHTS_DIR, 'best.pt')


# ── MODEL ─────────────────────────────────────────────────────────────
class TurfSAM(nn.Module):
    def __init__(self, ckpt=SAM_CKPT):
        super().__init__()
        from mobile_sam import sam_model_registry
        self.sam = sam_model_registry['vit_t'](checkpoint=ckpt)
        for p in self.sam.image_encoder.parameters():  p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters(): p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():   p.requires_grad = True

    def _grid_prompts(self, B, H, W, device):
        xs = torch.linspace(0, W - 1, 16, device=device)
        ys = torch.linspace(0, H - 1, 16, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pts = torch.stack([gx.flatten(), gy.flatten()], -1).unsqueeze(0).expand(B, -1, -1)
        lbl = torch.ones(B, pts.shape[1], device=device)
        return pts, lbl

    def forward(self, images):
        B, C, H, W = images.shape
        all_masks = []
        for b in range(B):
            img = images[b:b+1]
            with torch.no_grad():
                emb = self.sam.image_encoder(img)
                pts = self._grid_prompts(1, H, W, img.device)
                sp, dp = self.sam.prompt_encoder(points=pts, boxes=None, masks=None)
            masks, _ = self.sam.mask_decoder(
                image_embeddings=emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False,
            )
            all_masks.append(masks)
        out = torch.cat(all_masks, dim=0)
        return F.interpolate(out, (H, W), mode='bilinear', align_corners=False)


# ── HELPERS ───────────────────────────────────────────────────────────
def to_uint8(arr):
    """Normalize any dtype aerial image to uint8 for model input."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def preprocess(tile):
    """Resize tile to IMG_SIZE and normalize for SAM."""
    t = torch.from_numpy(cv2.resize(tile, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict_full(model, img, conf=0.35):
    """Sliding window inference over full image."""
    H, W  = img.shape[:2]
    full  = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)

    def starts(size, stride, dim):
        s = list(range(0, dim - size + 1, stride))
        if not s or s[-1] + size < dim:
            s.append(max(0, dim - size))
        return s

    for r in starts(TILE_INF, STRIDE_INF, H):
        for c in starts(TILE_INF, STRIDE_INF, W):
            r2, c2 = min(r + TILE_INF, H), min(c + TILE_INF, W)
            tile = img[r:r2, c:c2]
            if tile.shape[0] < TILE_INF or tile.shape[1] < TILE_INF:
                tile = np.pad(tile, ((0, TILE_INF - tile.shape[0]),
                                     (0, TILE_INF - tile.shape[1]), (0, 0)))
            prob = torch.sigmoid(model(preprocess(tile))).squeeze().cpu().numpy()
            prob = cv2.resize(prob, (c2 - c, r2 - r))
            full[r:r2, c:c2] += prob
            count[r:r2, c:c2] += 1

    full /= np.maximum(count, 1)
    binary = (full > conf).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary, full


def save_outputs(stem, img, binary, prob, meta, out_dir):
    """Save overlay PNG, mask GeoTIFF, and GeoJSON."""
    os.makedirs(out_dir, exist_ok=True)

    # ── overlay PNG ──
    scale = min(1.0, 1200 / max(img.shape[:2]))
    H2, W2 = int(img.shape[0] * scale), int(img.shape[1] * scale)
    img_s  = cv2.resize(img,    (W2, H2))
    bin_s  = cv2.resize(binary, (W2, H2), interpolation=cv2.INTER_NEAREST)
    prob_s = cv2.resize(prob,   (W2, H2))

    ov = img_s.astype(np.float32)
    g  = np.zeros_like(ov)
    g[bin_s == 1] = [0, 200, 0]
    ov = (ov * 0.65 + g * 0.35).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(bin_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, contours, -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(img_s);  axes[0].set_title('Original');           axes[0].axis('off')
    axes[1].imshow(prob_s, cmap='RdYlGn', vmin=0, vmax=1);
    axes[1].set_title('Confidence Map');                              axes[1].axis('off')
    axes[2].imshow(ov);     axes[2].set_title('Detection Overlay');  axes[2].axis('off')
    cov = 100 * binary.mean()
    plt.suptitle(f'{stem} — Turf Coverage: {cov:.2f}%', fontsize=12)
    plt.tight_layout()
    overlay_path = os.path.join(out_dir, f'{stem}_overlay.png')
    plt.savefig(overlay_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Overlay  → {overlay_path}')

    # ── mask GeoTIFF ──
    profile = {
        'driver': 'GTiff', 'dtype': 'uint8',
        'width': meta['width'], 'height': meta['height'],
        'count': 1, 'compress': 'lzw',
    }
    if meta['crs']:
        profile['crs']       = meta['crs']
        profile['transform'] = meta['transform']
    mask_path = os.path.join(out_dir, f'{stem}_mask.tif')
    with rasterio.open(mask_path, 'w', **profile) as dst:
        dst.write(binary[np.newaxis])
    print(f'  Mask     → {mask_path}')

    # ── GeoJSON ──
    MIN_PX = 200
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(binary)
    cv2.drawContours(clean, [c for c in cnts if cv2.contourArea(c) > MIN_PX], -1, 1, -1)
    polys = [shp_shape(g) for g, v in
             rio_shapes(clean, mask=clean, transform=meta['transform']) if v == 1]

    if polys:
        gdf = gpd.GeoDataFrame(
            {'class':   ['turf'] * len(polys),
             'area_m2': [round(p.area, 2) for p in polys]},
            geometry=polys,
            crs=meta['crs'] if meta['crs'] else 'EPSG:4326',
        )
        geojson_path = os.path.join(out_dir, f'{stem}_turf.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f'  GeoJSON  → {geojson_path}  ({len(polys)} polygons, coverage {cov:.1f}%)')
    else:
        print(f'  GeoJSON  → no turf detected above threshold')


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Turf Detection Inference')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Path to a single .tif / .tiff file')
    group.add_argument('--input', help='Folder containing .tif / .tiff files')
    parser.add_argument('--output', default='./output', help='Output folder (default: ./output)')
    parser.add_argument('--conf',   type=float, default=0.35, help='Confidence threshold (default: 0.35)')
    parser.add_argument('--sam_ckpt',  default=SAM_CKPT,  help='Path to mobile_sam.pt')
    parser.add_argument('--best_ckpt', default=BEST_CKPT, help='Path to best.pt')
    args = parser.parse_args()

    # ── validate checkpoints ──
    for p in [args.sam_ckpt, args.best_ckpt]:
        if not os.path.exists(p):
            print(f'ERROR: checkpoint not found: {p}')
            print('Place mobile_sam.pt and best.pt in the weights/ folder, or pass --sam_ckpt / --best_ckpt')
            sys.exit(1)

    # ── load model ──
    print(f'Loading model on {DEVICE}...')
    model = TurfSAM(ckpt=args.sam_ckpt).to(DEVICE)
    ckpt  = torch.load(args.best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded checkpoint — epoch {ckpt["epoch"]}, val IoU {ckpt["val_iou"]:.4f}\n')

    # ── collect images ──
    if args.image:
        images = [args.image]
    else:
        exts   = ('.tif', '.tiff')
        images = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        )
        if not images:
            print(f'ERROR: no .tif / .tiff files found in {args.input}')
            sys.exit(1)

    print(f'Running inference on {len(images)} image(s)...\n')

    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f'[{stem}]')

        with rasterio.open(path) as src:
            img_arr = src.read()[:3].transpose(1, 2, 0)
            meta    = {
                'crs':       src.crs,
                'transform': src.transform,
                'width':     src.width,
                'height':    src.height,
            }
        print(f'  Size: {meta["width"]}x{meta["height"]} | CRS: {meta["crs"]}')

        img_arr = to_uint8(img_arr)
        binary, prob = predict_full(model, img_arr, conf=args.conf)
        save_outputs(stem, img_arr, binary, prob, meta, args.output)
        print()

    print(f'Done. All outputs saved to {args.output}/')


if __name__ == '__main__':
    main()
