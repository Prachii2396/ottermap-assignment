import os
import sys
import tempfile
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import cv2
import json
import gradio as gr
from huggingface_hub import hf_hub_download
from shapely.geometry import shape as shp_shape, mapping
from rasterio.features import shapes as rio_shapes

# ── CONFIG ────────────────────────────────────────────────────────────
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE   = 1024
TILE_INF   = 640
STRIDE_INF = 320
MAX_DIM    = 2000   # cap input size for CPU speed
REPO_ID    = 'prachi2396/turf-detection-weights'

# ── DOWNLOAD WEIGHTS ──────────────────────────────────────────────────
print('Downloading weights...')
SAM_CKPT  = hf_hub_download(repo_id=REPO_ID, filename='mobile_sam.pt')
BEST_CKPT = hf_hub_download(repo_id=REPO_ID, filename='best.pt')
print('Weights ready.')

# ── MODEL ─────────────────────────────────────────────────────────────
class TurfSAM(nn.Module):
    def __init__(self, ckpt):
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

print('Loading model...')
model = TurfSAM(SAM_CKPT).to(DEVICE)
ckpt  = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'Model ready — epoch {ckpt["epoch"]}, val IoU {ckpt["val_iou"]:.4f}')

# ── HELPERS ───────────────────────────────────────────────────────────
def to_uint8(arr):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

def preprocess(tile):
    t = torch.from_numpy(cv2.resize(tile, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_full(img, conf=0.35):
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

# ── INFERENCE FUNCTION ────────────────────────────────────────────────
def run_inference(tif_file, conf_threshold):
    if tif_file is None:
        return None, None, "No file uploaded."

    try:
        with rasterio.open(tif_file.name) as src:
            img_arr  = src.read()[:3].transpose(1, 2, 0)
            crs       = src.crs
            transform = src.transform

        # cap size for CPU
        H, W = img_arr.shape[:2]
        if max(H, W) > MAX_DIM:
            scale   = MAX_DIM / max(H, W)
            new_H   = int(H * scale)
            new_W   = int(W * scale)
            img_arr = cv2.resize(img_arr, (new_W, new_H))
            info    = f"Image resized from {W}x{H} to {new_W}x{new_H} for CPU inference."
        else:
            info = f"Image size: {W}x{H}"

        img_arr = to_uint8(img_arr)
        binary, prob = predict_full(img_arr, conf=conf_threshold)

        # ── overlay ──
        ov = img_arr.astype(np.float32)
        g  = np.zeros_like(ov)
        g[binary == 1] = [0, 200, 0]
        ov = (ov * 0.65 + g * 0.35).clip(0, 255).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
        coverage = 100 * binary.mean()

        # ── geojson ──
        MIN_PX = 200
        clean  = np.zeros_like(binary)
        cv2.drawContours(clean, [c for c in cnts if cv2.contourArea(c) > MIN_PX], -1, 1, -1)

        from rasterio.transform import from_bounds
        if transform is None or str(transform) == str(rasterio.transform.IDENTITY):
            h, w = binary.shape
            transform = from_bounds(0, 0, w, h, w, h)

        polys = [shp_shape(g) for g, v in
                 rio_shapes(clean, mask=clean, transform=transform) if v == 1]

        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(p),
                    "properties": {"class": "turf", "area": round(p.area, 2)}
                }
                for p in polys
            ]
        }
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_turf.geojson', mode='w')
        json.dump(geojson_data, tmp)
        tmp.close()

        status = (
            f"Coverage: {coverage:.1f}%  |  "
            f"Polygons: {len(polys)}  |  "
            f"{info}  |  "
            f"Model: epoch {ckpt['epoch']}, val IoU {ckpt['val_iou']:.4f}"
        )
        return ov, tmp.name, status

    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ── GRADIO UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="Turf Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Turf Detection from Aerial Imagery
    **MobileSAM fine-tuned on aerial GeoTIFF imagery**
    Upload a `.tif` or `.tiff` aerial image to detect turf/grass areas.
    Outputs a visual overlay and a downloadable GeoJSON file.
    > Note: Running on CPU — inference takes 2 to 5 minutes depending on image size.
    """)

    with gr.Row():
        with gr.Column():
            tif_input   = gr.File(label="Upload Aerial Image (.tif / .tiff)", file_types=['.tif', '.tiff'])
            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.05,
                                    label="Confidence Threshold (lower = more detections)")
            run_btn     = gr.Button("Run Inference", variant="primary")

        with gr.Column():
            overlay_out = gr.Image(label="Detection Overlay (green = turf)")
            geojson_out = gr.File(label="Download GeoJSON")
            status_out  = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=run_inference,
        inputs=[tif_input, conf_slider],
        outputs=[overlay_out, geojson_out, status_out]
    )

    gr.Markdown("""
    ---
    Built by Prachi Priyadarshini | IIIT Naya Raipur | Ottermap 72-Hour Challenge
    """)

demo.launch()
