# Generated from: ottermap-assignment-train.ipynb
# Converted at: 2026-06-27T17:35:58.401Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# # check GPU


import torch
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
else:
    print('⚠️  NO GPU DETECTED — Go to Settings → Accelerator → GPU T4 x2 and restart')

# # Install dependencies


%%capture
!pip install rasterio geopandas shapely albumentations timm supervision -q
!pip install git+https://github.com/ChaoningZhang/MobileSAM.git -q

# # Download MobileSAM Weights


import os
os.makedirs('/kaggle/working/weights', exist_ok=True)

if not os.path.exists('/kaggle/working/weights/mobile_sam.pt'):
    print('Downloading MobileSAM weights (~38MB)...')
    !wget -q -P /kaggle/working/weights/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
    print('✅ Downloaded')
else:
    print('✅ Already downloaded')

!ls -lh /kaggle/working/weights/

import glob

# Find your uploaded files — searches all nested subfolders
tiffs    = sorted(glob.glob('/kaggle/input/**/*.tiff', recursive=True) + 
                  glob.glob('/kaggle/input/**/*.tif',  recursive=True))

# Only from GeoJSON folder, not ShapeFile
geojsons = sorted(glob.glob('/kaggle/input/**/GeoJSON/*.geojson', recursive=True))

print('TIFFs found:')
for f in tiffs: print(' ', f)

print('\nGeoJSONs found:')
for f in geojsons: print(' ', f)

if not tiffs or not geojsons:
    print('\n⚠️  FILES NOT FOUND — Check your dataset structure')
else:
    print(f'\n✅ Found {len(tiffs)} TIFFs and {len(geojsons)} GeoJSONs — ready to proceed')

# # project setup


import os, shutil, glob

# Create folder structure
DIRS = [
    '/kaggle/working/data/raw',
    '/kaggle/working/data/masks',
    '/kaggle/working/data/tiles/images/train',
    '/kaggle/working/data/tiles/images/val',
    '/kaggle/working/data/tiles/masks/train',
    '/kaggle/working/data/tiles/masks/val',
    '/kaggle/working/results/overlays',
    '/kaggle/working/results/predictions',
    '/kaggle/working/results/geojson',
    '/kaggle/working/results/verification',
    '/kaggle/working/weights',
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Copy TIFFs — searches ALL subfolders recursively
for f in glob.glob('/kaggle/input/**/*.tiff', recursive=True) + \
         glob.glob('/kaggle/input/**/*.tif',  recursive=True):
    dst = f'/kaggle/working/data/raw/{os.path.basename(f)}'
    shutil.copy(f, dst)
    print(f'✅ Copied TIFF: {os.path.basename(f)}')

# Copy GeoJSONs from GeoJSON folder only (not ShapeFile folder)
for f in glob.glob('/kaggle/input/**/GeoJSON/*.geojson', recursive=True):
    dst = f'/kaggle/working/data/raw/{os.path.basename(f)}'
    shutil.copy(f, dst)
    print(f'✅ Copied GeoJSON: {os.path.basename(f)}')

print('\nFiles in data/raw:')
for f in sorted(os.listdir('/kaggle/working/data/raw')):
    print(' ', f)

import rasterio

DATA_DIR = '/kaggle/working/data/raw'

for img_id in ['1', '2', '3']:
    path = f'{DATA_DIR}/{img_id}.tiff'
    with rasterio.open(path) as src:
        print(f"\n=== Image {img_id} ===")
        print(f"  Size:      {src.width} × {src.height} px")
        print(f"  CRS:       {src.crs}")         # None = not georeferenced
        print(f"  Transform: {src.transform}")    # identity = no real coords
        print(f"  Bounds:    {src.bounds}")

# Tune these per-image — start here and adjust based on Cell 8 output
BUFFER = {
    '1': 0.15,   # try 0.10, 0.15, 0.20, 0.25
    '2': 0.15,
    '3': 0.15,
}

def georeference_tiff(image_id):
    tiff_path    = f'{DATA_DIR}/{image_id}.tiff'
    geojson_path = f'{DATA_DIR}/{image_id}.geojson'
    out_path     = f'{DATA_DIR}/{image_id}_georef.tiff'

    gdf = gpd.read_file(geojson_path)
    west, south, east, north = gdf.total_bounds

    buf = BUFFER[image_id]           # ← per-image now
    lon_span = east - west
    lat_span = north - south
    west  -= lon_span * buf
    east  += lon_span * buf
    south -= lat_span * buf
    north += lat_span * buf
    # rest stays the same...

# # georeferencing TIFFs


import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Add these at the top of Cell 8
DATA_DIR = '/kaggle/working/data/raw'
MASK_DIR = '/kaggle/working/data/masks'

# rest of Cell 8 continues below as normal...
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
# ...
IMAGE_IDS = ['1', '2', '3']

def georeference_tiff(image_id):
    tiff_path    = f'{DATA_DIR}/{image_id}.tiff'
    geojson_path = f'{DATA_DIR}/{image_id}.geojson'
    out_path     = f'{DATA_DIR}/{image_id}_georef.tiff'

    # Get real-world bounds from GeoJSON annotations
    gdf = gpd.read_file(geojson_path)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    west, south, east, north = bounds

    # 10% buffer so annotations don't sit at very edge of image
    lon_span = east - west
    lat_span = north - south
    buf = 0.10
    west  -= lon_span * buf
    east  += lon_span * buf
    south -= lat_span * buf
    north += lat_span * buf

    with rasterio.open(tiff_path) as src:
        height, width = src.shape
        data    = src.read()
        profile = src.profile.copy()

    transform = from_bounds(west, south, east, north, width, height)
    profile.update({'crs': CRS.from_epsg(4326), 'transform': transform})

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)

    gsd = (east - west) * 111320 / width
    print(f'[{image_id}] ✅ Georeferenced  |  Size: {width}×{height}  |  Est. GSD: {gsd:.3f} m/px')

for img_id in IMAGE_IDS:
    georeference_tiff(img_id)

print('\n✅ Step 1 complete')

# # Burn Masks


from rasterio.features import rasterize

MASK_DIR = '/kaggle/working/data/masks'

def burn_mask(image_id):
    georef_path  = f'{DATA_DIR}/{image_id}_georef.tiff'
    geojson_path = f'{DATA_DIR}/{image_id}.geojson'
    out_path     = f'{MASK_DIR}/{image_id}_mask.tiff'

    with rasterio.open(georef_path) as src:
        transform = src.transform
        crs       = src.crs
        height, width = src.shape

    gdf = gpd.read_file(geojson_path)
    shapes_list = [(geom, 1) for geom in gdf.geometry if geom is not None]

    mask = rasterize(
        shapes=shapes_list,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )

    profile = {
        'driver': 'GTiff', 'dtype': np.uint8,
        'width': width, 'height': height, 'count': 1,
        'crs': crs, 'transform': transform, 'compress': 'lzw',
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask[np.newaxis])

    coverage = 100.0 * mask.sum() / (height * width)
    print(f'[{image_id}] ✅ Mask saved  |  Turf coverage: {coverage:.2f}%  |  Turf px: {mask.sum():,}')

for img_id in IMAGE_IDS:
    burn_mask(img_id)

print('\n✅ Step 2 complete')

# # Verify Alignments


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from rasterio.features import rasterize
import rasterio

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('Alignment Verification — Green overlay must sit ON grass', fontsize=14)

for i, img_id in enumerate(['1', '2', '3']):
    with rasterio.open(f'{DATA_DIR}/{img_id}_georef.tiff') as src:
        img = src.read()[:3].transpose(1, 2, 0)
    with rasterio.open(f'{MASK_DIR}/{img_id}_mask.tiff') as src:
        mask = src.read(1)

    # Downsample for display
    scale = min(1.0, 1200 / max(img.shape[:2]))
    H2, W2 = int(img.shape[0]*scale), int(img.shape[1]*scale)
    img_s  = cv2.resize(img,  (W2, H2))
    mask_s = cv2.resize(mask, (W2, H2), interpolation=cv2.INTER_NEAREST)

    overlay = img_s.copy().astype(np.float32)
    green = np.zeros_like(overlay)
    green[mask_s == 1] = [0, 200, 0]
    overlay = (overlay * 0.65 + green * 0.35).clip(0,255).astype(np.uint8)

    axes[i][0].imshow(img_s);     axes[i][0].set_title(f'Image {img_id} — Original'); axes[i][0].axis('off')
    axes[i][1].imshow(mask_s, cmap='Greens'); axes[i][1].set_title(f'Mask ({100*mask_s.mean():.1f}% turf)'); axes[i][1].axis('off')
    axes[i][2].imshow(overlay);   axes[i][2].set_title('Overlay (green = turf)'); axes[i][2].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/results/verification/alignment_check.png', dpi=100, bbox_inches='tight')
plt.show()
print('\n✅ Check overlay above — green must sit ON grass areas')

# # Images + Masks into 640*640 patches


from PIL import Image as PILImage
from tqdm.notebook import tqdm

TILE_DIR       = '/kaggle/working/data/tiles'
TILE_SIZE      = 640
STRIDE         = 320       # 50% overlap
MIN_TURF_RATIO = 0.01      # skip tiles with <1% turf
SPLIT = {'1': 'train', '2': 'train', '3': 'val'}

def tile_image(image_id):
    split = SPLIT[image_id]
    img_out  = f'{TILE_DIR}/images/{split}'
    mask_out = f'{TILE_DIR}/masks/{split}'
    os.makedirs(img_out,  exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    with rasterio.open(f'{DATA_DIR}/{image_id}_georef.tiff') as src:
        img_arr = src.read()[:3].transpose(1, 2, 0)   # (H, W, 3)
        H, W   = src.shape

    with rasterio.open(f'{MASK_DIR}/{image_id}_mask.tiff') as src:
        mask_arr = src.read(1)                         # (H, W)

    def starts(size, stride, dim):
        s = list(range(0, dim - size + 1, stride))
        if not s or s[-1] + size < dim:
            s.append(max(0, dim - size))
        return s

    rows = starts(TILE_SIZE, STRIDE, H)
    cols = starts(TILE_SIZE, STRIDE, W)
    saved = skipped = 0

    for r in tqdm(rows, desc=f'Image {image_id} [{split}]'):
        for c in cols:
            img_tile  = img_arr[r:r+TILE_SIZE, c:c+TILE_SIZE]
            mask_tile = mask_arr[r:r+TILE_SIZE, c:c+TILE_SIZE]

            if mask_tile.size == 0:
                continue
            if mask_tile.sum() / mask_tile.size < MIN_TURF_RATIO:
                skipped += 1
                continue

            name = f'{image_id}_{r}_{c}'
            PILImage.fromarray(img_tile, 'RGB').save(f'{img_out}/{name}.png')
            PILImage.fromarray((mask_tile * 255).astype(np.uint8), 'L').save(f'{mask_out}/{name}.png')
            saved += 1

    print(f'  [{image_id}] Saved: {saved} tiles | Skipped (low turf): {skipped}')
    return saved

total = sum(tile_image(i) for i in ['1', '2', '3'])
train_n = len(os.listdir(f'{TILE_DIR}/images/train'))
val_n   = len(os.listdir(f'{TILE_DIR}/images/val'))
print(f'\n✅ Tiling complete: {train_n} train tiles | {val_n} val tiles')

# # preview Some Tiles


import random

train_imgs = sorted(os.listdir(f'{TILE_DIR}/images/train'))
samples    = random.sample(train_imgs, min(6, len(train_imgs)))

fig, axes = plt.subplots(2, 6, figsize=(20, 7))
fig.suptitle('Sample Training Tiles — Image (top) vs Mask (bottom)', fontsize=12)

for i, name in enumerate(samples):
    img_tile  = np.array(PILImage.open(f'{TILE_DIR}/images/train/{name}'))
    mask_tile = np.array(PILImage.open(f'{TILE_DIR}/masks/train/{name}'))
    axes[0][i].imshow(img_tile);              axes[0][i].axis('off')
    axes[1][i].imshow(mask_tile, cmap='Greens'); axes[1][i].axis('off')

plt.tight_layout()
plt.show()
print('Top row = image tile | Bottom row = turf mask (white = turf)')

# # Define datasets + loss + model


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 1024   # SAM always expects 1024x1024
print(f'Device: {DEVICE}')

# ── DATASET ──────────────────────────────────────────────────────────
class TurfDataset(Dataset):
    def __init__(self, split='train', augment=True):
        self.img_dir  = f'{TILE_DIR}/images/{split}'
        self.mask_dir = f'{TILE_DIR}/masks/{split}'
        self.augment  = augment and (split == 'train')
        self.samples  = sorted([
            p for p in os.listdir(self.img_dir)
            if os.path.exists(f'{self.mask_dir}/{p}')
        ])
        print(f'[{split}] {len(self.samples)} tiles loaded')

    def __len__(self): return len(self.samples)

    def _augment(self, img, mask):
        import random
        if random.random() > 0.5: img = img.flip(-1);  mask = mask.flip(-1)
        if random.random() > 0.5: img = img.flip(-2);  mask = mask.flip(-2)
        k = random.randint(0, 3)
        if k: img = torch.rot90(img, k, [-2,-1]); mask = torch.rot90(mask, k, [-2,-1])
        if random.random() > 0.5:
            img = (img * random.uniform(0.75, 1.25)).clamp(0, 1)
        return img, mask

    def __getitem__(self, idx):
        name = self.samples[idx]
        img  = np.array(PILImage.open(f'{self.img_dir}/{name}').convert('RGB'))
        mask = np.array(PILImage.open(f'{self.mask_dir}/{name}').convert('L'))
        mask = (mask > 127).astype(np.float32)

        img_r  = np.array(PILImage.fromarray(img).resize((IMG_SIZE, IMG_SIZE)))
        mask_r = np.array(PILImage.fromarray((mask*255).astype(np.uint8)).resize(
                     (IMG_SIZE, IMG_SIZE), PILImage.NEAREST)) / 255.0

        img_t  = torch.from_numpy(img_r).permute(2,0,1).float() / 255.0
        mask_t = torch.from_numpy(mask_r).unsqueeze(0).float()

        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t)

        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img_t = (img_t - mean) / std
        return img_t, mask_t


# ── LOSS ─────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def dice(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        return 1 - (2*(p*t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)

    def forward(self, logits, targets):
        return 0.4 * self.bce(logits, targets) + 0.6 * self.dice(logits, targets)


# ── MODEL ─────────────────────────────────────────────────────────────
class TurfSAM(nn.Module):
    def __init__(self, ckpt='/kaggle/working/weights/mobile_sam.pt'):
        super().__init__()
        from mobile_sam import sam_model_registry
        self.sam = sam_model_registry['vit_t'](checkpoint=ckpt)

        # Freeze encoder + prompt encoder — only train mask decoder
        for p in self.sam.image_encoder.parameters():  p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters(): p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():   p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)')

    def _grid_prompts(self, B, H, W, device):
        xs = torch.linspace(0, W-1, 16, device=device)
        ys = torch.linspace(0, H-1, 16, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pts = torch.stack([gx.flatten(), gy.flatten()], -1).unsqueeze(0).expand(B,-1,-1)
        lbl = torch.ones(B, pts.shape[1], device=device)
        return pts, lbl

    def forward(self, images):
        B, C, H, W = images.shape
        all_masks = []
    
        for b in range(B):
            img = images[b:b+1]                          # (1, C, H, W)
    
            # encoder + prompt encoder are frozen — no grad needed
            with torch.no_grad():
                emb = self.sam.image_encoder(img)
                pts = self._grid_prompts(1, H, W, img.device)
                sp, dp = self.sam.prompt_encoder(points=pts, boxes=None, masks=None)
    
            # mask decoder IS being trained — must be outside no_grad
            masks, _ = self.sam.mask_decoder(
                image_embeddings=emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False,
            )
            all_masks.append(masks)                      # (1, 1, H', W')
    
        out = torch.cat(all_masks, dim=0)                # (B, 1, H', W')
        return F.interpolate(out, (H, W), mode='bilinear', align_corners=False)

print('\n✅ Dataset, Loss, Model classes defined')

# # *MOST IMPORTANT* train mobileSAM


from tqdm.notebook import tqdm as tqdm_nb

# ── HYPERPARAMS ──────────────────────────────────────
EPOCHS     = 40
BATCH_SIZE = 4
LR         = 1e-4
# ─────────────────────────────────────────────────────

train_ds = TurfDataset('train', augment=True)
val_ds   = TurfDataset('val',   augment=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model     = TurfSAM().to(DEVICE)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = DiceBCELoss()

history    = {'train_loss':[], 'val_loss':[], 'val_iou':[], 'val_f1':[]}
best_iou   = 0.0
CKPT_PATH  = '/kaggle/working/weights/best.pt'

def metrics(logits, targets):
    p = (torch.sigmoid(logits) > 0.5).float().view(-1)
    t = targets.view(-1)
    tp = (p*t).sum().item()
    fp = (p*(1-t)).sum().item()
    fn = ((1-p)*t).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    f1  = 2*tp / (2*tp + fp + fn + 1e-6)
    return iou, f1

print(f'Training {EPOCHS} epochs on {DEVICE}...\n')

for epoch in range(1, EPOCHS+1):
    # ── Train ──
    model.train()
    t_losses = []
    for imgs, masks in tqdm_nb(train_dl, desc=f'Ep {epoch}/{EPOCHS} train', leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_losses.append(loss.item())
    scheduler.step()

    # ── Validate ──
    model.eval()
    v_losses, v_ious, v_f1s = [], [], []
    with torch.no_grad():
        for imgs, masks in tqdm_nb(val_dl, desc=f'Ep {epoch}/{EPOCHS} val  ', leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            v_losses.append(criterion(logits, masks).item())
            iou, f1 = metrics(logits, masks)
            v_ious.append(iou); v_f1s.append(f1)

    tl = np.mean(t_losses); vl = np.mean(v_losses)
    vi = np.mean(v_ious);   vf = np.mean(v_f1s)
    history['train_loss'].append(tl); history['val_loss'].append(vl)
    history['val_iou'].append(vi);    history['val_f1'].append(vf)

    tag = ''
    if vi > best_iou:
        best_iou = vi
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'val_iou': vi, 'val_f1': vf}, CKPT_PATH)
        tag = ' ⭐ BEST'

    print(f'Ep {epoch:3d}/{EPOCHS} | TrLoss:{tl:.4f} | ValLoss:{vl:.4f} | IoU:{vi:.4f} | F1:{vf:.4f}{tag}')

print(f'\n✅ Training done. Best Val IoU: {best_iou:.4f}')

# # plot training curves


fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('MobileSAM Fine-tuning — Training Curves', fontsize=13)

axes[0].plot(history['train_loss'], label='Train', color='royalblue')
axes[0].plot(history['val_loss'],   label='Val',   color='tomato')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history['val_iou'], color='seagreen', marker='o', ms=3)
axes[1].set_title('Validation IoU'); axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

axes[2].plot(history['val_f1'], color='darkorange', marker='o', ms=3)
axes[2].set_title('Validation F1 (Dice)'); axes[2].set_xlabel('Epoch')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/results/training_curves.png', dpi=130, bbox_inches='tight')
plt.show()
print(f'Best Val IoU: {max(history["val_iou"]):.4f}')
print(f'Best Val F1:  {max(history["val_f1"]):.4f}')

# # Run inference on Training images


from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape

CONF = 0.35
TILE_INF = 640
STRIDE_INF = 320

# Load best model
model.eval()
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
print(f"Loaded best model from epoch {ckpt['epoch']} (IoU={ckpt['val_iou']:.4f})")

def to_uint8(arr):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    
def preprocess(tile):
    t = torch.from_numpy(cv2.resize(tile, (IMG_SIZE, IMG_SIZE))).permute(2,0,1).float()/255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return ((t - mean) / std).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_full(img):
    H, W = img.shape[:2]
    full  = np.zeros((H,W), np.float32)
    count = np.zeros((H,W), np.float32)

    def starts(size, stride, dim):
        s = list(range(0, dim-size+1, stride))
        if not s or s[-1]+size < dim: s.append(max(0, dim-size))
        return s

    for r in starts(TILE_INF, STRIDE_INF, H):
        for c in starts(TILE_INF, STRIDE_INF, W):
            r2, c2 = min(r+TILE_INF, H), min(c+TILE_INF, W)
            tile = img[r:r2, c:c2]
            if tile.shape[0] < TILE_INF or tile.shape[1] < TILE_INF:
                tile = np.pad(tile, ((0,TILE_INF-tile.shape[0]),(0,TILE_INF-tile.shape[1]),(0,0)))
            prob = torch.sigmoid(model(preprocess(tile))).squeeze().cpu().numpy()
            prob = cv2.resize(prob, (c2-c, r2-r))
            full[r:r2, c:c2] += prob
            count[r:r2, c:c2] += 1

    full /= np.maximum(count, 1)
    binary = (full > CONF).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary, full

def save_outputs(img_id, img, binary, prob, meta):
    stem = img_id

    # Overlay
    scale = min(1.0, 1200/max(img.shape[:2]))
    H2,W2 = int(img.shape[0]*scale), int(img.shape[1]*scale)
    img_s   = cv2.resize(img,    (W2, H2))
    bin_s   = cv2.resize(binary, (W2, H2), interpolation=cv2.INTER_NEAREST)
    prob_s  = cv2.resize(prob,   (W2, H2))
    ov = img_s.copy().astype(np.float32)
    g  = np.zeros_like(ov); g[bin_s==1] = [0,200,0]
    ov = (ov*0.65 + g*0.35).clip(0,255).astype(np.uint8)
    contours,_ = cv2.findContours(bin_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, contours, -1, (0,255,0), 2)

    fig, axes = plt.subplots(1,3,figsize=(20,6))
    axes[0].imshow(img_s);  axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(prob_s, cmap='RdYlGn', vmin=0, vmax=1); axes[1].set_title('Confidence'); axes[1].axis('off')
    axes[2].imshow(ov);     axes[2].set_title('Detection Overlay'); axes[2].axis('off')
    cov = 100*binary.mean()
    plt.suptitle(f'Image {img_id} — Turf Coverage: {cov:.2f}%', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/results/overlays/{stem}_overlay.png', dpi=130, bbox_inches='tight')
    plt.show()

    # Mask GeoTIFF
    profile = {'driver':'GTiff','dtype':'uint8','width':meta['width'],'height':meta['height'],'count':1,'compress':'lzw'}
    if meta['crs']: profile['crs']=meta['crs']; profile['transform']=meta['transform']
    with rasterio.open(f'/kaggle/working/results/predictions/{stem}_mask.tiff','w',**profile) as dst:
        dst.write(binary[np.newaxis])

    # GeoJSON
    MIN_PX = 200
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary)
    cv2.drawContours(clean_mask, [c for c in contours if cv2.contourArea(c) > MIN_PX], -1, 1, -1)
    polys = [shp_shape(g) for g,v in rio_shapes(clean_mask, mask=clean_mask, transform=meta['transform']) if v==1]
    if polys:
        gdf = gpd.GeoDataFrame({'class':['turf']*len(polys),'area_m2':[round(p.area,2) for p in polys]},
                                geometry=polys, crs=meta['crs'] if meta['crs'] else 'EPSG:4326')
        gdf.to_file(f'/kaggle/working/results/geojson/{stem}_turf.geojson', driver='GeoJSON')
        print(f'  [{img_id}] GeoJSON: {len(polys)} polygons | Coverage: {cov:.2f}%')
    else:
        print(f'  [{img_id}] No polygons found')

# Run on all 3 training images
for img_id in ['1', '2', '3']:
    with rasterio.open(f'{DATA_DIR}/{img_id}_georef.tiff') as src:
        img_arr = src.read()[:3].transpose(1,2,0)
        meta = {'crs':src.crs, 'transform':src.transform, 'width':src.width, 'height':src.height}
    img_arr = to_uint8(img_arr)
    
    print(f'Running inference on image {img_id}...')
    binary, prob = predict_full(img_arr)
    save_outputs(img_id, img_arr, binary, prob, meta)

# # generalization test


import glob

external_images = sorted(glob.glob('/kaggle/input/datasets/prachi232005/external-aerial-image/*.tif') +
                         glob.glob('/kaggle/input/datasets/prachi232005/external-aerial-image/*.tiff'))

if not external_images:
    print('⚠️  No external images found — check dataset is attached')
else:
    for i, path in enumerate(external_images):
        ext_id = f'external_{i+1}'
        print(f'Running generalization test on: {path}')
        with rasterio.open(path) as src:
            ext_img  = to_uint8(src.read()[:3].transpose(1,2,0))   # normalize
            ext_meta = {'crs':src.crs, 'transform':src.transform,
                        'width':src.width, 'height':src.height}
        print(f'  Size: {ext_meta["width"]}×{ext_meta["height"]} | CRS: {ext_meta["crs"]}')
        binary, prob = predict_full(ext_img)
        save_outputs(ext_id, ext_img, binary, prob, ext_meta)

    print('\n✅ Generalization test complete')

# # saving outputs


import zipfile

# Zip results + weights for download
output_zip = '/kaggle/working/turf_detection_outputs.zip'
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['results', 'weights']:
        for root, dirs, files in os.walk(f'/kaggle/working/{folder}'):
            for file in files:
                full = os.path.join(root, file)
                arcname = full.replace('/kaggle/working/', '')
                zf.write(full, arcname)

size_mb = os.path.getsize(output_zip) / 1e6
print(f'✅ Outputs zipped: {output_zip} ({size_mb:.1f} MB)')
print('\nContents:')
with zipfile.ZipFile(output_zip) as zf:
    for name in sorted(zf.namelist()):
        print(' ', name)