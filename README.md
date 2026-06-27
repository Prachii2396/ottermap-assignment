# Turf Detection — Aerial Imagery Segmentation

MobileSAM fine-tuned to detect turf/grass from high-resolution aerial imagery. Produces GeoJSON polygons and mask GeoTIFFs compatible with downstream GIS workflows.

## Setup

```bash
pip install -r requirements.txt
```

Download model weights and place them in the `weights/` folder:
- `weights/mobile_sam.pt`
- `weights/best.pt`

**Weights download:** [Google Drive link]

## Inference

Single image:
```bash
python inference.py --image input_image.tif
```

Batch (folder of images):
```bash
python inference.py --input ./images/ --output ./results
```

Optional flags:
```
--output   Output folder (default: ./output)
--conf     Confidence threshold 0–1 (default: 0.35)
--sam_ckpt Path to mobile_sam.pt (default: weights/mobile_sam.pt)
--best_ckpt Path to best.pt (default: weights/best.pt)
```

## Outputs

For each input image the script produces:
- `{stem}_overlay.png` — visual overlay with detected turf highlighted
- `{stem}_mask.tif` — binary mask GeoTIFF
- `{stem}_turf.geojson` — polygonized detections with area in m²

## Repository Structure

```
turf-detection/
├── inference.py          ← run this for evaluation
├── train.py              ← full training pipeline
├── requirements.txt
├── README.md
├── weights/
│   ├── mobile_sam.pt     ← base MobileSAM weights
│   └── best.pt           ← fine-tuned checkpoint
└── results/
    ├── overlays/
    ├── predictions/
    └── geojson/
```

## Model

- **Architecture:** MobileSAM (ViT-Tiny image encoder + mask decoder)
- **Training:** Frozen encoder, fine-tuned mask decoder only
- **Input:** 1024×1024 patches, sliding window inference at 640px stride 320px
- **Loss:** Dice + BCE (0.6 / 0.4 weighting)
- **Training data:** 3 aerial images across California, Washington, South Carolina
