# Dataset Setup Guide

## ⚠️ DO NOT UPLOAD ACTUAL IMAGES TO GITHUB

This folder contains only the **structure** for the dataset, not the actual images.

## Download DeepPCB Dataset

1. **Source**: [DeepPCB Dataset Download Link]
   - PCB images with 6 defect types
   - Template (defect-free) + Test (with defects) pairs

2. **Dataset Contents**:
   - Missing_hole
   - Mouse_bite
   - Open_circuit
   - Short
   - Spur
   - Spurious_copper

## Setup Instructions

### 1. Download and Extract Dataset

```bash
# Download DeepPCB dataset
# Extract to a temporary location
```

### 2. Organize Files

Place images in this structure:

```
data/
├── raw/
│   ├── template/          ← Place defect-free PCB images here
│   │   ├── 01.jpg
│   │   ├── 04.jpg
│   │   ├── 05.jpg
│   │   └── ...
│   │
│   ├── test/              ← Place test images here (organized by category)
│   │   ├── Missing_hole/
│   │   │   ├── 01_missing_hole_01.jpg
│   │   │   ├── 01_missing_hole_02.jpg
│   │   │   └── ...
│   │   ├── Mouse_bite/
│   │   ├── Open_circuit/
│   │   ├── Short/
│   │   ├── Spur/
│   │   └── Spurious_copper/
│   │
│   └── gt_mask/           ← Ground truth masks (optional, for evaluation)
│
└── processed/             ← Pipeline outputs (auto-created)
    ├── diff/
    ├── binary_mask/
    └── aligned/
```

### 3. Naming Convention

**Template images**: `XX.jpg` where XX is the template ID (e.g., 01, 04, 05)

**Test images**: `XX_category_YY.jpg`
- XX = template ID
- category = defect type
- YY = sequence number

Example: `01_missing_hole_01.jpg` uses template `01.jpg`

### 4. Verify Setup

```bash
# Inspect dataset
python src/module1_dataset_subtraction/inspect_dataset.py \
  data/raw/template \
  data/raw/test
```

Expected output:
```
✓ Templates found: 10
✓ Test images found: 500+
✓ Valid pairs: 500+
✅ DATASET OK - Ready for processing
```

## Important Notes

- **Never commit raw images to GitHub** (they're in `.gitignore`)
- Only commit this README and folder structure
- Ground truth masks are optional (only needed for evaluation)
- If someone can't run your code after cloning → your repo is incomplete

## File Sizes

Typical dataset sizes:
- Template images: ~10 files (~50MB)
- Test images: ~500 files (~2.5GB)
- **Total**: ~2.5GB (DO NOT upload to GitHub!)

## Troubleshooting

### "No valid pairs found"
- Check template files are in `data/raw/template/`
- Verify naming: template `01.jpg` for test `01_missing_hole_01.jpg`

### "Size mismatch"
- Some test images may have different dimensions
- Pipeline will auto-resize if needed

### "Broken images"
- Check image files are not corrupted
- Re-download affected images

---

**After setup, your `data/` folder should look like this, but is NOT uploaded to GitHub.**
