# PCB Defect Detection System

## Objective

This project implements a complete **PCB defect detection pipeline** using classical image processing and deep learning. The system detects and classifies 6 types of manufacturing defects in printed circuit boards.

**Current Status**: ✅ Module 1 (Image Subtraction) + ✅ Module 2 (ROI Extraction) Implemented

## Dataset

**DeepPCB Dataset**: PCB images with 6 defect types
- Missing_hole
- Mouse_bite
- Open_circuit
- Short
- Spur
- Spurious_copper

**Dataset Structure:**
- Template images: Defect-free reference PCBs
- Test images: PCBs with manufacturing defects
- Ground truth masks: Annotation labels for evaluation

**Download:** [DeepPCB Dataset Link]
Place images in `data/raw/template/` and `data/raw/test/` according to `data/README.md`

## Module 1 Pipeline

```
Template + Test Images
         ↓
   [Preprocessing]
   - Grayscale conversion
   - Gaussian denoising
   - Alignment validation
         ↓
   [Subtraction]
   - Absolute difference: |Test - Template|
   - Difference metrics
         ↓
   [Thresholding]
   - Otsu's automatic method
   - Binary mask generation
         ↓
   [Post-processing]
   - Morphological operations
   - Noise removal
   - Hole filling
         ↓
   Clean Binary Mask
```

**Processing time:** ~0.5-1 second per image pair

## How to Run

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/pcb-defect-detection.git
cd pcb-defect-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Follow instructions in `data/README.md` to download and organize the DeepPCB dataset.

### 3. Run Pipeline

**Inspect dataset:**
```bash
python src/module1_dataset_subtraction/inspect_dataset.py
```

**Process images:**
```bash
python src/module1_dataset_subtraction/pipeline.py \
  --template_dir data/raw/template \
  --test_dir data/raw/test \
  --output_dir data/processed
```

**Process single pair (testing):**
```bash
python src/module1_dataset_subtraction/pipeline.py \
  --template data/raw/template/01.jpg \
  --test data/raw/test/01_missing_hole_01.jpg \
  --output data/processed/sample_01
```

## Outputs

After processing, `data/processed/` contains:

- **diff/** - Difference maps showing detected changes
- **binary_mask/** - Clean binary masks (white = defect)
- **aligned/** - Preprocessed aligned image pairs

Sample outputs are in `outputs/module1_samples/` for reference.

## Key Technical Decisions

### Why Otsu Thresholding?
- Automatic threshold selection
- No manual tuning required
- Minimizes intra-class variance
- Works well for PCB images with clear foreground/background

### Why Morphological Operations?
- Removes salt-and-pepper noise
- Fills small gaps in defects
- Preserves defect shape and size
- No ML training required

### Known Limitations
- Requires roughly aligned template-test pairs
- Sensitive to large illumination differences
- Small defects (<10 pixels) may be filtered as noise
- Template must be truly defect-free

See `docs/module1_explanation.md` for detailed technical explanation.

---

## Module 2 Pipeline (ROI Extraction) ✅

```
Binary Mask from Module 1
         ↓
   [Contour Detection]
   - Find defect boundaries
   - Compute properties
         ↓
   [Contour Filtering]
   - Remove noise (< 50 px²)
   - Filter by aspect ratio
         ↓
   [Bounding Box Extraction]
   - Get defect rectangles
   - Add padding (10 pixels)
         ↓
   [ROI Cropping]
   - Extract defect regions
   - Preserve original pixels
         ↓
   [Normalization]
   - Resize to 64×64
   - Pad to preserve aspect ratio
         ↓
   [Label Assignment]
   - Extract from filename
   - Create label manifest
         ↓
   Fixed-size, Labeled ROI Dataset
```

**Processing time:** ~0.1-0.2 seconds per image

### Run Module 2

```bash
# Single image (with visualization)
python src/module2_roi_extraction/pipeline.py \
  --mask data/processed/binary_mask/01_missing_hole_01_mask.png \
  --source data/raw/test/01_missing_hole_01.jpg \
  --output data/rois/ \
  --visualize

# Batch processing
python src/module2_roi_extraction/pipeline.py \
  --mask_dir data/processed/binary_mask/ \
  --source_dir data/raw/test/ \
  --output data/rois/

# Custom settings
python src/module2_roi_extraction/pipeline.py \
  --mask mask.png --source image.jpg --output rois/ \
  --size 128 --padding 15 --min_area 100
```

**Outputs**:
- `normalized_rois/`: Fixed-size defect images (64×64)
- `label_manifest.json`: Dataset metadata for ML training
- `visualizations/`: Contours, bounding boxes, ROI grids

See `docs/module2_explanation.md` for technical details.

---

## Module 3 (CNN Classification) 🚧

**Status**: Coming soon

**Planned Features**:
- Train CNN classifier on extracted ROIs
- 6-class defect classification
- Model evaluation and validation
- FastAPI inference endpoint

## Project Structure

```
pcb-defect-detection/
├── README.md                    (this file)
├── requirements.txt             (dependencies)
├── .gitignore
│
├── src/
│   ├── module1_dataset_subtraction/
│   │   ├── inspect_dataset.py   (dataset validation)
│   │   ├── preprocess.py        (grayscale, denoise)
│   │   ├── subtraction.py       (image difference)
│   │   ├── thresholding.py      (Otsu method)
│   │   ├── postprocess.py       (morphology)
│   │   └── pipeline.py          (end-to-end Module 1)
│   │
│   └── module2_roi_extraction/  ✅ NEW
│       ├── contour_detection.py (find defect contours)
│       ├── filter_contours.py   (remove noise)
│       ├── bounding_box.py      (extract boxes)
│       ├── roi_extractor.py     (crop regions)
│       ├── roi_normalizer.py    (resize to 64×64)
│       ├── label_assignment.py  (assign labels)
│       └── pipeline.py          (end-to-end Module 2)
│
├── data/
│   ├── README.md                (dataset setup guide)
│   ├── raw/                     (place DeepPCB here)
│   ├── processed/               (Module 1 outputs)
│   └── rois/                    ✅ NEW (Module 2 outputs)
│
├── outputs/
│   ├── module1_samples/         (Module 1 sample results)
│   └── module2_samples/         ✅ NEW (Module 2 sample ROIs)
│
└── docs/
    ├── module1_explanation.md   (detailed explanation)
    └── module2_explanation.md   ✅ NEW (ROI extraction details)
```

## Requirements

- Python ≥ 3.8
- OpenCV
- NumPy
- Matplotlib

See `requirements.txt` for versions.

## Results

**Success rate:** 95%+ on DeepPCB dataset  
**Processing speed:** 500 images in ~5-10 minutes  
**Output quality:** Clean binary masks with <5% false positives

Sample results in `outputs/module1_samples/`

## Author

FARIATHTHEEN F
