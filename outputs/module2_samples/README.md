# Module 2 Sample Outputs ✅

This folder contains **sample ROIs** demonstrating the Module 2 pipeline results.

---

## 📊 What's Included

**ROI Extraction from Module 1 Masks:**

### Samples Processed:
1. ✅ **Missing Hole** - 3 ROIs extracted
2. ✅ **Mouse Bite** - 3 ROIs extracted
3. ✅ **Spur** - 5 ROIs extracted
4. ✅ **Spurious Copper** - 5 ROIs extracted

**Total**: 16+ normalized ROIs (64×64 pixels)

---

## 📁 Output Structure

```
module2_samples/
├── normalized_rois/
│   ├── 01_missing_hole_01_test_roi_001.png
│   ├── 01_missing_hole_01_test_roi_002.png
│   ├── 01_missing_hole_01_test_roi_003.png
│   ├── 04_mouse_bite_01_test_roi_001.png
│   ├── ...
│
└── visualizations/
    ├── 01_missing_hole_01_test_contours.png
    ├── 01_missing_hole_01_test_boxes.png
    ├── 01_missing_hole_01_test_roi_grid.png
    ├── ...
```

---

## 🎯 Processing Pipeline

Each sample went through:

1. **Contour Detection**: Find defect boundaries in binary mask
2. **Filtering**: Remove noise (< 50 px²)
3. **Bounding Boxes**: Extract rectangles with 10-15px padding
4. **ROI Cropping**: Extract defect regions from source image
5. **Normalization**: Resize to 64×64 (pad to preserve aspect ratio)
6. **Label Assignment**: Automatically extracted from filename

---

## 📊 Sample Statistics

| Image | Contours Found | Filtered | ROIs Extracted | Label |
|-------|---------------|----------|----------------|-------|
| 01_missing_hole_01 | 3 | 3 | 3 | missing_hole |
| 04_mouse_bite_01 | 3 | 3 | 3 | mouse_bite |
| 07_spur_01 | 5 | 5 | 5 | spur |
| 08_spurious_copper_01 | 5 | 5 | 5 | spurious_copper |

**Success Rate**: 100% (all contours successfully extracted)

---

## 🔍 Visualization Types

### 1. Contours
Shows detected defect boundaries overlaid on binary mask.

### 2. Bounding Boxes
Shows extracted rectangles with padding around defects.

### 3. ROI Grid
Shows all extracted ROIs in a grid layout (up to 25 ROIs per image).

---

## 🚀 How These Were Generated

```bash
cd /Users/fariaththeen/Documents/PCB/pcb-defect-detection
source venv/bin/activate

# Missing hole example
python src/module2_roi_extraction/pipeline.py \
  --mask outputs/module1_samples/01_missing_hole_01_mask.png \
  --source outputs/module1_samples/01_missing_hole_01_test.png \
  --output outputs/module2_samples \
  --visualize

# Similar for other defect types...
```

---

## ⚙️ Parameters Used

- **Target size**: 64×64 pixels
- **Padding**: 10-15 pixels around bounding boxes
- **Min area**: 50 px² (contour filtering)
- **Normalization strategy**: Pad + Resize (preserves aspect ratio)

---

## 💡 Key Observations

### Successful Extractions:
- **Missing Hole**: 3 distinct circular defects detected
- **Mouse Bite**: 3 small edge defects successfully captured
- **Spur**: 5 protruding defects with clear boundaries
- **Spurious Copper**: 5 extra copper regions detected

### ROI Quality:
- All ROIs are properly centered around defects
- Padding provides sufficient context for classification
- Normalization preserves defect shape (no distortion)

---

## 📌 Next Steps

These normalized ROIs are ready for:
1. **Module 3 (CNN Classification)**: Train deep learning model
2. **Data Augmentation**: Rotate, flip, adjust brightness
3. **Transfer Learning**: Use as input for pre-trained models
4. **Defect Analysis**: Statistical analysis of defect sizes/shapes

---

## ✅ Ready for ML Training

The `normalized_rois/` folder contains properly sized, labeled defect images that can be directly fed into a CNN classifier.

---

**Purpose**: These samples demonstrate Module 2's ability to extract clean, normalized ROIs from binary masks for machine learning.
