# Module 1 Sample Outputs ✅

This folder contains **36 actual sample images** demonstrating the Module 1 pipeline results across all 6 PCB defect types.

---

## 📊 What's Included

**6 Defect Types × 6 Outputs Each = 36 Images**

### Defect Types Demonstrated:
1. ✅ **Missing Hole** (`01_missing_hole_01`)
2. ✅ **Mouse Bite** (`04_mouse_bite_01`)
3. ✅ **Open Circuit** (`05_open_circuit_01`)
4. ✅ **Short** (`06_short_01`)
5. ✅ **Spur** (`07_spur_01`)
6. ✅ **Spurious Copper** (`08_spurious_copper_01`)

### Output Types Per Sample:
- `*_template.png` - Original template (defect-free PCB)
- `*_test.png` - Test image (with defect)
- `*_diff.png` - Absolute difference map (grayscale)
- `*_mask.png` - Binary mask from Otsu thresholding
- `*_overlay.png` - Defects highlighted in red
- `*_visualization.png` - 6-panel comparison view

**Total Size**: ~65 MB

---

## 🎯 Detection Results Summary

| Defect Type | Otsu Threshold | Status |
|-------------|----------------|--------|
| Missing Hole | 43 | ✅ Clear detection |
| Mouse Bite | 13 | ✅ Small defects visible |
| Open Circuit | 0 | ⚠️ Minimal difference |
| Short | 0 | ⚠️ Minimal difference |
| Spur | 11 | ✅ Good detection |
| Spurious Copper | 15 | ✅ Good detection |

---

## 🔍 How These Were Generated

```bash
cd /Users/fariaththeen/Documents/PCB/pcb-defect-detection
source venv/bin/activate

# Example: Missing hole
python src/module1_dataset_subtraction/pipeline.py \
  --template ../PCB_DATASET/PCB_USED/01.JPG \
  --test ../PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg \
  --output outputs/module1_samples

# Repeated for all 6 defect types
```

---

## 💡 Key Observations

**Successful Detections:**
- **Missing Hole**: High threshold (43) indicates clear defects
- **Mouse Bite**: Detected despite small size
- **Spur & Spurious Copper**: Well-isolated defects

**Challenging Cases:**
- **Open Circuit & Short**: Very low thresholds (0) suggest minimal visual difference or alignment issues

---

## 📁 File Naming

```
{template_id}_{defect_type}_{test_id}_{output_type}.png
```

Examples:
- `01_missing_hole_01_diff.png`
- `04_mouse_bite_01_mask.png`
- `08_spurious_copper_01_overlay.png`

---

**Purpose**: These samples prove the Module 1 pipeline works across diverse defect types without uploading the full dataset.
