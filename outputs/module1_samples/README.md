# Sample Outputs

This folder contains **sample results** from Module 1 pipeline to demonstrate that the code works.

## Rules for GitHub

✅ **Upload**: 5-10 representative sample images  
❌ **Do NOT upload**: Full processed dataset  

## What's Included

Sample outputs showing successful defect detection:

- `sample_01_diff.png` - Difference map
- `sample_01_mask.png` - Clean binary mask
- `sample_01_overlay.png` - Defects highlighted on original

These prove your implementation works without cluttering the repository.

## Generating Samples

After setting up your dataset, run:

```bash
# Process one sample
python src/module1_dataset_subtraction/pipeline.py \
  --template data/raw/template/01.jpg \
  --test data/raw/test/01_missing_hole_01.jpg \
  --output outputs/module1_samples
```

Select your best 5-10 results to commit to GitHub.

## Guidelines

- Show variety: different defect types
- Show success: clear defect detection
- Keep small: optimize image sizes
- No huge folders: reviewers want quick verification, not full dataset

---

**Purpose**: This proves your code actually works. Reviewers can see results without running the full pipeline.
