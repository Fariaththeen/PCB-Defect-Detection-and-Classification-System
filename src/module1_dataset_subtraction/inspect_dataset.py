"""
Dataset Inspection and Validation
Verifies image pairs, dimensions, and integrity before processing
"""

import cv2
import os
from pathlib import Path
import sys


def inspect_dataset(template_dir, test_dir):
    """
    Validate dataset before processing
    
    Args:
        template_dir: Directory with template (defect-free) images
        test_dir: Directory with test (defective) images
    
    Returns:
        dict: Inspection results
    """
    print("="*80)
    print("DATASET INSPECTION")
    print("="*80)
    
    results = {
        "total_templates": 0,
        "total_tests": 0,
        "valid_pairs": 0,
        "broken_images": [],
        "size_mismatches": []
    }
    
    # Check template directory
    template_path = Path(template_dir)
    if not template_path.exists():
        print(f"❌ Template directory not found: {template_dir}")
        return results
    
    # Load templates
    template_files = {}
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        for f in template_path.glob(ext):
            template_id = f.stem
            img = cv2.imread(str(f))
            if img is not None:
                template_files[template_id] = {
                    'path': f,
                    'shape': img.shape
                }
                results["total_templates"] += 1
            else:
                results["broken_images"].append(str(f))
    
    print(f"\n✓ Templates found: {results['total_templates']}")
    
    # Check test directory
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return results
    
    # Validate test images
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        for f in test_path.glob(ext):
            results["total_tests"] += 1
            
            # Extract template ID from filename (e.g., "01" from "01_missing_hole_01.jpg")
            filename = f.name
            template_id = filename.split('_')[0]
            
            if template_id not in template_files:
                print(f"⚠️  No template for: {filename}")
                continue
            
            # Load test image
            test_img = cv2.imread(str(f))
            if test_img is None:
                results["broken_images"].append(str(f))
                continue
            
            # Check dimensions
            template_shape = template_files[template_id]['shape']
            if test_img.shape != template_shape:
                results["size_mismatches"].append({
                    'file': filename,
                    'template_shape': template_shape,
                    'test_shape': test_img.shape
                })
            else:
                results["valid_pairs"] += 1
    
    print(f"✓ Test images found: {results['total_tests']}")
    print(f"✓ Valid pairs: {results['valid_pairs']}")
    
    # Report issues
    if results["broken_images"]:
        print(f"\n⚠️  Broken images ({len(results['broken_images'])}):")
        for img in results["broken_images"][:5]:
            print(f"  - {img}")
        if len(results["broken_images"]) > 5:
            print(f"  ... and {len(results['broken_images']) - 5} more")
    
    if results["size_mismatches"]:
        print(f"\n⚠️  Size mismatches ({len(results['size_mismatches'])}):")
        for item in results["size_mismatches"][:5]:
            print(f"  - {item['file']}")
    
    # Final verdict
    print("\n" + "="*80)
    if results["valid_pairs"] > 0 and not results["broken_images"]:
        print("✅ DATASET OK - Ready for processing")
    elif results["valid_pairs"] > 0:
        print("⚠️  DATASET HAS ISSUES - Can proceed with caution")
    else:
        print("❌ DATASET FAILED - Fix issues before processing")
    print("="*80)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_dataset.py <template_dir> <test_dir>")
        print("\nExample:")
        print("  python inspect_dataset.py data/raw/template data/raw/test")
        sys.exit(1)
    
    template_dir = sys.argv[1]
    test_dir = sys.argv[2]
    
    inspect_dataset(template_dir, test_dir)
