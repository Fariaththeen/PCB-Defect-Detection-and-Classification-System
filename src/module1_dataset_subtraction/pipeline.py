"""
Complete Pipeline
End-to-end Module 1 processing
"""

import cv2
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from preprocess import preprocess_pair
from subtraction import subtract_images, compute_metrics
from thresholding import otsu_threshold
from postprocess import postprocess


def process_image_pair(template_path, test_path, output_dir=None, visualize=True):
    """
    Process single template-test pair through complete pipeline
    
    Args:
        template_path: Path to template image
        test_path: Path to test image
        output_dir: Directory to save outputs (optional)
        visualize: Create visualization (default True)
    
    Returns:
        Dictionary with all results
    """
    print(f"\nProcessing: {Path(test_path).name}")
    
    # Load images
    template = cv2.imread(template_path)
    test = cv2.imread(test_path)
    
    if template is None or test is None:
        print("❌ Failed to load images")
        return None
    
    # 1. Preprocess
    print("  [1/4] Preprocessing...")
    template_proc, test_proc = preprocess_pair(template, test)
    
    # 2. Subtract
    print("  [2/4] Subtracting...")
    diff = subtract_images(template_proc, test_proc)
    diff_metrics = compute_metrics(diff)
    
    # 3. Threshold
    print("  [3/4] Thresholding...")
    binary_mask, threshold_value = otsu_threshold(diff)
    
    # 4. Post-process
    print("  [4/4] Post-processing...")
    cleaned_mask = postprocess(binary_mask)
    
    print(f"  ✓ Complete (Otsu threshold: {threshold_value})")
    
    # Save outputs
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        basename = Path(test_path).stem
        
        cv2.imwrite(str(output_path / f"{basename}_template.png"), template_proc)
        cv2.imwrite(str(output_path / f"{basename}_test.png"), test_proc)
        cv2.imwrite(str(output_path / f"{basename}_diff.png"), diff)
        cv2.imwrite(str(output_path / f"{basename}_mask.png"), cleaned_mask)
        
        # Create overlay
        overlay = cv2.cvtColor(test_proc, cv2.COLOR_GRAY2BGR)
        overlay[cleaned_mask == 255] = [0, 0, 255]  # Red for defects
        cv2.imwrite(str(output_path / f"{basename}_overlay.png"), overlay)
        
        print(f"  ✓ Saved to: {output_dir}")
    
    # Visualization
    if visualize and output_dir:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Module 1 Pipeline: {Path(test_path).name}", fontsize=14, fontweight='bold')
        
        axes[0, 0].imshow(template_proc, cmap='gray')
        axes[0, 0].set_title("Template (Defect-free)")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(test_proc, cmap='gray')
        axes[0, 1].set_title("Test (With Defects)")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title("Difference Map")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title(f"Otsu Threshold ({threshold_value})")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cleaned_mask, cmap='gray')
        axes[1, 1].set_title("Cleaned Mask")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Defects Highlighted")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(output_path / f"{basename}_visualization.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        "template_proc": template_proc,
        "test_proc": test_proc,
        "diff": diff,
        "binary_mask": binary_mask,
        "cleaned_mask": cleaned_mask,
        "threshold_value": threshold_value,
        "metrics": diff_metrics
    }


def process_directory(template_dir, test_dir, output_dir):
    """
    Process all image pairs in directories
    
    Args:
        template_dir: Directory with template images
        test_dir: Directory with test images
        output_dir: Directory for outputs
    """
    template_path = Path(template_dir)
    test_path = Path(test_dir)
    
    # Load template mapping
    templates = {}
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        for f in template_path.glob(ext):
            templates[f.stem] = f
    
    print(f"Found {len(templates)} templates")
    
    # Process test images
    test_files = []
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        test_files.extend(list(test_path.glob(ext)))
    
    print(f"Found {len(test_files)} test images")
    
    success = 0
    failed = 0
    
    for test_file in test_files:
        # Extract template ID
        template_id = test_file.name.split('_')[0]
        
        if template_id not in templates:
            print(f"⚠️  No template for {test_file.name}")
            failed += 1
            continue
        
        template_file = templates[template_id]
        
        # Process
        result = process_image_pair(
            str(template_file),
            str(test_file),
            output_dir,
            visualize=True
        )
        
        if result:
            success += 1
        else:
            failed += 1
    
    print("\n" + "="*80)
    print(f"PROCESSING COMPLETE: {success} successful, {failed} failed")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Module 1: Dataset Setup & Image Subtraction")
    parser.add_argument('--template', type=str, help='Template image path (single pair mode)')
    parser.add_argument('--test', type=str, help='Test image path (single pair mode)')
    parser.add_argument('--template_dir', type=str, help='Template directory (batch mode)')
    parser.add_argument('--test_dir', type=str, help='Test directory (batch mode)')
    parser.add_argument('--output', '--output_dir', type=str, required=True, 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Single pair mode
    if args.template and args.test:
        process_image_pair(args.template, args.test, args.output)
    
    # Batch mode
    elif args.template_dir and args.test_dir:
        process_directory(args.template_dir, args.test_dir, args.output)
    
    else:
        print("Error: Provide either (--template + --test) or (--template_dir + --test_dir)")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
