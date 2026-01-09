"""
Module 2: Complete ROI Extraction Pipeline
End-to-end processing from binary masks to normalized, labeled ROIs
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from typing import List, Tuple
import json

from contour_detection import find_contours, compute_contour_properties, visualize_contours
from filter_contours import ContourFilter, drop_overlapping_contours
from bounding_box import extract_bounding_boxes, compute_box_statistics, visualize_boxes
from roi_extractor import crop_rois, save_rois, visualize_roi_grid, extract_roi_metadata
from roi_normalizer import ROINormalizer, compute_normalization_stats
from label_assignment import (assign_labels_to_rois, create_label_manifest,
                              compute_label_distribution, validate_labels,
                              DEFECT_TYPES)


def process_single_image(mask_path: str,
                        source_image_path: str,
                        output_dir: str,
                        target_size: int = 64,
                        padding: int = 10,
                        min_area: float = 50.0,
                        normalize: bool = True,
                        visualize: bool = True) -> dict:
    """
    Process single image through Module 2 pipeline
    
    Args:
        mask_path: Path to binary mask from Module 1
        source_image_path: Path to source image (test or template)
        output_dir: Output directory for ROIs
        target_size: Target size for normalization (e.g., 64 → 64×64)
        padding: Padding around bounding boxes
        min_area: Minimum contour area to keep
        normalize: Whether to normalize ROIs to fixed size
        visualize: Whether to create visualization images
    
    Returns:
        Dictionary with processing results and statistics
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Load images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(source_image_path)
    
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    if source is None:
        raise ValueError(f"Could not load source: {source_image_path}")
    
    image_name = Path(source_image_path).stem
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}")
    
    # Step 1: Find contours
    print("\n[1/7] Finding contours...")
    contours = find_contours(mask, mode='external')
    print(f"  ✓ Found {len(contours)} contours")
    
    if visualize and len(contours) > 0:
        vis_contours = visualize_contours(mask, contours, color=(0, 255, 0), thickness=2)
        vis_path = os.path.join(vis_dir, f"{image_name}_contours.png")
        cv2.imwrite(vis_path, vis_contours)
    
    # Step 2: Filter contours
    print("\n[2/7] Filtering contours...")
    filter_obj = ContourFilter(
        min_area=min_area,
        max_area=100000.0,
        min_aspect_ratio=0.1,
        max_aspect_ratio=10.0
    )
    filtered_contours, contour_props = filter_obj.filter_contours(contours, verbose=False)
    print(f"  ✓ Filtered: {len(contours)} → {len(filtered_contours)} contours")
    
    if len(filtered_contours) == 0:
        print("  ⚠ No contours after filtering!")
        return {
            'image_name': image_name,
            'num_contours': 0,
            'num_rois': 0,
            'status': 'no_contours'
        }
    
    # Step 3: Extract bounding boxes
    print("\n[3/7] Extracting bounding boxes...")
    boxes = extract_bounding_boxes(filtered_contours, padding=padding, image_shape=mask.shape)
    box_stats = compute_box_statistics(boxes)
    print(f"  ✓ Extracted {len(boxes)} bounding boxes")
    print(f"  ✓ Avg size: {box_stats['avg_width']:.1f} × {box_stats['avg_height']:.1f}")
    
    if visualize and len(boxes) > 0:
        vis_boxes = visualize_boxes(source, boxes, color=(0, 255, 0), thickness=2)
        vis_path = os.path.join(vis_dir, f"{image_name}_boxes.png")
        cv2.imwrite(vis_path, vis_boxes)
    
    # Step 4: Crop ROIs
    print("\n[4/7] Cropping ROIs...")
    rois = crop_rois(source, boxes, min_size=(10, 10))
    print(f"  ✓ Cropped {len(rois)} ROIs")
    
    if len(rois) == 0:
        print("  ⚠ No valid ROIs extracted!")
        return {
            'image_name': image_name,
            'num_contours': len(filtered_contours),
            'num_rois': 0,
            'status': 'no_rois'
        }
    
    # Step 5: Normalize ROIs (optional)
    if normalize:
        print("\n[5/7] Normalizing ROIs...")
        normalizer = ROINormalizer(
            target_size=(target_size, target_size),
            strategy='pad',  # Preserves aspect ratio
            interpolation=cv2.INTER_LINEAR
        )
        normalized_rois = normalizer.normalize_batch(rois)
        norm_stats = compute_normalization_stats(rois, normalized_rois)
        print(f"  ✓ Normalized to {target_size}×{target_size}")
        print(f"  ✓ Avg reduction: {norm_stats['size_reduction_ratio']:.2f}x")
        
        # Save normalized ROIs
        roi_dir = os.path.join(output_dir, 'normalized_rois')
        os.makedirs(roi_dir, exist_ok=True)
        saved_paths = []
        for i, norm_roi in enumerate(normalized_rois):
            filename = f"{image_name}_roi_{i+1:03d}.png"
            filepath = os.path.join(roi_dir, filename)
            cv2.imwrite(filepath, norm_roi)
            saved_paths.append(filename)
    else:
        print("\n[5/7] Skipping normalization...")
        # Save original-size ROIs
        roi_dir = os.path.join(output_dir, 'rois')
        saved_paths = save_rois(rois, roi_dir, prefix="roi", image_name=image_name)
        saved_paths = [os.path.basename(p) for p in saved_paths]
    
    # Step 6: Assign labels
    print("\n[6/7] Assigning labels...")
    labeled_rois = assign_labels_to_rois(saved_paths, source_image_path)
    label = labeled_rois[0]['label'] if labeled_rois else 'unknown'
    print(f"  ✓ Assigned label: {label}")
    
    # Step 7: Create visualizations
    if visualize:
        print("\n[7/7] Creating visualizations...")
        
        # ROI grid
        if normalize:
            grid_rois = [normalized_rois[i] for i in range(min(25, len(normalized_rois)))]
        else:
            grid_rois = rois[:25]
        
        grid = visualize_roi_grid(grid_rois, max_rois=25, grid_cols=5)
        grid_path = os.path.join(vis_dir, f"{image_name}_roi_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"  ✓ Saved ROI grid visualization")
    else:
        print("\n[7/7] Skipping visualization...")
    
    print(f"\n{'='*70}")
    print(f"✓ Complete: Extracted {len(rois)} ROIs from {image_name}")
    print(f"{'='*70}\n")
    
    return {
        'image_name': image_name,
        'num_contours': len(filtered_contours),
        'num_rois': len(rois),
        'label': label,
        'labeled_rois': labeled_rois,
        'box_stats': box_stats,
        'status': 'success'
    }


def process_directory(mask_dir: str,
                     source_dir: str,
                     output_dir: str,
                     target_size: int = 64,
                     padding: int = 10,
                     min_area: float = 50.0,
                     normalize: bool = True) -> dict:
    """
    Process entire directory of masks and source images
    
    Args:
        mask_dir: Directory containing binary masks from Module 1
        source_dir: Directory containing source images
        output_dir: Output directory for all ROIs
        target_size: Target size for normalization
        padding: Padding around bounding boxes
        min_area: Minimum contour area
        normalize: Whether to normalize ROIs
    
    Returns:
        Dictionary with batch processing results
    """
    # Find all mask files
    mask_paths = sorted(Path(mask_dir).glob('*.png'))
    
    if len(mask_paths) == 0:
        print(f"No mask files found in {mask_dir}")
        return {'status': 'no_files'}
    
    print(f"Found {len(mask_paths)} masks to process")
    
    all_results = []
    all_labeled_rois = []
    total_rois = 0
    
    for i, mask_path in enumerate(mask_paths, 1):
        # Find corresponding source image
        mask_name = mask_path.stem
        
        # Try to find source image (handle various naming patterns)
        # Pattern 1: mask is named "01_missing_hole_01_mask.png"
        source_name = mask_name.replace('_mask', '').replace('_binary', '')
        
        # Search for source image
        source_path = None
        for ext in ['.jpg', '.png', '.JPG', '.PNG']:
            candidate = Path(source_dir) / f"{source_name}{ext}"
            if candidate.exists():
                source_path = str(candidate)
                break
        
        if source_path is None:
            print(f"\n[{i}/{len(mask_paths)}] ⚠ Skipping {mask_name}: source not found")
            continue
        
        print(f"\n[{i}/{len(mask_paths)}]")
        
        # Process image
        try:
            result = process_single_image(
                mask_path=str(mask_path),
                source_image_path=source_path,
                output_dir=output_dir,
                target_size=target_size,
                padding=padding,
                min_area=min_area,
                normalize=normalize,
                visualize=False  # Disable per-image visualization in batch mode
            )
            
            all_results.append(result)
            
            if result['status'] == 'success':
                all_labeled_rois.extend(result['labeled_rois'])
                total_rois += result['num_rois']
        
        except Exception as e:
            print(f"  ✗ Error processing {mask_name}: {e}")
            all_results.append({
                'image_name': mask_name,
                'status': 'error',
                'error': str(e)
            })
    
    # Create label manifest
    if all_labeled_rois:
        manifest_path = os.path.join(output_dir, 'label_manifest.json')
        create_label_manifest(all_labeled_rois, manifest_path)
        
        # Validate labels
        validation = validate_labels(all_labeled_rois)
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total images processed: {len(all_results)}")
        print(f"Total ROIs extracted: {total_rois}")
        print(f"Valid labels: {validation['valid_count']}")
        print(f"Unknown labels: {validation['unknown_count']}")
        
        print(f"\nLabel Distribution:")
        for label, count in validation['distribution'].items():
            if count > 0:
                print(f"  {label}: {count}")
        
        if validation['warnings']:
            print(f"\n⚠ Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        print(f"\n✓ Label manifest saved: {manifest_path}")
        print(f"{'='*70}\n")
    
    return {
        'status': 'complete',
        'num_processed': len(all_results),
        'total_rois': total_rois,
        'results': all_results,
        'labeled_rois': all_labeled_rois
    }


def main():
    parser = argparse.ArgumentParser(
        description='Module 2: ROI Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python pipeline.py --mask mask.png --source image.jpg --output rois/
  
  # Process directory (batch)
  python pipeline.py --mask_dir masks/ --source_dir images/ --output rois/
  
  # Custom settings
  python pipeline.py --mask mask.png --source image.jpg --output rois/ \\
    --size 128 --padding 15 --min_area 100
        """
    )
    
    # Input arguments
    parser.add_argument('--mask', type=str,
                       help='Path to binary mask from Module 1')
    parser.add_argument('--source', type=str,
                       help='Path to source image')
    parser.add_argument('--mask_dir', type=str,
                       help='Directory containing masks (batch mode)')
    parser.add_argument('--source_dir', type=str,
                       help='Directory containing source images (batch mode)')
    
    # Output arguments
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for ROIs')
    
    # Processing parameters
    parser.add_argument('--size', type=int, default=64,
                       help='Target size for normalized ROIs (default: 64)')
    parser.add_argument('--padding', type=int, default=10,
                       help='Padding around bounding boxes (default: 10)')
    parser.add_argument('--min_area', type=float, default=50.0,
                       help='Minimum contour area in pixels (default: 50)')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Skip ROI normalization (keep original sizes)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization images (single image mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mask and args.source:
        # Single image mode
        result = process_single_image(
            mask_path=args.mask,
            source_image_path=args.source,
            output_dir=args.output,
            target_size=args.size,
            padding=args.padding,
            min_area=args.min_area,
            normalize=not args.no_normalize,
            visualize=args.visualize
        )
        
        print(f"\n✓ Result: {result['status']}")
        if result['status'] == 'success':
            print(f"  Extracted {result['num_rois']} ROIs")
            print(f"  Label: {result['label']}")
    
    elif args.mask_dir and args.source_dir:
        # Batch mode
        result = process_directory(
            mask_dir=args.mask_dir,
            source_dir=args.source_dir,
            output_dir=args.output,
            target_size=args.size,
            padding=args.padding,
            min_area=args.min_area,
            normalize=not args.no_normalize
        )
    
    else:
        parser.print_help()
        print("\nError: Provide either (--mask + --source) or (--mask_dir + --source_dir)")
        sys.exit(1)


if __name__ == "__main__":
    main()
