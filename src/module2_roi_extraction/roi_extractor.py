"""
PHASE 4: ROI Extraction
Crops defect regions from original images using bounding boxes
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict


def crop_rois(image: np.ndarray,
              boxes: List[Tuple[int, int, int, int]],
              min_size: Tuple[int, int] = (10, 10)) -> List[np.ndarray]:
    """
    Crop regions of interest from image using bounding boxes
    
    Args:
        image: Source image (template or test)
        boxes: List of (x, y, w, h) bounding boxes
        min_size: (min_width, min_height) to filter tiny ROIs
    
    Returns:
        List of cropped ROI images
    
    Technical Details:
        - ROIs preserve original pixel values
        - Empty/invalid boxes return empty list
        - ROIs smaller than min_size are skipped
    """
    rois = []
    min_w, min_h = min_size
    
    for x, y, w, h in boxes:
        # Skip tiny ROIs
        if w < min_w or h < min_h:
            continue
        
        # Ensure bounds are valid
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            continue
        
        # Crop ROI
        roi = image[y:y+h, x:x+w]
        
        if roi.size > 0:
            rois.append(roi)
    
    return rois


def crop_roi_pairs(template: np.ndarray,
                   test: np.ndarray,
                   boxes: List[Tuple[int, int, int, int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Crop corresponding ROI pairs from template and test images
    
    Useful for keeping template-test alignment in ROIs
    
    Args:
        template: Template image (defect-free)
        test: Test image (with defects)
        boxes: Bounding boxes extracted from difference mask
    
    Returns:
        Tuple of (template_rois, test_rois)
    """
    template_rois = crop_rois(template, boxes)
    test_rois = crop_rois(test, boxes)
    
    # Ensure same number of ROIs
    if len(template_rois) != len(test_rois):
        min_len = min(len(template_rois), len(test_rois))
        template_rois = template_rois[:min_len]
        test_rois = test_rois[:min_len]
    
    return template_rois, test_rois


def save_rois(rois: List[np.ndarray],
              output_dir: str,
              prefix: str = "roi",
              image_name: str = "") -> List[str]:
    """
    Save ROIs to disk with systematic naming
    
    Args:
        rois: List of cropped ROI images
        output_dir: Directory to save ROIs
        prefix: Filename prefix (e.g., "roi", "template_roi")
        image_name: Original image name for tracking
    
    Returns:
        List of saved file paths
    
    File naming: {image_name}_{prefix}_{index:03d}.png
    Example: 01_missing_hole_01_roi_001.png
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    for i, roi in enumerate(rois):
        if image_name:
            filename = f"{image_name}_{prefix}_{i+1:03d}.png"
        else:
            filename = f"{prefix}_{i+1:03d}.png"
        
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, roi)
        saved_paths.append(filepath)
    
    return saved_paths


def extract_roi_metadata(roi: np.ndarray,
                        box: Tuple[int, int, int, int],
                        original_shape: Tuple[int, int]) -> Dict:
    """
    Extract metadata for a single ROI
    
    Useful for dataset tracking and analysis
    
    Args:
        roi: Cropped ROI image
        box: (x, y, w, h) bounding box
        original_shape: (height, width) of source image
    
    Returns:
        Dictionary with metadata:
            - roi_shape: (height, width) of ROI
            - bounding_box: (x, y, w, h) in original coordinates
            - roi_area: Area in pixels
            - position_ratio: (center_x/img_w, center_y/img_h) normalized position
    """
    x, y, w, h = box
    roi_h, roi_w = roi.shape[:2]
    orig_h, orig_w = original_shape
    
    # Center position normalized to [0, 1]
    center_x = (x + w / 2) / orig_w
    center_y = (y + h / 2) / orig_h
    
    return {
        'roi_shape': (roi_h, roi_w),
        'bounding_box': box,
        'roi_area': roi_h * roi_w,
        'position_ratio': (center_x, center_y),
        'original_shape': original_shape
    }


def visualize_roi_grid(rois: List[np.ndarray],
                       max_rois: int = 25,
                       grid_cols: int = 5) -> np.ndarray:
    """
    Create grid visualization of ROIs
    
    Args:
        rois: List of ROI images
        max_rois: Maximum ROIs to show (default: 25)
        grid_cols: Number of columns in grid (default: 5)
    
    Returns:
        Grid image with ROIs arranged in rows
    """
    if len(rois) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Limit number of ROIs
    display_rois = rois[:max_rois]
    
    # Find max dimensions for uniform sizing
    max_h = max(roi.shape[0] for roi in display_rois)
    max_w = max(roi.shape[1] for roi in display_rois)
    
    # Pad ROIs to uniform size
    padded_rois = []
    for roi in display_rois:
        # Convert grayscale to BGR if needed
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        # Pad to max size
        pad_h = max_h - roi.shape[0]
        pad_w = max_w - roi.shape[1]
        
        padded = cv2.copyMakeBorder(roi, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_CONSTANT, value=(50, 50, 50))
        padded_rois.append(padded)
    
    # Arrange in grid
    grid_rows = (len(padded_rois) + grid_cols - 1) // grid_cols
    
    rows = []
    for r in range(grid_rows):
        start_idx = r * grid_cols
        end_idx = min(start_idx + grid_cols, len(padded_rois))
        row_rois = padded_rois[start_idx:end_idx]
        
        # Pad row if needed
        while len(row_rois) < grid_cols:
            row_rois.append(np.zeros_like(padded_rois[0]))
        
        row_img = np.hstack(row_rois)
        rows.append(row_img)
    
    grid = np.vstack(rows)
    
    return grid


if __name__ == "__main__":
    # Test ROI extraction
    import sys
    from contour_detection import find_contours
    from filter_contours import ContourFilter
    from bounding_box import extract_bounding_boxes
    
    if len(sys.argv) < 3:
        print("Usage: python roi_extractor.py <binary_mask> <source_image> [output_dir]")
        print("\nTest: Extracts and saves ROIs from source image")
        sys.exit(1)
    
    # Load images
    mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(sys.argv[2])
    
    if mask is None or source is None:
        print("Error: Could not load images")
        sys.exit(1)
    
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "extracted_rois"
    
    print(f"✓ Loaded mask: {mask.shape}")
    print(f"✓ Loaded source: {source.shape}")
    
    # Extract ROIs
    contours = find_contours(mask)
    print(f"✓ Found {len(contours)} contours")
    
    filter_obj = ContourFilter(min_area=50.0)
    filtered, _ = filter_obj.filter_contours(contours, verbose=False)
    print(f"✓ Filtered to {len(filtered)} contours")
    
    boxes = extract_bounding_boxes(filtered, padding=10, image_shape=mask.shape)
    print(f"✓ Extracted {len(boxes)} bounding boxes")
    
    rois = crop_rois(source, boxes)
    print(f"✓ Cropped {len(rois)} ROIs")
    
    # Save ROIs
    image_name = Path(sys.argv[2]).stem
    saved_paths = save_rois(rois, output_dir, prefix="roi", image_name=image_name)
    
    print(f"\n✓ Saved {len(saved_paths)} ROIs to: {output_dir}/")
    
    # Show ROI statistics
    if len(rois) > 0:
        areas = [roi.shape[0] * roi.shape[1] for roi in rois]
        print(f"\nROI Statistics:")
        print(f"  Count: {len(rois)}")
        print(f"  Avg size: {np.mean([r.shape[1] for r in rois]):.1f} × {np.mean([r.shape[0] for r in rois]):.1f}")
        print(f"  Area range: {min(areas)} - {max(areas)} px²")
        
        # Create grid visualization
        grid = visualize_roi_grid(rois, max_rois=25, grid_cols=5)
        grid_path = os.path.join(output_dir, f"{image_name}_roi_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"  ✓ Saved grid: {grid_path}")
