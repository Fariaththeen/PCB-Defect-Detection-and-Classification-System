"""
Morphological Post-processing
Cleans binary masks by removing noise and filling gaps
"""

import cv2
import numpy as np


def clean_mask(binary_mask, kernel_size=(3, 3), open_iterations=2, dilate_iterations=1):
    """
    Clean binary mask using morphological operations
    
    Pipeline:
    1. Opening (erosion + dilation) - removes noise
    2. Dilation - strengthens remaining defects
    
    Args:
        binary_mask: Binary mask from thresholding
        kernel_size: Morphological kernel size
        open_iterations: Iterations for opening operation
        dilate_iterations: Iterations for final dilation
    
    Returns:
        Cleaned binary mask
    """
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Opening: removes small noise dots
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, 
                               iterations=open_iterations)
    
    # Dilation: strengthen defect boundaries
    cleaned = cv2.dilate(cleaned, kernel, iterations=dilate_iterations)
    
    return cleaned


def remove_small_objects(mask, min_area=10):
    """
    Remove connected components smaller than min_area
    
    Args:
        mask: Binary mask
        min_area: Minimum component size in pixels
    
    Returns:
        Mask with small objects removed
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Create output mask
    cleaned = np.zeros_like(mask)
    
    # Keep only large components (skip label 0 = background)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255
    
    return cleaned


def postprocess(binary_mask, min_defect_area=10):
    """
    Complete post-processing pipeline
    
    Args:
        binary_mask: Binary mask from thresholding
        min_defect_area: Minimum defect size in pixels
    
    Returns:
        Clean binary mask
    """
    # Morphological cleaning
    cleaned = clean_mask(binary_mask)
    
    # Remove small objects
    cleaned = remove_small_objects(cleaned, min_area=min_defect_area)
    
    return cleaned


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python postprocess.py <binary_mask_path>")
        sys.exit(1)
    
    binary_mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    
    if binary_mask is None:
        print("Failed to load binary mask")
        sys.exit(1)
    
    # Post-process
    cleaned_mask = postprocess(binary_mask)
    
    original_white = np.sum(binary_mask == 255)
    cleaned_white = np.sum(cleaned_mask == 255)
    removed = original_white - cleaned_white
    
    print(f"✓ Post-processing complete")
    print(f"  Original white pixels: {original_white}")
    print(f"  Cleaned white pixels: {cleaned_white}")
    print(f"  Removed: {removed} ({removed/max(original_white,1)*100:.1f}%)")
    
    # Save
    cv2.imwrite("cleaned_mask.png", cleaned_mask)
    print("\n✓ Saved: cleaned_mask.png")
