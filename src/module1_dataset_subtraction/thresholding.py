"""
Thresholding
Converts grayscale difference maps to binary masks using Otsu's method
"""

import cv2
import numpy as np


def otsu_threshold(diff):
    """
    Apply Otsu's automatic thresholding
    
    Otsu's method automatically determines optimal threshold by
    minimizing intra-class variance
    
    Args:
        diff: Grayscale difference map
    
    Returns:
        Tuple of (binary_mask, threshold_value)
    """
    threshold_value, binary_mask = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary_mask, int(threshold_value)


def compute_mask_stats(binary_mask):
    """
    Compute statistics for binary mask
    
    Args:
        binary_mask: Binary mask (0 or 255)
    
    Returns:
        Dictionary of statistics
    """
    white_pixels = np.sum(binary_mask == 255)
    total_pixels = binary_mask.size
    
    return {
        "white_pixels": int(white_pixels),
        "black_pixels": int(total_pixels - white_pixels),
        "white_percentage": float(white_pixels / total_pixels * 100)
    }


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python thresholding.py <difference_map_path>")
        sys.exit(1)
    
    diff = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    
    if diff is None:
        print("Failed to load difference map")
        sys.exit(1)
    
    # Threshold
    binary_mask, threshold_value = otsu_threshold(diff)
    stats = compute_mask_stats(binary_mask)
    
    print(f"✓ Otsu threshold value: {threshold_value}")
    print("\nMask Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save
    cv2.imwrite("binary_mask.png", binary_mask)
    print("\n✓ Saved: binary_mask.png")
