"""
Image Subtraction
Computes absolute difference between template and test images
"""

import cv2
import numpy as np


def subtract_images(template, test):
    """
    Compute absolute difference between images
    
    Formula: diff(x,y) = |test(x,y) - template(x,y)|
    
    Args:
        template: Preprocessed template image (grayscale)
        test: Preprocessed test image (grayscale)
    
    Returns:
        Difference map (uint8)
    """
    if template.shape != test.shape:
        raise ValueError(f"Shape mismatch: {template.shape} vs {test.shape}")
    
    # Compute absolute difference
    diff = cv2.absdiff(test, template)
    
    return diff


def compute_metrics(diff):
    """
    Compute statistics for difference map
    
    Args:
        diff: Difference map
    
    Returns:
        Dictionary of metrics
    """
    return {
        "mean": float(np.mean(diff)),
        "std": float(np.std(diff)),
        "min": int(np.min(diff)),
        "max": int(np.max(diff)),
        "non_zero_percentage": float(np.count_nonzero(diff) / diff.size * 100)
    }


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python subtraction.py <template_path> <test_path>")
        sys.exit(1)
    
    from preprocess import preprocess_pair
    
    template = cv2.imread(sys.argv[1])
    test = cv2.imread(sys.argv[2])
    
    if template is None or test is None:
        print("Failed to load images")
        sys.exit(1)
    
    # Preprocess
    template_proc, test_proc = preprocess_pair(template, test)
    
    # Subtract
    diff = subtract_images(template_proc, test_proc)
    metrics = compute_metrics(diff)
    
    print("✓ Difference computed")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save
    cv2.imwrite("difference_map.png", diff)
    print("\n✓ Saved: difference_map.png")
