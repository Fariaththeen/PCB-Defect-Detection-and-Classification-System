"""
PHASE 5: ROI Normalization
Resizes ROIs to fixed dimensions for ML model input
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class ROINormalizer:
    """
    Normalizes ROIs to fixed size with various strategies
    
    Supports:
        - Resize: Direct resize (may distort aspect ratio)
        - Pad: Pad to square, then resize (preserves aspect ratio)
        - Crop: Center crop to square, then resize (may lose edges)
    """
    
    def __init__(self,
                 target_size: Tuple[int, int] = (64, 64),
                 strategy: str = 'pad',
                 interpolation: int = cv2.INTER_LINEAR):
        """
        Initialize normalizer
        
        Args:
            target_size: (width, height) for output ROIs
            strategy: Normalization strategy
                - 'resize': Direct resize (fastest, may distort)
                - 'pad': Pad to square then resize (preserves aspect, adds borders)
                - 'crop': Center crop then resize (may lose info)
            interpolation: OpenCV interpolation method
                - cv2.INTER_LINEAR (default, good quality/speed)
                - cv2.INTER_CUBIC (higher quality, slower)
                - cv2.INTER_AREA (best for downscaling)
        """
        self.target_size = target_size
        self.strategy = strategy
        self.interpolation = interpolation
        
        if strategy not in ['resize', 'pad', 'crop']:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'resize', 'pad', or 'crop'")
    
    def normalize(self, roi: np.ndarray) -> np.ndarray:
        """
        Normalize single ROI to target size
        
        Args:
            roi: Input ROI image (any size)
        
        Returns:
            Normalized ROI at target_size
        """
        if self.strategy == 'resize':
            return self._resize(roi)
        elif self.strategy == 'pad':
            return self._pad_resize(roi)
        elif self.strategy == 'crop':
            return self._crop_resize(roi)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def normalize_batch(self, rois: List[np.ndarray]) -> np.ndarray:
        """
        Normalize batch of ROIs to fixed size
        
        Args:
            rois: List of ROI images (variable sizes)
        
        Returns:
            NumPy array of shape (N, H, W, C) or (N, H, W) for grayscale
        """
        if len(rois) == 0:
            # Return empty array with correct shape
            if len(self.target_size) == 2:
                return np.empty((0, self.target_size[1], self.target_size[0]), dtype=np.uint8)
            else:
                return np.empty((0, self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        normalized = [self.normalize(roi) for roi in rois]
        return np.array(normalized)
    
    def _resize(self, roi: np.ndarray) -> np.ndarray:
        """Direct resize (may distort aspect ratio)"""
        return cv2.resize(roi, self.target_size, interpolation=self.interpolation)
    
    def _pad_resize(self, roi: np.ndarray) -> np.ndarray:
        """
        Pad to square, then resize (preserves aspect ratio)
        
        Best for preserving defect shape without distortion
        """
        h, w = roi.shape[:2]
        
        # Make square by padding
        if h > w:
            # Add padding to width
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            
            if len(roi.shape) == 2:
                roi_square = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right,
                                               cv2.BORDER_CONSTANT, value=0)
            else:
                roi_square = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif w > h:
            # Add padding to height
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            
            if len(roi.shape) == 2:
                roi_square = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0,
                                               cv2.BORDER_CONSTANT, value=0)
            else:
                roi_square = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            roi_square = roi
        
        # Resize to target
        return cv2.resize(roi_square, self.target_size, interpolation=self.interpolation)
    
    def _crop_resize(self, roi: np.ndarray) -> np.ndarray:
        """
        Center crop to square, then resize (may lose edges)
        
        Use when edges are not important
        """
        h, w = roi.shape[:2]
        
        # Crop to square
        if h > w:
            # Crop height
            start = (h - w) // 2
            roi_square = roi[start:start+w, :]
        elif w > h:
            # Crop width
            start = (w - h) // 2
            roi_square = roi[:, start:start+h]
        else:
            roi_square = roi
        
        # Resize to target
        return cv2.resize(roi_square, self.target_size, interpolation=self.interpolation)


def compute_normalization_stats(original_rois: List[np.ndarray],
                                normalized_rois: np.ndarray) -> dict:
    """
    Compute statistics comparing original and normalized ROIs
    
    Args:
        original_rois: List of original variable-size ROIs
        normalized_rois: NumPy array of normalized fixed-size ROIs
    
    Returns:
        Dictionary with statistics
    """
    if len(original_rois) == 0:
        return {'count': 0}
    
    orig_areas = [roi.shape[0] * roi.shape[1] for roi in original_rois]
    norm_area = normalized_rois.shape[1] * normalized_rois.shape[2]
    
    orig_aspect_ratios = [roi.shape[1] / roi.shape[0] for roi in original_rois]
    
    return {
        'count': len(original_rois),
        'original_avg_area': float(np.mean(orig_areas)),
        'original_area_range': (float(np.min(orig_areas)), float(np.max(orig_areas))),
        'normalized_area': norm_area,
        'avg_aspect_ratio': float(np.mean(orig_aspect_ratios)),
        'aspect_ratio_std': float(np.std(orig_aspect_ratios)),
        'size_reduction_ratio': float(np.mean(orig_areas) / norm_area)
    }


if __name__ == "__main__":
    # Test normalization
    import sys
    import os
    from pathlib import Path
    from roi_extractor import crop_rois
    from contour_detection import find_contours
    from filter_contours import ContourFilter
    from bounding_box import extract_bounding_boxes
    
    if len(sys.argv) < 3:
        print("Usage: python roi_normalizer.py <binary_mask> <source_image> [strategy] [size]")
        print("\nStrategies: resize, pad, crop")
        print("Size: target dimension (default: 64)")
        sys.exit(1)
    
    # Load images
    mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(sys.argv[2])
    
    if mask is None or source is None:
        print("Error: Could not load images")
        sys.exit(1)
    
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'pad'
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    
    print(f"✓ Loaded images")
    print(f"✓ Strategy: {strategy}")
    print(f"✓ Target size: {size}×{size}")
    
    # Extract ROIs
    contours = find_contours(mask)
    filter_obj = ContourFilter(min_area=50.0)
    filtered, _ = filter_obj.filter_contours(contours, verbose=False)
    boxes = extract_bounding_boxes(filtered, padding=10, image_shape=mask.shape)
    rois = crop_rois(source, boxes)
    
    print(f"✓ Extracted {len(rois)} ROIs")
    
    if len(rois) == 0:
        print("No ROIs extracted!")
        sys.exit(1)
    
    # Normalize
    normalizer = ROINormalizer(target_size=(size, size), strategy=strategy)
    normalized = normalizer.normalize_batch(rois)
    
    print(f"✓ Normalized to shape: {normalized.shape}")
    
    # Compute stats
    stats = compute_normalization_stats(rois, normalized)
    print(f"\nNormalization Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Original avg area: {stats['original_avg_area']:.1f} px²")
    print(f"  Original area range: {stats['original_area_range'][0]:.0f} - {stats['original_area_range'][1]:.0f} px²")
    print(f"  Normalized area: {stats['normalized_area']} px²")
    print(f"  Avg aspect ratio: {stats['avg_aspect_ratio']:.2f}")
    print(f"  Size reduction: {stats['size_reduction_ratio']:.2f}x")
    
    # Save samples
    output_dir = "normalized_rois"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, norm_roi in enumerate(normalized[:10]):  # Save first 10
        filename = f"normalized_{strategy}_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, norm_roi)
    
    print(f"\n✓ Saved {min(10, len(normalized))} normalized ROIs to: {output_dir}/")
