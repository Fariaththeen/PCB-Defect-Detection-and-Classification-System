"""
PHASE 2: Contour Filtering
Filters out noise and invalid contours based on area, aspect ratio, etc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from contour_detection import compute_contour_properties


class ContourFilter:
    """
    Filters contours based on multiple criteria
    
    Default thresholds tuned for PCB defects:
        - Min area: 50 px² (removes tiny noise)
        - Max area: 100,000 px² (removes full-image masks)
        - Min aspect ratio: 0.1 (very elongated OK)
        - Max aspect ratio: 10.0 (very elongated OK)
    """
    
    def __init__(self,
                 min_area: float = 50.0,
                 max_area: float = 100000.0,
                 min_aspect_ratio: float = 0.1,
                 max_aspect_ratio: float = 10.0,
                 min_width: int = 5,
                 min_height: int = 5):
        """
        Initialize filter with thresholds
        
        Args:
            min_area: Minimum contour area in pixels
            max_area: Maximum contour area in pixels
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            min_width: Minimum bounding box width
            min_height: Minimum bounding box height
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_width = min_width
        self.min_height = min_height
    
    def filter_contours(self, contours: List[np.ndarray], 
                       verbose: bool = True) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Filter contours based on initialized thresholds
        
        Args:
            contours: List of contours from find_contours()
            verbose: Print filtering statistics
        
        Returns:
            Tuple of (filtered_contours, properties_list)
        """
        filtered = []
        properties = []
        
        stats = {
            'total': len(contours),
            'too_small': 0,
            'too_large': 0,
            'bad_aspect_ratio': 0,
            'too_thin': 0,
            'passed': 0
        }
        
        for contour in contours:
            props = compute_contour_properties(contour)
            x, y, w, h = props['bounding_rect']
            area = props['area']
            aspect_ratio = props['aspect_ratio']
            
            # Area check
            if area < self.min_area:
                stats['too_small'] += 1
                continue
            
            if area > self.max_area:
                stats['too_large'] += 1
                continue
            
            # Aspect ratio check
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                stats['bad_aspect_ratio'] += 1
                continue
            
            # Dimension check
            if w < self.min_width or h < self.min_height:
                stats['too_thin'] += 1
                continue
            
            # Passed all filters
            filtered.append(contour)
            properties.append(props)
            stats['passed'] += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CONTOUR FILTERING RESULTS")
            print(f"{'='*60}")
            print(f"Total contours:        {stats['total']}")
            print(f"✓ Passed filters:      {stats['passed']} ({stats['passed']/max(stats['total'],1)*100:.1f}%)")
            print(f"\nRejection reasons:")
            print(f"  - Too small (<{self.min_area} px²):    {stats['too_small']}")
            print(f"  - Too large (>{self.max_area} px²):  {stats['too_large']}")
            print(f"  - Bad aspect ratio:  {stats['bad_aspect_ratio']}")
            print(f"  - Too thin:          {stats['too_thin']}")
            print(f"{'='*60}\n")
        
        return filtered, properties
    
    def filter_by_area_percentile(self, contours: List[np.ndarray],
                                   lower_percentile: float = 5.0,
                                   upper_percentile: float = 95.0) -> List[np.ndarray]:
        """
        Filter contours by area percentiles (adaptive filtering)
        
        Useful when defect sizes vary significantly across images
        
        Args:
            contours: List of contours
            lower_percentile: Remove smallest X% of contours
            upper_percentile: Remove largest Y% of contours
        
        Returns:
            Filtered contours
        """
        if len(contours) == 0:
            return []
        
        # Compute all areas
        areas = [cv2.contourArea(c) for c in contours]
        
        # Compute percentile thresholds
        lower_thresh = np.percentile(areas, lower_percentile)
        upper_thresh = np.percentile(areas, upper_percentile)
        
        # Filter
        filtered = [c for c, a in zip(contours, areas) 
                   if lower_thresh <= a <= upper_thresh]
        
        print(f"Percentile filtering: {len(contours)} → {len(filtered)} contours")
        print(f"  Area range: {lower_thresh:.1f} - {upper_thresh:.1f} px²")
        
        return filtered


def drop_overlapping_contours(contours: List[np.ndarray],
                              iou_threshold: float = 0.5) -> List[np.ndarray]:
    """
    Remove overlapping contours (keep larger one)
    
    Useful when defects are close together and their bounding boxes overlap
    
    Args:
        contours: List of contours
        iou_threshold: IoU threshold for considering overlap (0.5 = 50% overlap)
    
    Returns:
        Non-overlapping contours
    """
    if len(contours) <= 1:
        return contours
    
    # Get bounding boxes and areas
    boxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]
    
    # Sort by area (largest first)
    indices = np.argsort(areas)[::-1]
    
    keep = []
    
    for i in indices:
        # Check if this contour overlaps with any kept contour
        x1, y1, w1, h1 = boxes[i]
        overlaps = False
        
        for j in keep:
            x2, y2, w2, h2 = boxes[j]
            
            # Compute IoU
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                overlaps = True
                break
        
        if not overlaps:
            keep.append(i)
    
    # Return kept contours
    kept_contours = [contours[i] for i in keep]
    
    print(f"Overlap filtering: {len(contours)} → {len(kept_contours)} contours")
    print(f"  Removed {len(contours) - len(kept_contours)} overlapping contours")
    
    return kept_contours


if __name__ == "__main__":
    # Test filtering
    import sys
    from contour_detection import find_contours
    
    if len(sys.argv) < 2:
        print("Usage: python filter_contours.py <binary_mask_path>")
        print("\nTest: Filters contours by area, aspect ratio, etc.")
        sys.exit(1)
    
    # Load mask
    mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error: Could not load {sys.argv[1]}")
        sys.exit(1)
    
    print(f"✓ Loaded mask: {mask.shape}")
    
    # Find contours
    contours = find_contours(mask)
    print(f"✓ Found {len(contours)} contours")
    
    # Create filter
    filter_obj = ContourFilter(
        min_area=50.0,
        max_area=100000.0,
        min_aspect_ratio=0.1,
        max_aspect_ratio=10.0
    )
    
    # Filter contours
    filtered, props = filter_obj.filter_contours(contours, verbose=True)
    
    print(f"\n✓ Filtering complete: {len(contours)} → {len(filtered)} contours")
    
    # Show properties of filtered contours
    if len(filtered) > 0:
        print("\nFiltered Contour Details:")
        print("-" * 80)
        for i, prop in enumerate(props[:5]):  # Show first 5
            print(f"\nContour {i+1}:")
            print(f"  Area: {prop['area']:.1f} px²")
            print(f"  Bounding box: {prop['bounding_rect']}")
            print(f"  Aspect ratio: {prop['aspect_ratio']:.2f}")
            print(f"  Circularity: {prop['circularity']:.3f}")
        
        if len(filtered) > 5:
            print(f"\n... and {len(filtered) - 5} more contours")
