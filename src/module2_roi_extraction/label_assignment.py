"""
PHASE 6: Label Assignment
Assigns defect type labels to extracted ROIs based on filename patterns
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


# Standard DeepPCB defect types
DEFECT_TYPES = [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper'
]

# Label to index mapping for ML models
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(DEFECT_TYPES)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}


def extract_label_from_filename(filename: str) -> Optional[str]:
    """
    Extract defect type label from filename
    
    Filename patterns:
        - 01_missing_hole_01.jpg → missing_hole
        - 04_mouse_bite_05.jpg → mouse_bite
        - short_test_01.jpg → short
    
    Args:
        filename: Image filename
    
    Returns:
        Defect type string or None if not found
    """
    filename_lower = filename.lower()
    
    # Try each defect type
    for defect_type in DEFECT_TYPES:
        if defect_type in filename_lower:
            return defect_type
    
    return None


def extract_label_from_path(filepath: str) -> Optional[str]:
    """
    Extract defect type from file path (directory structure)
    
    Path patterns:
        - .../Missing_hole/01_missing_hole_01.jpg → missing_hole
        - .../Open_circuit/05_open_circuit_03.jpg → open_circuit
    
    Args:
        filepath: Full file path
    
    Returns:
        Defect type string or None if not found
    """
    path_lower = filepath.lower()
    
    # Check directory names
    for defect_type in DEFECT_TYPES:
        # Handle both underscored and CamelCase versions
        if defect_type in path_lower:
            return defect_type
        
        # Check CamelCase version (e.g., Missing_hole)
        camel_case = defect_type.replace('_', ' ').title().replace(' ', '_')
        if camel_case.lower() in path_lower:
            return defect_type
    
    return None


def assign_labels_to_rois(roi_filenames: List[str],
                          source_image_path: str) -> List[Dict[str, any]]:
    """
    Assign labels to ROIs based on source image
    
    Args:
        roi_filenames: List of ROI filenames
        source_image_path: Path to original source image
    
    Returns:
        List of dictionaries with:
            - roi_filename: ROI filename
            - label: Defect type string
            - label_index: Integer index for ML
            - source_image: Original image filename
    """
    # Extract label from source image
    label = extract_label_from_path(source_image_path)
    if label is None:
        label = extract_label_from_filename(os.path.basename(source_image_path))
    
    if label is None:
        label = 'unknown'
        label_index = -1
    else:
        label_index = LABEL_TO_INDEX.get(label, -1)
    
    source_basename = os.path.basename(source_image_path)
    
    # Assign same label to all ROIs from this image
    labeled_rois = []
    for roi_filename in roi_filenames:
        labeled_rois.append({
            'roi_filename': roi_filename,
            'label': label,
            'label_index': label_index,
            'source_image': source_basename
        })
    
    return labeled_rois


def create_label_manifest(labeled_rois: List[Dict[str, any]],
                         output_path: str) -> None:
    """
    Create JSON manifest file with all ROI labels
    
    Useful for dataset loading in Module 3
    
    Args:
        labeled_rois: List of labeled ROI dictionaries
        output_path: Path to save JSON manifest
    """
    manifest = {
        'num_rois': len(labeled_rois),
        'num_classes': len(DEFECT_TYPES),
        'classes': DEFECT_TYPES,
        'label_mapping': LABEL_TO_INDEX,
        'rois': labeled_rois
    }
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created label manifest: {output_path}")


def compute_label_distribution(labeled_rois: List[Dict[str, any]]) -> Dict[str, int]:
    """
    Compute distribution of labels in dataset
    
    Args:
        labeled_rois: List of labeled ROI dictionaries
    
    Returns:
        Dictionary mapping label → count
    """
    distribution = {defect: 0 for defect in DEFECT_TYPES}
    distribution['unknown'] = 0
    
    for roi_data in labeled_rois:
        label = roi_data['label']
        if label in distribution:
            distribution[label] += 1
        else:
            distribution['unknown'] += 1
    
    return distribution


def validate_labels(labeled_rois: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Validate label assignments and detect issues
    
    Args:
        labeled_rois: List of labeled ROI dictionaries
    
    Returns:
        Dictionary with validation results:
            - valid_count: Number of valid labels
            - unknown_count: Number of unknown labels
            - missing_labels: List of defect types with 0 ROIs
            - warnings: List of warning messages
    """
    distribution = compute_label_distribution(labeled_rois)
    
    valid_count = sum(count for label, count in distribution.items() if label != 'unknown')
    unknown_count = distribution['unknown']
    
    missing_labels = [label for label in DEFECT_TYPES if distribution[label] == 0]
    
    warnings = []
    if unknown_count > 0:
        warnings.append(f"{unknown_count} ROIs have unknown labels")
    
    if missing_labels:
        warnings.append(f"Missing labels: {', '.join(missing_labels)}")
    
    # Check for imbalanced classes
    if valid_count > 0:
        max_count = max(distribution[label] for label in DEFECT_TYPES)
        min_count = min(distribution[label] for label in DEFECT_TYPES if distribution[label] > 0)
        
        if max_count > 0 and min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 5.0:
                warnings.append(f"Class imbalance detected: {imbalance_ratio:.1f}x difference")
    
    return {
        'valid_count': valid_count,
        'unknown_count': unknown_count,
        'missing_labels': missing_labels,
        'warnings': warnings,
        'distribution': distribution
    }


def organize_rois_by_label(roi_dir: str, labeled_rois: List[Dict[str, any]]) -> None:
    """
    Organize ROIs into subdirectories by label
    
    Creates structure:
        roi_dir/
            missing_hole/
                roi_001.png
                roi_002.png
            mouse_bite/
                roi_001.png
            ...
    
    Args:
        roi_dir: Directory containing ROI files
        labeled_rois: List of labeled ROI dictionaries
    """
    import shutil
    
    # Create label subdirectories
    for label in DEFECT_TYPES:
        label_dir = os.path.join(roi_dir, label)
        os.makedirs(label_dir, exist_ok=True)
    
    # Create unknown directory
    unknown_dir = os.path.join(roi_dir, 'unknown')
    os.makedirs(unknown_dir, exist_ok=True)
    
    # Move/copy ROIs to labeled folders
    moved_count = 0
    for roi_data in labeled_rois:
        roi_filename = roi_data['roi_filename']
        label = roi_data['label']
        
        source_path = os.path.join(roi_dir, roi_filename)
        
        if not os.path.exists(source_path):
            continue
        
        dest_dir = os.path.join(roi_dir, label)
        dest_path = os.path.join(dest_dir, roi_filename)
        
        # Copy (not move) to preserve originals
        shutil.copy2(source_path, dest_path)
        moved_count += 1
    
    print(f"✓ Organized {moved_count} ROIs into {len(DEFECT_TYPES)} label folders")


if __name__ == "__main__":
    # Test label assignment
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python label_assignment.py <image_path_or_filename>")
        print("\nTest: Extracts label from filename/path")
        print("\nExample filenames:")
        print("  01_missing_hole_01.jpg")
        print("  04_mouse_bite_05.jpg")
        print("  ../images/Open_circuit/05_open_circuit_03.jpg")
        sys.exit(1)
    
    test_input = sys.argv[1]
    
    print(f"Input: {test_input}")
    print(f"\n{'='*60}")
    
    # Try extracting from path
    label_from_path = extract_label_from_path(test_input)
    print(f"Label from path: {label_from_path}")
    
    # Try extracting from filename
    label_from_filename = extract_label_from_filename(os.path.basename(test_input))
    print(f"Label from filename: {label_from_filename}")
    
    # Final label
    final_label = label_from_path or label_from_filename or 'unknown'
    label_index = LABEL_TO_INDEX.get(final_label, -1)
    
    print(f"\n{'='*60}")
    print(f"✓ Final Label: {final_label}")
    print(f"✓ Label Index: {label_index}")
    
    # Show all available labels
    print(f"\n{'='*60}")
    print("Available Defect Types:")
    for i, defect in enumerate(DEFECT_TYPES):
        print(f"  {i}: {defect}")
