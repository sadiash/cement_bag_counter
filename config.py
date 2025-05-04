#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adjusted Configuration file for Cement Bag Counter

This file contains parameters adjusted to fix the double-counting issue
and improve region-based counting in the cement bag detection system.
"""

# Optimized detection parameters for cement bags with motion blur
# These parameters are specifically tuned for the user's conveyor belt setup
DETECTION_PARAMS = {
    'history': 10,           # Shorter history for faster adaptation to changes
    'var_threshold': 6,       # Much lower threshold to detect subtle movements
    'detect_shadows': False,  # Turn off shadow detection to improve performance
    'min_area': 3000,         # Reduced to catch smaller/partial bag appearances
    'max_area': 25000,        # Adjusted based on observed bag size
    'aspect_ratio_range': (0.4, 1.8),  # Wider range to account for motion blur and angle
    'learning_rate': 0.03     # Faster learning rate to adapt to changing conditions
}

# Adjusted tracking parameters to fix double-counting
TRACKING_PARAMS = {
    'max_age': 30,            # Increased from 15 to 30 to maintain identity longer
    'min_hits': 3,            # Increased from 2 to 3 to require more consistent detections
    'iou_threshold': 0.25     # Decreased from 0.3 to be more lenient with matching
}

# Adjusted line counting parameters to fix double-counting
LINE_COUNTING_PARAMS = {
    'line_position': ((0, 240), (640, 240)),  # Horizontal line in the middle
    'direction': 'bottom',       # Count bags moving in any direction
    'min_distance': 100,       # Increased from 3 to 10 to require more movement
    'cooldown_frames': 10     # Increased from 20 to 40 to prevent double-counting
}

""" # Adjusted region counting parameters for better detection
REGION_COUNTING_PARAMS = {
    # Make the region larger and position it where the bags are visible
    'region': (200, 150, 300, 250),  # (x, y, width, height) - covering the left side where bags appear
    'entry_direction': 'bottom',  # Count bags entering from any direction
    'min_frames_in_region': 5,  # Only require one frame in region to be counted
    'cooldown_frames': 40     # Same cooldown as line counter to prevent double-counting
} """

# Adjusted region counting parameters for better detection
REGION_COUNTING_PARAMS = {
    # Make the region larger and position it where the bags are visible
    'region': (200, 80, 300, 280),  # (x, y, width, height) - covering the left side where bags appear
    'entry_direction': 'bottom',  # Count bags entering from any direction
    'min_frames_in_region': 4,  # to play with - at least 4 frames in region to be counted
    'cooldown_frames': 20     # to play with - same cooldown as line counter to prevent double-counting
}

# Region of interest parameters to focus detection
# This helps eliminate false detections from surrounding areas
ROI_PARAMS = {
    'use_roi': True,
    'roi': (195, 75, 360, 340)  # to play with - a bit bigger than the region
}

# Pre-processing parameters to enhance detection
PREPROCESSING_PARAMS = {
    'apply_blur': True,
    'blur_kernel_size': 3,
    'apply_contrast_enhancement': True,
    'contrast_clip_limit': 2.0,
    'apply_morphology': True,
    'morph_kernel_size': 5
}

# Alternative approach: Contour-based detection parameters
# This approach may work better than background subtraction in some cases
CONTOUR_DETECTION_PARAMS = {
    'threshold_min': 120,     # Minimum threshold for binary conversion
    'threshold_max': 255,     # Maximum threshold for binary conversion
    'min_area': 3000,         # Minimum contour area
    'max_area': 25000,        # Maximum contour area
    'aspect_ratio_range': (0.4, 1.8),  # Width/height ratio range
    'solidity_min': 0.7,      # Minimum solidity (area/convex hull area)
    'use_adaptive_threshold': True,  # Use adaptive thresholding
    'adaptive_block_size': 19,  # Block size for adaptive threshold
    'adaptive_c': 2           # Constant subtracted from mean
}

# Frame differencing parameters (alternative to background subtraction)
FRAME_DIFF_PARAMS = {
    'enabled': True,
    'history_length': 3,      # Number of previous frames to compare
    'diff_threshold': 20,     # Minimum difference to consider as movement
    'use_absolute_diff': True # Use absolute difference instead of background subtraction
}

# Hybrid detection approach parameters
HYBRID_DETECTION_PARAMS = {
    'use_background_subtraction': True,
    'use_frame_differencing': True,
    'use_contour_detection': True,
    'confidence_threshold': 0.3,  # Minimum confidence to consider a detection valid
    'combine_method': 'union'     # How to combine detections: 'union', 'intersection', 'weighted'
}

# Debug visualization parameters
DEBUG_PARAMS = {
    'show_background_mask': True,
    'show_frame_diff': True,
    'show_contours': True,
    'show_roi': True,
    'show_preprocessing': True
}

# Additional parameters for extreme cases of double-counting
# Only use these if the standard adjustments don't solve the issue
EXTREME_ANTI_DOUBLE_COUNTING = {
    'cooldown_frames': 60,    # Very long cooldown period
    'min_distance': 15,       # Require significant movement
    'max_age': 40,            # Very long track lifetime
    'min_hits': 4,            # Require many consistent detections
    'iou_threshold': 0.2      # Very lenient matching
}
