#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration file for Cement Bag Counter

This file contains optimized parameters for detecting, tracking, and counting
cement bags on a conveyor belt based on the specific video characteristics.
"""

# Optimized detection parameters for cement bags
# These parameters are tuned for white/light-colored bags on a darker conveyor belt
DETECTION_PARAMS = {
    'history': 300,           # Shorter history for faster adaptation to changes
    'var_threshold': 10,      # Lower threshold to better detect light objects on dark background
    'min_area': 5000,         # Increased to match the actual size of cement bags
    'max_area': 30000,        # Adjusted based on bag size
    'aspect_ratio_range': (0.6, 1.2),  # Tuned to match the shape of bags
    'learning_rate': 0.01     # Standard learning rate
}

# Optimized tracking parameters
# These parameters are tuned for tracking cement bags on a conveyor belt
TRACKING_PARAMS = {
    'max_age': 15,            # Allow tracks to live longer to handle occlusions
    'min_hits': 2,            # Require fewer hits to establish a track
    'iou_threshold': 0.3      # Standard IOU threshold
}

# Optimized counting parameters (line-based)
# These parameters are tuned for counting cement bags crossing a line
LINE_COUNTING_PARAMS = {
    'line_position': ((0, 240), (640, 240)),  # Horizontal line in the middle (adjust to your frame size)
    'direction': 'top-to-bottom',  # Count bags moving downward
    'min_distance': 5,        # Smaller distance to detect crossings more reliably
    'cooldown_frames': 15     # Longer cooldown to prevent double-counting
}

# Optimized counting parameters (region-based)
# These parameters are tuned for counting cement bags entering a region
REGION_COUNTING_PARAMS = {
    'region': (160, 120, 320, 240),  # Region in the center (adjust to your frame size)
    'entry_direction': 'top',  # Count bags entering from the top
    'min_frames_in_region': 2,  # Require fewer frames in region to count
    'cooldown_frames': 15     # Longer cooldown to prevent double-counting
}

# Alternative parameters for different lighting conditions
# Use these parameters if the lighting is brighter or there's less contrast
BRIGHT_LIGHTING_PARAMS = {
    'history': 500,
    'var_threshold': 8,
    'min_area': 5000,
    'max_area': 30000,
    'aspect_ratio_range': (0.6, 1.2),
    'learning_rate': 0.005
}

# Alternative parameters for different bag sizes
# Use these parameters if the bags are smaller or larger than standard
SMALL_BAG_PARAMS = {
    'min_area': 3000,
    'max_area': 15000,
    'aspect_ratio_range': (0.5, 1.0)
}

LARGE_BAG_PARAMS = {
    'min_area': 10000,
    'max_area': 50000,
    'aspect_ratio_range': (0.8, 1.5)
}

# Alternative parameters for faster conveyor belt speed
# Use these parameters if the conveyor belt moves faster than normal
FAST_CONVEYOR_PARAMS = {
    'history': 200,
    'learning_rate': 0.02,
    'max_age': 8,
    'min_hits': 1,
    'cooldown_frames': 8
}
