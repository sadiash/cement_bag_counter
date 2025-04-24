#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced implementation for Cement Bag Counter

This module provides an improved implementation that combines multiple detection
approaches to handle challenging conditions like motion blur and poor contrast.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

class BagDetector:
    """Enhanced cement bag detector using multiple detection approaches."""
    
    def __init__(self, config=None):
        """
        Initialize the enhanced bag detector with configurable parameters.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        
        # Extract parameters from config
        self.history = self.config.get('history', 120)
        self.var_threshold = self.config.get('var_threshold', 6)
        self.detect_shadows = self.config.get('detect_shadows', False)
        self.min_area = self.config.get('min_area', 3000)
        self.max_area = self.config.get('max_area', 25000)
        self.aspect_ratio_range = self.config.get('aspect_ratio_range', (0.4, 1.8))
        self.learning_rate = self.config.get('learning_rate', 0.03)
        
        # ROI parameters
        self.use_roi = self.config.get('use_roi', True)
        self.roi = self.config.get('roi', (100, 100, 440, 280))
        
        # Preprocessing parameters
        self.apply_blur = self.config.get('apply_blur', True)
        self.blur_kernel_size = self.config.get('blur_kernel_size', 3)
        self.apply_contrast_enhancement = self.config.get('apply_contrast_enhancement', True)
        self.contrast_clip_limit = self.config.get('contrast_clip_limit', 2.0)
        self.apply_morphology = self.config.get('apply_morphology', True)
        self.morph_kernel_size = self.config.get('morph_kernel_size', 5)
        
        # Contour detection parameters
        self.threshold_min = self.config.get('threshold_min', 120)
        self.threshold_max = self.config.get('threshold_max', 255)
        self.solidity_min = self.config.get('solidity_min', 0.7)
        self.use_adaptive_threshold = self.config.get('use_adaptive_threshold', True)
        self.adaptive_block_size = self.config.get('adaptive_block_size', 19)
        self.adaptive_c = self.config.get('adaptive_c', 2)
        
        # Frame differencing parameters
        self.frame_diff_enabled = self.config.get('frame_diff_enabled', True)
        self.history_length = self.config.get('history_length', 3)
        self.diff_threshold = self.config.get('diff_threshold', 20)
        self.use_absolute_diff = self.config.get('use_absolute_diff', True)
        
        # Hybrid detection parameters
        self.use_background_subtraction = self.config.get('use_background_subtraction', True)
        self.use_frame_differencing = self.config.get('use_frame_differencing', True)
        self.use_contour_detection = self.config.get('use_contour_detection', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.combine_method = self.config.get('combine_method', 'union')
        
        # Debug parameters
        self.debug = self.config.get('debug', {})
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        
        # Initialize frame history for frame differencing
        self.frame_history = deque(maxlen=self.history_length)
        
        # For adaptive parameters
        self.frame_count = 0
        self.detected_bags_history = []
        
        # Debug images
        self.debug_images = {}
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame to enhance detection.
        
        Args:
            frame: Input image frame
            
        Returns:
            Preprocessed frame
        """
        # Apply ROI if enabled
        if self.use_roi:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w].copy()
        else:
            roi_frame = frame.copy()
            
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        if self.apply_blur:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
            
        # Apply contrast enhancement
        if self.apply_contrast_enhancement:
            clahe = cv2.createCLAHE(clipLimit=self.contrast_clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
        if self.debug.get('show_preprocessing', False):
            self.debug_images['preprocessed'] = gray.copy()
            
        return gray
        
    def detect_with_background_subtraction(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect bags using background subtraction.
        
        Args:
            frame: Preprocessed grayscale frame
            
        Returns:
            List of detections
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Remove shadows (they are marked as gray (127))
        binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to remove noise and fill holes
        if self.apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug.get('show_background_mask', False):
            self.debug_images['bg_mask'] = binary_mask.copy()
            
        # Find contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to get detections
        detections = self._process_contours(contours, frame.shape, 'bg_subtraction')
        
        return detections
    
    def detect_with_frame_differencing(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect bags using frame differencing.
        
        Args:
            frame: Preprocessed grayscale frame
            
        Returns:
            List of detections
        """
        # Add current frame to history
        self.frame_history.append(frame)
        
        # Need at least 2 frames for differencing
        if len(self.frame_history) < 2:
            return []
            
        # Compute difference between current and previous frames
        diff_mask = np.zeros_like(frame)
        
        if self.use_absolute_diff:
            # Use absolute difference
            for i in range(len(self.frame_history) - 1):
                prev_frame = self.frame_history[i]
                curr_frame = self.frame_history[i + 1]
                frame_diff = cv2.absdiff(prev_frame, curr_frame)
                _, thresh_diff = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
                diff_mask = cv2.bitwise_or(diff_mask, thresh_diff)
        else:
            # Use accumulated difference
            prev_frame = self.frame_history[0]
            curr_frame = self.frame_history[-1]
            frame_diff = cv2.absdiff(prev_frame, curr_frame)
            _, diff_mask = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise and fill holes
        if self.apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug.get('show_frame_diff', False):
            self.debug_images['frame_diff'] = diff_mask.copy()
            
        # Find contours in the mask
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to get detections
        detections = self._process_contours(contours, frame.shape, 'frame_diff')
        
        return detections
    
    def detect_with_contours(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect bags using contour analysis on thresholded image.
        
        Args:
            frame: Preprocessed grayscale frame
            
        Returns:
            List of detections
        """
        # Apply thresholding
        if self.use_adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                frame, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                self.adaptive_block_size, 
                self.adaptive_c
            )
        else:
            _, binary = cv2.threshold(frame, self.threshold_min, self.threshold_max, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to remove noise and fill holes
        if self.apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        if self.debug.get('show_contours', False):
            self.debug_images['contours'] = binary.copy()
            
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to get detections
        detections = self._process_contours(contours, frame.shape, 'contour')
        
        return detections
    
    def _process_contours(self, contours, frame_shape, detection_type) -> List[Dict[str, Any]]:
        """
        Process contours to extract valid detections.
        
        Args:
            contours: List of contours
            frame_shape: Shape of the frame
            detection_type: Type of detection method
            
        Returns:
            List of detections
        """
        detections = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area <= area <= self.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Filter by aspect ratio
                if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                    # Calculate solidity (area / convex hull area)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    # Filter by solidity
                    if solidity >= self.solidity_min:
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = x + w // 2, y + h // 2
                        
                        # Calculate a confidence score based on area and shape
                        # This is a simple heuristic and can be improved
                        confidence = min(1.0, (area / self.max_area) * solidity)
                        
                        # Add detection
                        detections.append({
                            'bbox': (x, y, w, h),
                            'centroid': (cx, cy),
                            'area': area,
                            'confidence': confidence,
                            'detection_type': detection_type
                        })
        
        return detections
    
    def combine_detections(self, detections_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Combine detections from multiple methods.
        
        Args:
            detections_list: List of detection lists from different methods
            
        Returns:
            Combined list of detections
        """
        if not detections_list:
            return []
            
        if len(detections_list) == 1:
            return detections_list[0]
            
        # Flatten all detections
        all_detections = [d for sublist in detections_list for d in sublist]
        
        if self.combine_method == 'union':
            # Simple union of all detections
            return all_detections
        elif self.combine_method == 'intersection':
            # Group overlapping detections
            final_detections = []
            used_indices = set()
            
            for i, det1 in enumerate(all_detections):
                if i in used_indices:
                    continue
                    
                x1, y1, w1, h1 = det1['bbox']
                overlapping_dets = [det1]
                
                for j, det2 in enumerate(all_detections):
                    if i == j or j in used_indices:
                        continue
                        
                    x2, y2, w2, h2 = det2['bbox']
                    
                    # Check if bounding boxes overlap
                    if (x1 < x2 + w2 and x1 + w1 > x2 and
                        y1 < y2 + h2 and y1 + h1 > y2):
                        overlapping_dets.append(det2)
                        used_indices.add(j)
                
                # Only keep detection if it was found by multiple methods
                if len(set(d['detection_type'] for d in overlapping_dets)) > 1:
                    # Use the detection with highest confidence
                    best_det = max(overlapping_dets, key=lambda d: d['confidence'])
                    final_detections.append(best_det)
                    
            return final_detections
        else:  # weighted
            # Group overlapping detections and combine their confidence
            final_detections = []
            used_indices = set()
            
            for i, det1 in enumerate(all_detections):
                if i in used_indices:
                    continue
                    
                x1, y1, w1, h1 = det1['bbox']
                overlapping_dets = [det1]
                
                for j, det2 in enumerate(all_detections):
                    if i == j or j in used_indices:
                        continue
                        
                    x2, y2, w2, h2 = det2['bbox']
                    
                    # Check if bounding boxes overlap
                    if (x1 < x2 + w2 and x1 + w1 > x2 and
                        y1 < y2 + h2 and y1 + h1 > y2):
                        overlapping_dets.append(det2)
                        used_indices.add(j)
                
                # Combine overlapping detections
                if overlapping_dets:
                    # Average the bounding boxes
                    avg_x = sum(d['bbox'][0] for d in overlapping_dets) // len(overlapping_dets)
                    avg_y = sum(d['bbox'][1] for d in overlapping_dets) // len(overlapping_dets)
                    avg_w = sum(d['bbox'][2] for d in overlapping_dets) // len(overlapping_dets)
                    avg_h = sum(d['bbox'][3] for d in overlapping_dets) // len(overlapping_dets)
                    
                    # Average the centroids
                    avg_cx = sum(d['centroid'][0] for d in overlapping_dets) // len(overlapping_dets)
                    avg_cy = sum(d['centroid'][1] for d in overlapping_dets) // len(overlapping_dets)
                    
                    # Combine confidences (higher weight for multiple detections)
                    combined_conf = sum(d['confidence'] for d in overlapping_dets)
                    combined_conf *= (1.0 + 0.2 * (len(overlapping_dets) - 1))  # Boost for multiple detections
                    combined_conf = min(1.0, combined_conf)  # Cap at 1.0
                    
                    # Create combined detection
                    combined_det = {
                        'bbox': (avg_x, avg_y, avg_w, avg_h),
                        'centroid': (avg_cx, avg_cy),
                        'area': sum(d['area'] for d in overlapping_dets) // len(overlapping_dets),
                        'confidence': combined_conf,
                        'detection_type': 'combined'
                    }
                    
                    final_detections.append(combined_det)
            
            return final_detections
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cement bags in the given frame using multiple methods.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing detection information
        """
        self.frame_count += 1
        self.debug_images = {}  # Reset debug images
        
        # Create a copy of the frame for visualization
        original = frame.copy()
        
        # Preprocess the frame
        preprocessed = self.preprocess_frame(frame)
        
        # Apply different detection methods
        detections_list = []
        
        if self.use_background_subtraction:
            bg_detections = self.detect_with_background_subtraction(preprocessed)
            detections_list.append(bg_detections)
            
        if self.use_frame_differencing and self.frame_diff_enabled:
            diff_detections = self.detect_with_frame_differencing(preprocessed)
            detections_list.append(diff_detections)
            
        if self.use_contour_detection:
            contour_detections = self.detect_with_contours(preprocessed)
            detections_list.append(contour_detections)
        
        # Combine detections from different methods
        combined_detections = self.combine_detections(detections_list)
        
        # Filter by confidence threshold
        final_detections = [d for d in combined_detections if d['confidence'] >= self.confidence_threshold]
        
        # If using ROI, adjust coordinates to original frame
        if self.use_roi:
            x_offset, y_offset, _, _ = self.roi
            for det in final_detections:
                x, y, w, h = det['bbox']
                cx, cy = det['centroid']
                det['bbox'] = (x + x_offset, y + y_offset, w, h)
                det['centroid'] = (cx + x_offset, cy + y_offset)
        
        # Update detection history for adaptive parameters
        self.detected_bags_history.append(len(final_detections))
        if len(self.detected_bags_history) > 30:  # Keep last 30 frames
            self.detected_bags_history.pop(0)
        
        # Adapt parameters if needed (every 100 frames)
        if self.frame_count % 100 == 0:
            self._adapt_parameters()
            
        return final_detections
    
    def _adapt_parameters(self):
        """Adapt detection parameters based on history."""
        if len(self.detected_bags_history) < 10:
            return  # Not enough history
            
        avg_detections = sum(self.detected_bags_history) / len(self.detected_bags_history)
        
        # If consistently finding too many or too few detections, adjust parameters
        if avg_detections > 5:  # Too many detections
            self.min_area = int(self.min_area * 1.1)  # Increase minimum area
            self.var_threshold = min(30, self.var_threshold + 1)  # Increase threshold
        elif avg_detections < 0.5:  # Too few detections
            self.min_area = max(500, int(self.min_area * 0.9))  # Decrease minimum area
            self.var_threshold = max(3, self.var_threshold - 1)  # Decrease threshold
    
    def visualize(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection results on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw ROI if enabled and debug is on
        if self.use_roi and self.debug.get('show_roi', False):
            x, y, w, h = self.roi
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw detections
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det['centroid']
            confidence = det['confidence']
            detection_type = det.get('detection_type', 'unknown')
            
            # Color based on detection type
            if detection_type == 'bg_subtraction':
                color = (0, 255, 0)  # Green
            elif detection_type == 'frame_diff':
                color = (0, 0, 255)  # Red
            elif detection_type == 'contour':
                color = (255, 0, 0)  # Blue
            elif detection_type == 'combined':
                color = (255, 0, 255)  # Magenta
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cv2.circle(vis_frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Draw confidence and type
            label = f"{confidence:.2f} {detection_type}"
            cv2.putText(vis_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add debug images if available
        if self.debug_images:
            debug_height = 150  # Height of debug images
            debug_width = frame.shape[1] // len(self.debug_images)
            
            for i, (name, img) in enumerate(self.debug_images.items()):
                # Resize debug image
                debug_img = cv2.resize(img, (debug_width, debug_height))
                
                # Convert to color if grayscale
                if len(debug_img.shape) == 2:
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                
                # Add label
                cv2.putText(debug_img, name, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Place at bottom of frame
                x_offset = i * debug_width
                y_offset = frame.shape[0] - debug_height
                
                vis_frame[y_offset:y_offset+debug_height, x_offset:x_offset+debug_width] = debug_img
        
        return vis_frame


# Simple test function to verify the detector works
def test_detector(video_path: str, config=None):
    """Test the detector on a video file."""
    cap = cv2.VideoCapture(video_path)
    detector = EnhancedBagDetector(config)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        vis_frame = detector.visualize(frame, detections)
        
        cv2.imshow('Enhanced Detections', vis_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    from config import ENHANCED_DETECTION_PARAMS, PREPROCESSING_PARAMS, CONTOUR_DETECTION_PARAMS, FRAME_DIFF_PARAMS, HYBRID_DETECTION_PARAMS, ROI_PARAMS, DEBUG_PARAMS
    
    # Combine all parameters
    config = {**ENHANCED_DETECTION_PARAMS, **PREPROCESSING_PARAMS, **CONTOUR_DETECTION_PARAMS, 
              **FRAME_DIFF_PARAMS, **HYBRID_DETECTION_PARAMS, **ROI_PARAMS, **DEBUG_PARAMS}
    
    if len(sys.argv) > 1:
        test_detector(sys.argv[1], config)
    else:
        print("Usage: python enhanced_detection.py <video_path>")
