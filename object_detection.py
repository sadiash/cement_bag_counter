#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object Detection Module for Cement Bag Counter

This module implements detection of cement bags on a conveyor belt using
background subtraction and contour analysis techniques.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class BagDetector:
    """Cement bag detector using background subtraction and contour analysis."""
    
    def __init__(self, 
                 history: int = 500,
                 var_threshold: float = 16,
                 detect_shadows: bool = True,
                 min_area: int = 1000,
                 max_area: int = 50000,
                 aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
                 learning_rate: float = 0.01):
        """
        Initialize the bag detector with configurable parameters.
        
        Args:
            history: Length of history for background subtractor
            var_threshold: Threshold for background/foreground decision
            detect_shadows: Whether to detect and mark shadows
            min_area: Minimum contour area to be considered a bag
            max_area: Maximum contour area to be considered a bag
            aspect_ratio_range: Valid range for width/height ratio
            learning_rate: Background model learning rate
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.learning_rate = learning_rate
        
        # For adaptive parameters
        self.frame_count = 0
        self.detected_bags_history = []
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cement bags in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing detection information:
            - 'bbox': (x, y, w, h) bounding box
            - 'centroid': (cx, cy) center point
            - 'area': contour area
            - 'confidence': detection confidence score
        """
        self.frame_count += 1
        
        # Create a copy of the frame for visualization
        original = frame.copy()
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Remove shadows (they are marked as gray (127))
        binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to remove noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
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
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w // 2, y + h // 2
                    
                    # Calculate a confidence score based on area and shape
                    # This is a simple heuristic and can be improved
                    confidence = min(1.0, area / self.max_area)
                    
                    # Add detection
                    detections.append({
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'area': area,
                        'confidence': confidence
                    })
        
        # Update detection history for adaptive parameters
        self.detected_bags_history.append(len(detections))
        if len(self.detected_bags_history) > 30:  # Keep last 30 frames
            self.detected_bags_history.pop(0)
        
        # Adapt parameters if needed (every 100 frames)
        if self.frame_count % 100 == 0:
            self._adapt_parameters()
            
        return detections
    
    def _adapt_parameters(self):
        """Adapt detection parameters based on history."""
        if len(self.detected_bags_history) < 10:
            return  # Not enough history
            
        avg_detections = sum(self.detected_bags_history) / len(self.detected_bags_history)
        
        # If consistently finding too many or too few detections, adjust area thresholds
        if avg_detections > 5:  # Too many detections
            self.min_area = int(self.min_area * 1.1)  # Increase minimum area
        elif avg_detections < 0.5:  # Too few detections
            self.min_area = max(500, int(self.min_area * 0.9))  # Decrease minimum area
    
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
        
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det['centroid']
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cv2.circle(vis_frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Draw confidence
            cv2.putText(vis_frame, f"{confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame


# Simple test function to verify the detector works
def test_detector(video_path: str):
    """Test the detector on a video file."""
    cap = cv2.VideoCapture(video_path)
    detector = BagDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        vis_frame = detector.visualize(frame, detections)
        
        cv2.imshow('Detections', vis_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_detector(sys.argv[1])
    else:
        print("Usage: python object_detection.py <video_path>")
