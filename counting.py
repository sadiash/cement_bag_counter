#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Counting Module for Cement Bag Counter

This module implements counting of cement bags as they cross a designated line
or region on a conveyor belt.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config import LINE_COUNTING_PARAMS, REGION_COUNTING_PARAMS


class LineCounter:
    """Counts objects crossing a virtual line."""
    
    def __init__(self, 
                 line_position: Tuple[Tuple[int, int], Tuple[int, int]] = LINE_COUNTING_PARAMS['line_position'],
                 direction: str = LINE_COUNTING_PARAMS['direction'],
                 min_distance: int = LINE_COUNTING_PARAMS['min_distance'],
                 cooldown_frames: int = LINE_COUNTING_PARAMS['cooldown_frames']):
        """
        Initialize the line counter with configurable parameters.
        
        Args:
            line_position: Line coordinates as ((x1, y1), (x2, y2))
            direction: Expected movement direction ('left-to-right', 'right-to-left', 
                      'top-to-bottom', 'bottom-to-top', or 'any')
            min_distance: Minimum distance from line to count a crossing
            cooldown_frames: Frames to wait before counting the same object again
        """
        self.line_position = line_position
        self.direction = direction
        self.min_distance = min_distance
        self.cooldown_frames = cooldown_frames
        
        # Calculate line parameters (for point-to-line distance)
        (x1, y1), (x2, y2) = line_position
        self.line_params = {
            'A': y2 - y1,
            'B': x1 - x2,
            'C': x2 * y1 - x1 * y2
        }
        self.line_length = np.sqrt(self.line_params['A']**2 + self.line_params['B']**2)
        
        # Determine line orientation
        if abs(x2 - x1) > abs(y2 - y1):
            self.line_orientation = 'horizontal'
        else:
            self.line_orientation = 'vertical'
            
        # Track objects that have crossed the line
        self.crossed_objects = {}  # track_id -> frame_last_crossed
        self.count = 0
        self.frame_count = 0
        
        # Store object positions from previous frame
        self.prev_positions = {}  # track_id -> (x, y)
        
    def update(self, tracks: List[Dict[str, Any]]) -> int:
        """
        Update counter with new tracks.
        
        Args:
            tracks: List of track dictionaries with 'track_id' and 'centroid' keys
            
        Returns:
            Current count of objects that have crossed the line
        """
        self.frame_count += 1
        
        # Process each track
        for track in tracks:
            track_id = track['track_id']
            centroid = track['centroid']
            
            # Skip inactive tracks
            if not track.get('active', True):
                continue
                
            # Check if this object has already been counted recently
            if track_id in self.crossed_objects:
                if self.frame_count - self.crossed_objects[track_id] < self.cooldown_frames:
                    # Object is in cooldown period, skip
                    continue
            
            # Calculate distance to line
            distance = self._point_to_line_distance(centroid)
            
            # Check if object is close enough to the line
            if distance <= self.min_distance:
                # Check if we have a previous position for this track
                if track_id in self.prev_positions:
                    prev_centroid = self.prev_positions[track_id]
                    
                    # Check if the object has crossed the line in the expected direction
                    if self._has_crossed_line(prev_centroid, centroid):
                        # Count the object
                        self.count += 1
                        self.crossed_objects[track_id] = self.frame_count
            
            # Update previous position
            self.prev_positions[track_id] = centroid
            
        # Clean up old entries
        self._cleanup_old_entries()
            
        return self.count
    
    def _point_to_line_distance(self, point: Tuple[int, int]) -> float:
        """
        Calculate the distance from a point to the counting line.
        
        Args:
            point: Point coordinates (x, y)
            
        Returns:
            Distance from point to line
        """
        x, y = point
        A, B, C = self.line_params['A'], self.line_params['B'], self.line_params['C']
        
        # Distance formula: |Ax + By + C| / sqrt(A^2 + B^2)
        return abs(A * x + B * y + C) / self.line_length
    
    def _has_crossed_line(self, prev_point: Tuple[int, int], curr_point: Tuple[int, int]) -> bool:
        """
        Check if the object has crossed the line in the expected direction.
        
        Args:
            prev_point: Previous position (x, y)
            curr_point: Current position (x, y)
            
        Returns:
            True if the object has crossed the line in the expected direction
        """
        (x1, y1), (x2, y2) = self.line_position
        prev_x, prev_y = prev_point
        curr_x, curr_y = curr_point
        
        # For horizontal line
        if self.line_orientation == 'horizontal':
            # Determine if points are on different sides of the line
            if (prev_y < y1 and curr_y >= y1) or (prev_y >= y1 and curr_y < y1):
                # Check direction
                if self.direction == 'top-to-bottom' and prev_y < curr_y:
                    return True
                elif self.direction == 'bottom-to-top' and prev_y > curr_y:
                    return True
                elif self.direction == 'any':
                    return True
        
        # For vertical line
        else:
            # Determine if points are on different sides of the line
            if (prev_x < x1 and curr_x >= x1) or (prev_x >= x1 and curr_x < x1):
                # Check direction
                if self.direction == 'left-to-right' and prev_x < curr_x:
                    return True
                elif self.direction == 'right-to-left' and prev_x > curr_x:
                    return True
                elif self.direction == 'any':
                    return True
                
        return False
    
    def _cleanup_old_entries(self):
        """Remove old entries from tracking dictionaries."""
        # Remove old crossed objects
        to_remove = []
        for track_id, frame in self.crossed_objects.items():
            if self.frame_count - frame > 100:  # Remove after 100 frames
                to_remove.append(track_id)
                
        for track_id in to_remove:
            self.crossed_objects.pop(track_id, None)
            
        # Remove old previous positions (simplified approach)
        # Just keep positions for objects we've counted or are still being tracked
        to_remove = []
        for track_id in self.prev_positions:
            if track_id not in self.crossed_objects:
                to_remove.append(track_id)
                
        for track_id in to_remove:
            self.prev_positions.pop(track_id, None)
    
    def reset(self):
        """Reset the counter."""
        self.crossed_objects = {}
        self.count = 0
        self.prev_positions = {}
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw counter visualization on the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw counting line
        (x1, y1), (x2, y2) = self.line_position
        cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw direction arrow
        if self.line_orientation == 'horizontal':
            if self.direction == 'top-to-bottom':
                arrow_x, arrow_y = (x1 + x2) // 2, y1 - 20
                cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y + 40), (0, 255, 255), 2)
            elif self.direction == 'bottom-to-top':
                arrow_x, arrow_y = (x1 + x2) // 2, y1 + 20
                cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), (0, 255, 255), 2)
        else:
            if self.direction == 'left-to-right':
                arrow_x, arrow_y = x1 - 20, (y1 + y2) // 2
                cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x + 40, arrow_y), (0, 255, 255), 2)
            elif self.direction == 'right-to-left':
                arrow_x, arrow_y = x1 + 20, (y1 + y2) // 2
                cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x - 40, arrow_y), (0, 255, 255), 2)
        
        # Draw count
        cv2.putText(vis_frame, f"Count: {self.count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return vis_frame


class RegionCounter:
    """Counts objects entering and exiting a region of interest."""
    
    def __init__(self, 
                 region: Tuple[int, int, int, int] = REGION_COUNTING_PARAMS['region'],
                 entry_direction: str = REGION_COUNTING_PARAMS['entry_direction'],
                 min_frames_in_region: int = REGION_COUNTING_PARAMS['min_frames_in_region'],
                 cooldown_frames: int = REGION_COUNTING_PARAMS['cooldown_frames']):
        """
        Initialize the region counter with configurable parameters.
        
        Args:
            region: Region coordinates as (x, y, w, h)
            entry_direction: Expected entry direction ('left', 'right', 'top', 'bottom', or 'any')
            min_frames_in_region: Minimum frames an object must be in the region to be counted
            cooldown_frames: Frames to wait before counting the same object again
        """
        self.region = region
        self.entry_direction = entry_direction
        self.min_frames_in_region = min_frames_in_region
        self.cooldown_frames = cooldown_frames
        
        # Track objects in the region
        self.objects_in_region = {}  # track_id -> frames_in_region
        self.counted_objects = {}  # track_id -> frame_last_counted
        self.count = 0
        self.frame_count = 0
        
        # Store object positions from previous frame
        self.prev_positions = {}  # track_id -> (x, y)
        
    def update(self, tracks: List[Dict[str, Any]]) -> int:
        """
        Update counter with new tracks.
        
        Args:
            tracks: List of track dictionaries with 'track_id' and 'centroid' keys
            
        Returns:
            Current count of objects that have entered the region
        """
        self.frame_count += 1
        active_track_ids = set()
        
        # Process each track
        for track in tracks:
            track_id = track['track_id']
            centroid = track['centroid']
            active_track_ids.add(track_id)
            
            # Skip inactive tracks
            if not track.get('active', True):
                continue
                
            # Check if this object has already been counted recently
            if track_id in self.counted_objects:
                if self.frame_count - self.counted_objects[track_id] < self.cooldown_frames:
                    # Object is in cooldown period, skip
                    continue
            
            # Check if object is in the region
            if self._is_point_in_region(centroid):
                # Object is in the region
                if track_id not in self.objects_in_region:
                    # New object in region, check entry direction
                    if track_id in self.prev_positions:
                        prev_centroid = self.prev_positions[track_id]
                        if self._check_entry_direction(prev_centroid, centroid):
                            # Object entered from the expected direction
                            self.objects_in_region[track_id] = 1
                    else:
                        # No previous position, assume valid entry
                        self.objects_in_region[track_id] = 1
                else:
                    # Increment frames in region
                    self.objects_in_region[track_id] += 1
                    
                    # Check if object has been in region long enough to count
                    if self.objects_in_region[track_id] >= self.min_frames_in_region:
                        if track_id not in self.counted_objects:
                            # Count the object
                            self.count += 1
                            self.counted_objects[track_id] = self.frame_count
            else:
                # Object is not in the region, remove from tracking
                if track_id in self.objects_in_region:
                    self.objects_in_region.pop(track_id)
            
            # Update previous position
            self.prev_positions[track_id] = centroid
            
        # Remove objects that are no longer being tracked
        to_remove = []
        for track_id in self.objects_in_region:
            if track_id not in active_track_ids:
                to_remove.append(track_id)
                
        for track_id in to_remove:
            self.objects_in_region.pop(track_id)
            
        # Clean up old entries
        self._cleanup_old_entries()
            
        return self.count
    
    def _is_point_in_region(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside the region.
        
        Args:
            point: Point coordinates (x, y)
            
        Returns:
            True if the point is inside the region
        """
        x, y = point
        rx, ry, rw, rh = self.region
        
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def _check_entry_direction(self, prev_point: Tuple[int, int], curr_point: Tuple[int, int]) -> bool:
        """
        Check if the object entered the region from the expected direction.
        
        Args:
            prev_point: Previous position (x, y)
            curr_point: Current position (x, y)
            
        Returns:
            True if the object entered from the expected direction
        """
        if self.entry_direction == 'any':
            return True
            
        prev_x, prev_y = prev_point
        curr_x, curr_y = curr_point
        rx, ry, rw, rh = self.region
        
        if self.entry_direction == 'left' and prev_x < rx:
            return True
        elif self.entry_direction == 'right' and prev_x > rx + rw:
            return True
        elif self.entry_direction == 'top' and prev_y < ry:
            return True
        elif self.entry_direction == 'bottom' and prev_y > ry + rh:
            return True
            
        return False
    
    def _cleanup_old_entries(self):
        """Remove old entries from tracking dictionaries."""
        # Remove old counted objects
        to_remove = []
        for track_id, frame in self.counted_objects.items():
            if self.frame_count - frame > 100:  # Remove after 100 frames
                to_remove.append(track_id)
                
        for track_id in to_remove:
            self.counted_objects.pop(track_id, None)
    
    def reset(self):
        """Reset the counter."""
        self.objects_in_region = {}
        self.counted_objects = {}
        self.count = 0
        self.prev_positions = {}
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw counter visualization on the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw region
        x, y, w, h = self.region
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        # Draw entry direction
        if self.entry_direction == 'left':
            arrow_x, arrow_y = x - 20, y + h // 2
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x + 40, arrow_y), (255, 0, 255), 2)
        elif self.entry_direction == 'right':
            arrow_x, arrow_y = x + w + 20, y + h // 2
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x - 40, arrow_y), (255, 0, 255), 2)
        elif self.entry_direction == 'top':
            arrow_x, arrow_y = x + w // 2, y - 20
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y + 40), (255, 0, 255), 2)
        elif self.entry_direction == 'bottom':
            arrow_x, arrow_y = x + w // 2, y + h + 20
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), (255, 0, 255), 2)
        
        # Draw count
        cv2.putText(vis_frame, f"Count: {self.count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        return vis_frame


# Simple test function to verify the counter works
def test_counter(video_path: str, counter_type: str = 'line'):
    """Test the counter on a video file."""
    import cv2
    from object_detection import BagDetector
    from object_tracking import BagTracker
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector = BagDetector()
    tracker = BagTracker()
    
    # Create counter based on type
    if counter_type == 'line':
        # Create a horizontal line in the middle of the frame
        counter = LineCounter(
            line_position=((0, height // 2), (width, height // 2)),
            direction='top-to-bottom'
        )
    else:
        # Create a region in the center of the frame
        region_width = width // 3
        region_height = height // 3
        region_x = (width - region_width) // 2
        region_y = (height - region_height) // 2
        counter = RegionCounter(
            region=(region_x, region_y, region_width, region_height),
            entry_direction='any'
        )
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect bags
        detections = detector.detect(frame)
        
        # Track bags
        tracks = tracker.update(detections)
        
        # Count bags
        count = counter.update(tracks)
        
        # Visualize
        vis_frame = detector.visualize(frame, detections)
        vis_frame = tracker.visualize(vis_frame, tracks)
        vis_frame = counter.visualize(vis_frame)
        
        # Display
        cv2.imshow('Counting', vis_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Final count: {counter.count}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        counter_type = 'line'
        if len(sys.argv) > 2:
            counter_type = sys.argv[2]
        test_counter(sys.argv[1], counter_type)
    else:
        print("Usage: python counting.py <video_path> [line|region]")
