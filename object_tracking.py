#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object Tracking Module for Cement Bag Counter

This module implements tracking of cement bags on a conveyor belt using
the SORT (Simple Online and Realtime Tracking) algorithm.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Bounding box in the format [x, y, w, h]
        """
        # Convert [x, y, w, h] to [x, y, x+w, y+h] format
        x, y, w, h = bbox
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # w
            [0, 0, 0, 1, 0, 0, 0],  # h
            [0, 0, 0, 0, 1, 0, 0],  # dx
            [0, 0, 0, 0, 0, 1, 0],  # dy
            [0, 0, 0, 0, 0, 0, 1]   # dw
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement uncertainty
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state with bounding box
        self.kf.x[:4] = np.array([x, y, w, h]).reshape((4, 1))
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centroid = (int(x + w/2), int(y + h/2))
        self.last_detection = {'bbox': (x, y, w, h), 'centroid': self.centroid}
        
    def update(self, bbox):
        """
        Update the state vector with observed bbox.
        
        Args:
            bbox: Bounding box in the format [x, y, w, h]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        x, y, w, h = bbox
        self.centroid = (int(x + w/2), int(y + h/2))
        self.last_detection = {'bbox': (x, y, w, h), 'centroid': self.centroid}
        
        self.kf.update(np.array([x, y, w, h]).reshape((4, 1)))
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        
        Returns:
            Predicted bounding box in the format [x, y, w, h]
        """
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[2] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(self.get_state())
        
        # Update centroid based on prediction
        x, y, w, h = self.get_state()
        self.centroid = (int(x + w/2), int(y + h/2))
        
        return self.history[-1]
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        
        Returns:
            Current bounding box in the format [x, y, w, h]
        """
        return self.kf.x[:4].flatten().tolist()


class BagTracker:
    """Cement bag tracker using SORT algorithm."""
    
    def __init__(self, 
                 max_age: int = 10,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize the bag tracker with configurable parameters.
        
        Args:
            max_age: Maximum number of frames to keep a track alive without matching detections
            min_hits: Minimum number of hits needed to establish a track
            iou_threshold: Intersection over Union threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update trackers with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
            
        Returns:
            List of track dictionaries with tracking information
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Convert detections to format [x, y, w, h]
        if len(detections) > 0:
            dets = np.array([d['bbox'] for d in detections])
        else:
            dets = np.empty((0, 4))
            
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
            
        # Return active tracks
        ret = []
        for i, trk in reversed(list(enumerate(self.trackers))):
            d = trk.get_state()
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Active track with sufficient hits
                ret.append({
                    'track_id': trk.id,
                    'bbox': tuple(map(int, d)),
                    'centroid': trk.centroid,
                    'age': trk.age,
                    'hits': trk.hits,
                    'time_since_update': trk.time_since_update,
                    'active': True
                })
            elif trk.time_since_update < self.max_age:
                # Track is still alive but not confirmed with detection
                ret.append({
                    'track_id': trk.id,
                    'bbox': tuple(map(int, d)),
                    'centroid': trk.centroid,
                    'age': trk.age,
                    'hits': trk.hits,
                    'time_since_update': trk.time_since_update,
                    'active': False
                })
            else:
                # Remove dead track
                self.trackers.pop(i)
                
        return ret
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=None):
        """
        Associate detections with trackers using IoU.
        
        Args:
            detections: Numpy array of detections in format [x, y, w, h]
            trackers: Numpy array of trackers in format [x, y, w, h]
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of matches, unmatched_detections, unmatched_trackers
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)
            
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0), dtype=int), np.arange(len(trackers))
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
                
        # Use Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                # When there is a single detection-tracker pair with IoU > threshold
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Use Hungarian algorithm for optimal assignment
                matched_indices = np.array(linear_sum_assignment(-iou_matrix)).T
        else:
            matched_indices = np.empty(shape=(0, 2))
            
        # Filter matches by IoU threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= iou_threshold:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        # Find unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matches[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matches[:, 1]:
                unmatched_trackers.append(t)
                
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def _iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            bbox1: First bounding box in format [x, y, w, h]
            bbox2: Second bounding box in format [x, y, w, h]
            
        Returns:
            IoU score
        """
        # Convert to [x1, y1, x2, y2] format
        bbox1 = np.array([bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]])
        bbox2 = np.array([bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]])
        
        # Determine intersection rectangle coordinates
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def visualize(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw tracking results on the frame.
        
        Args:
            frame: Input image frame
            tracks: List of track dictionaries
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for track in tracks:
            x, y, w, h = track['bbox']
            track_id = track['track_id']
            active = track['active']
            
            # Color based on active status
            color = (0, 255, 0) if active else (0, 165, 255)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID
            cv2.putText(vis_frame, f"ID: {track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw centroid
            cv2.circle(vis_frame, track['centroid'], 4, (0, 0, 255), -1)
            
        return vis_frame


# Simple test function to verify the tracker works
def test_tracker(video_path: str):
    """Test the tracker on a video file."""
    import cv2
    from object_detection import BagDetector
    
    cap = cv2.VideoCapture(video_path)
    detector = BagDetector()
    tracker = BagTracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect bags
        detections = detector.detect(frame)
        
        # Track bags
        tracks = tracker.update(detections)
        
        # Visualize
        vis_frame = detector.visualize(frame, detections)
        vis_frame = tracker.visualize(vis_frame, tracks)
        
        # Display
        cv2.imshow('Tracking', vis_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_tracker(sys.argv[1])
    else:
        print("Usage: python object_tracking.py <video_path>")
