#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Updated Enhanced Usage Example for Cement Bag Counter

This script demonstrates how to use the enhanced cement bag detection
and counting system with the adjusted parameters to fix double-counting
and improve region-based counting.
"""

import cv2
import argparse
import os
import time
from typing import Dict, Any

from detection import BagDetector
from object_tracking import BagTracker
from counting import LineCounter, RegionCounter
from config import (
    DETECTION_PARAMS,
    TRACKING_PARAMS,
    LINE_COUNTING_PARAMS,
    REGION_COUNTING_PARAMS,
    PREPROCESSING_PARAMS,
    CONTOUR_DETECTION_PARAMS,
    FRAME_DIFF_PARAMS,
    HYBRID_DETECTION_PARAMS,
    ROI_PARAMS,
    DEBUG_PARAMS,
    EXTREME_ANTI_DOUBLE_COUNTING
)


class UpdatedCementBagCounter:
    """Updated application for cement bag counting with adjusted parameters."""
    
    def __init__(self,
                 counter_type: str = 'region',  # Default to region counter
                 detection_config: Dict = None,
                 tracking_config: Dict = None,
                 counting_config: Dict = None,
                 show_debug: bool = True,
                 extreme_mode: bool = False):
        """
        Initialize the updated cement bag counter application.
        
        Args:
            counter_type: Type of counter to use ('line' or 'region')
            detection_config: Configuration for the bag detector
            tracking_config: Configuration for the bag tracker
            counting_config: Configuration for the counter
            show_debug: Whether to show debug visualizations
            extreme_mode: Whether to use extreme anti-double-counting parameters
        """
        # Combine all detection parameters
        self.detection_config = {
            **DETECTION_PARAMS,
            **PREPROCESSING_PARAMS,
            **CONTOUR_DETECTION_PARAMS,
            **FRAME_DIFF_PARAMS,
            **HYBRID_DETECTION_PARAMS,
            **ROI_PARAMS
        }
        
        # Add debug parameters if enabled
        if show_debug:
            self.detection_config.update(DEBUG_PARAMS)
            
        # Override with user-provided config if any
        if detection_config:
            self.detection_config.update(detection_config)
        
        # Set up tracking parameters
        self.tracking_config = dict(TRACKING_PARAMS)
        if tracking_config:
            self.tracking_config.update(tracking_config)
            
        # Apply extreme mode if enabled
        if extreme_mode:
            self.tracking_config.update(EXTREME_ANTI_DOUBLE_COUNTING)
            
        # Initialize components
        self.detector = BagDetector(self.detection_config)
        self.tracker = BagTracker(**self.tracking_config)
        
        # Set up counter type and parameters
        self.counter_type = counter_type
        
        # Select appropriate counting parameters based on counter type
        if counter_type == 'line':
            self.counting_config = dict(LINE_COUNTING_PARAMS)
        else:  # region
            self.counting_config = dict(REGION_COUNTING_PARAMS)
            
        # Override with user-provided counting config if any
        if counting_config:
            self.counting_config.update(counting_config)
            
        # Apply extreme mode to counting parameters if enabled
        if extreme_mode:
            self.counting_config.update(EXTREME_ANTI_DOUBLE_COUNTING)
            
        self.counter = None  # Will be initialized when processing first frame
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # Keep track of last 30 frames for FPS calculation
        
        # Debug info
        self.show_debug = show_debug
        self.extreme_mode = extreme_mode
        
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (visualized frame, current count)
        """
        start_time = time.time()
        
        # Initialize counter if not already done
        if self.counter is None:
            height, width = frame.shape[:2]
            self._initialize_counter(width, height)
        
        # Detect bags using enhanced detector
        detections = self.detector.detect(frame)
        
        # Track bags
        tracks = self.tracker.update(detections)
        
        # Count bags
        count = self.counter.update(tracks)
        
        # Create visualization
        vis_frame = self.detector.visualize(frame, detections)
        vis_frame = self.tracker.visualize(vis_frame, tracks)
        vis_frame = self.counter.visualize(vis_frame)
        
        # Add mode information to visualization
        mode_text = f"Mode: {'Extreme' if self.extreme_mode else 'Standard'}"
        counter_text = f"Counter: {self.counter_type.capitalize()}"
        cv2.putText(vis_frame, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, counter_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate and show FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame, count
    
    def _initialize_counter(self, width, height):
        """
        Initialize the counter based on frame dimensions.
        
        Args:
            width: Frame width
            height: Frame height
        """
        if self.counter_type == 'line':
            # Use line counter with adjusted parameters
            line_position = self.counting_config.get(
                'line_position', 
                ((0, height // 2), (width, height // 2))
            )
            direction = self.counting_config.get('direction', 'any')
            
            self.counter = LineCounter(
                line_position=line_position,
                direction=direction,
                min_distance=self.counting_config.get('min_distance', 10),
                cooldown_frames=self.counting_config.get('cooldown_frames', 40)
            )
        else:  # region
            # Use region counter with adjusted parameters
            region = self.counting_config.get(
                'region',
                (50, 100, 350, 300)  # Default to left side of frame
            )
            
            self.counter = RegionCounter(
                region=region,
                entry_direction=self.counting_config.get('entry_direction', 'any'),
                min_frames_in_region=self.counting_config.get('min_frames_in_region', 1),
                cooldown_frames=self.counting_config.get('cooldown_frames', 40)
            )
    
    def process_video(self, input_path, output_path=None, display=True):
        """
        Process a video file.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file (optional)
            display: Whether to display the video while processing
            
        Returns:
            Final count of bags
        """
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        final_count = 0
        
        print(f"Processing video with {self.counter_type} counter in {'extreme' if self.extreme_mode else 'standard'} mode")
        print(f"Debug visualization: {'enabled' if self.show_debug else 'disabled'}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            vis_frame, count = self.process_frame(frame)
            final_count = count
            
            # Write frame to output video
            if writer:
                writer.write(vis_frame)
                
            # Display frame
            if display:
                cv2.imshow('Updated Cement Bag Counter', vis_frame)
                
                # Add progress information
                frame_count += 1
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"\rProcessing: {progress:.1f}% (Frame {frame_count}/{total_frames}) | Current count: {count}", end="")
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    # Toggle extreme mode
                    self.extreme_mode = not self.extreme_mode
                    print(f"\nSwitched to {'extreme' if self.extreme_mode else 'standard'} mode")
                    
                    # Update parameters
                    if self.extreme_mode:
                        self.tracking_config.update(EXTREME_ANTI_DOUBLE_COUNTING)
                        self.counting_config.update(EXTREME_ANTI_DOUBLE_COUNTING)
                    else:
                        # Reset to default parameters
                        self.tracking_config = dict(TRACKING_PARAMS)
                        if self.counter_type == 'line':
                            self.counting_config = dict(LINE_COUNTING_PARAMS)
                        else:
                            self.counting_config = dict(REGION_COUNTING_PARAMS)
                    
                    # Reinitialize components with new parameters
                    self.tracker = BagTracker(**self.tracking_config)
                    self.counter = None  # Will be reinitialized on next frame
                elif key == ord('c'):
                    # Toggle counter type
                    self.counter_type = 'region' if self.counter_type == 'line' else 'line'
                    print(f"\nSwitched to {self.counter_type} counter")
                    
                    # Update counting parameters
                    if self.counter_type == 'line':
                        self.counting_config = dict(LINE_COUNTING_PARAMS)
                    else:
                        self.counting_config = dict(REGION_COUNTING_PARAMS)
                        
                    # Apply extreme mode if enabled
                    if self.extreme_mode:
                        self.counting_config.update(EXTREME_ANTI_DOUBLE_COUNTING)
                        
                    # Reinitialize counter
                    self.counter = None  # Will be reinitialized on next frame
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
            print()  # New line after progress
            
        return final_count


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Updated Cement Bag Counter')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file')
    parser.add_argument('-t', '--counter-type', choices=['line', 'region'], default='region',
                        help='Type of counter to use (default: region)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Show debug visualizations')
    parser.add_argument('-e', '--extreme', action='store_true',
                        help='Use extreme anti-double-counting parameters')
    parser.add_argument('-n', '--no-display', action='store_true',
                        help='Do not display video while processing')
    
    args = parser.parse_args()
    
    # Create counter
    counter = UpdatedCementBagCounter(
        counter_type=args.counter_type,
        show_debug=args.debug,
        extreme_mode=args.extreme
    )
    
    # Process video
    try:
        final_count = counter.process_video(
            args.input,
            args.output,
            not args.no_display
        )
        
        print(f"Final count: {final_count}")
        
        if args.output:
            print(f"Output video saved to: {args.output}")
            
        print("\nKeyboard controls during playback:")
        print("  q: Quit")
        print("  e: Toggle extreme mode")
        print("  c: Toggle counter type (line/region)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
