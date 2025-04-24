#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Application for Cement Bag Counter

This module integrates detection, tracking, and counting components
into a complete application for counting cement bags on a conveyor belt.
"""

import cv2
import numpy as np
import argparse
import time
import os
from typing import List, Dict, Any, Tuple, Optional

from object_detection import BagDetector
from object_tracking import BagTracker
from counting import LineCounter, RegionCounter


class CementBagCounter:
    """Main application for cement bag counting."""
    
    def __init__(self,
                 counter_type: str = 'line',
                 detection_params: Dict = None,
                 tracking_params: Dict = None,
                 counting_params: Dict = None,
                 show_detections: bool = True,
                 show_tracks: bool = True,
                 show_counter: bool = True,
                 show_fps: bool = True):
        """
        Initialize the cement bag counter application.
        
        Args:
            counter_type: Type of counter to use ('line' or 'region')
            detection_params: Parameters for the bag detector
            tracking_params: Parameters for the bag tracker
            counting_params: Parameters for the counter
            show_detections: Whether to show detection visualizations
            show_tracks: Whether to show tracking visualizations
            show_counter: Whether to show counter visualizations
            show_fps: Whether to show FPS information
        """
        # Initialize components
        self.detector = BagDetector(**(detection_params or {}))
        self.tracker = BagTracker(**(tracking_params or {}))
        
        self.counter_type = counter_type
        self.counter = None  # Will be initialized when processing first frame
        self.counting_params = counting_params or {}
        
        # Visualization options
        self.show_detections = show_detections
        self.show_tracks = show_tracks
        self.show_counter = show_counter
        self.show_fps = show_fps
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # Keep track of last 30 frames for FPS calculation
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (visualization frame, current count)
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        start_time = time.time()
        
        # Initialize counter if not already initialized
        if self.counter is None:
            self._initialize_counter(width, height)
        
        # Detect bags
        detections = self.detector.detect(frame)
        
        # Track bags
        tracks = self.tracker.update(detections)
        
        # Count bags
        count = self.counter.update(tracks)
        
        # Visualize results
        vis_frame = frame.copy()
        
        if self.show_detections:
            vis_frame = self.detector.visualize(vis_frame, detections)
            
        if self.show_tracks:
            vis_frame = self.tracker.visualize(vis_frame, tracks)
            
        if self.show_counter:
            vis_frame = self.counter.visualize(vis_frame)
        
        # Calculate and show FPS
        if self.show_fps:
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
                
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame, count
    
    def _initialize_counter(self, width: int, height: int):
        """
        Initialize the counter based on frame dimensions.
        
        Args:
            width: Frame width
            height: Frame height
        """
        if self.counter_type == 'line':
            # Default: horizontal line in the middle of the frame
            line_position = self.counting_params.get(
                'line_position', 
                ((0, height // 2), (width, height // 2))
            )
            direction = self.counting_params.get('direction', 'top-to-bottom')
            
            self.counter = LineCounter(
                line_position=line_position,
                direction=direction,
                min_distance=self.counting_params.get('min_distance', 10),
                cooldown_frames=self.counting_params.get('cooldown_frames', 10)
            )
        else:  # region
            # Default: region in the center of the frame
            region_width = width // 3
            region_height = height // 3
            region_x = (width - region_width) // 2
            region_y = (height - region_height) // 2
            
            region = self.counting_params.get(
                'region',
                (region_x, region_y, region_width, region_height)
            )
            
            self.counter = RegionCounter(
                region=region,
                entry_direction=self.counting_params.get('entry_direction', 'any'),
                min_frames_in_region=self.counting_params.get('min_frames_in_region', 3),
                cooldown_frames=self.counting_params.get('cooldown_frames', 10)
            )
    
    def process_video(self, 
                      input_path: str, 
                      output_path: str = None,
                      display: bool = True) -> int:
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
                cv2.imshow('Cement Bag Counter', vis_frame)
                
                # Add progress information
                frame_count += 1
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"\rProcessing: {progress:.1f}% (Frame {frame_count}/{total_frames})", end="")
                
                # Check for user input
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
            print()  # New line after progress
            
        return final_count
    
    def configure_counting_line(self, 
                               frame: np.ndarray, 
                               line_position: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                               direction: str = 'top-to-bottom') -> None:
        """
        Configure the counting line interactively.
        
        Args:
            frame: Reference frame for configuration
            line_position: Initial line position as ((x1, y1), (x2, y2))
            direction: Initial direction ('left-to-right', 'right-to-left', 
                      'top-to-bottom', 'bottom-to-top')
        """
        if self.counter_type != 'line':
            print("Counter type is not 'line'. Cannot configure counting line.")
            return
            
        height, width = frame.shape[:2]
        
        # Set default line position if not provided
        if line_position is None:
            line_position = ((0, height // 2), (width, height // 2))
            
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw the line
        (x1, y1), (x2, y2) = line_position
        cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw direction arrow
        if direction == 'top-to-bottom':
            arrow_x, arrow_y = (x1 + x2) // 2, y1 - 20
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y + 40), (0, 255, 255), 2)
        elif direction == 'bottom-to-top':
            arrow_x, arrow_y = (x1 + x2) // 2, y1 + 20
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), (0, 255, 255), 2)
        elif direction == 'left-to-right':
            arrow_x, arrow_y = x1 - 20, (y1 + y2) // 2
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x + 40, arrow_y), (0, 255, 255), 2)
        elif direction == 'right-to-left':
            arrow_x, arrow_y = x1 + 20, (y1 + y2) // 2
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x - 40, arrow_y), (0, 255, 255), 2)
        
        # Display instructions
        cv2.putText(vis_frame, "Configure Counting Line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_frame, "Press 'Enter' to confirm, 'Esc' to cancel", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_frame, "Press 'd' to change direction", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Configure Counting Line', vis_frame)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13:  # Enter key
                # Save configuration
                self.counting_params['line_position'] = line_position
                self.counting_params['direction'] = direction
                
                # Initialize counter with new parameters
                self.counter = LineCounter(
                    line_position=line_position,
                    direction=direction,
                    min_distance=self.counting_params.get('min_distance', 10),
                    cooldown_frames=self.counting_params.get('cooldown_frames', 10)
                )
                
                break
            elif key == 27:  # Esc key
                # Cancel configuration
                break
            elif key == ord('d'):
                # Change direction
                directions = ['top-to-bottom', 'bottom-to-top', 'left-to-right', 'right-to-left']
                current_index = directions.index(direction)
                direction = directions[(current_index + 1) % len(directions)]
                
                # Update visualization
                vis_frame = frame.copy()
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Draw direction arrow
                if direction == 'top-to-bottom':
                    arrow_x, arrow_y = (x1 + x2) // 2, y1 - 20
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y + 40), (0, 255, 255), 2)
                elif direction == 'bottom-to-top':
                    arrow_x, arrow_y = (x1 + x2) // 2, y1 + 20
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), (0, 255, 255), 2)
                elif direction == 'left-to-right':
                    arrow_x, arrow_y = x1 - 20, (y1 + y2) // 2
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x + 40, arrow_y), (0, 255, 255), 2)
                elif direction == 'right-to-left':
                    arrow_x, arrow_y = x1 + 20, (y1 + y2) // 2
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x - 40, arrow_y), (0, 255, 255), 2)
                
                # Display instructions
                cv2.putText(vis_frame, "Configure Counting Line", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_frame, "Press 'Enter' to confirm, 'Esc' to cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_frame, "Press 'd' to change direction", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_frame, f"Direction: {direction}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Configure Counting Line', vis_frame)
        
        cv2.destroyWindow('Configure Counting Line')
    
    def configure_counting_region(self, 
                                 frame: np.ndarray, 
                                 region: Tuple[int, int, int, int] = None,
                                 entry_direction: str = 'any') -> None:
        """
        Configure the counting region interactively.
        
        Args:
            frame: Reference frame for configuration
            region: Initial region as (x, y, w, h)
            entry_direction: Initial entry direction ('left', 'right', 'top', 'bottom', 'any')
        """
        if self.counter_type != 'region':
            print("Counter type is not 'region'. Cannot configure counting region.")
            return
            
        height, width = frame.shape[:2]
        
        # Set default region if not provided
        if region is None:
            region_width = width // 3
            region_height = height // 3
            region_x = (width - region_width) // 2
            region_y = (height - region_height) // 2
            region = (region_x, region_y, region_width, region_height)
            
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw the region
        x, y, w, h = region
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        # Draw entry direction arrow
        if entry_direction == 'left':
            arrow_x, arrow_y = x - 20, y + h // 2
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x + 40, arrow_y), (255, 0, 255), 2)
        elif entry_direction == 'right':
            arrow_x, arrow_y = x + w + 20, y + h // 2
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x - 40, arrow_y), (255, 0, 255), 2)
        elif entry_direction == 'top':
            arrow_x, arrow_y = x + w // 2, y - 20
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y + 40), (255, 0, 255), 2)
        elif entry_direction == 'bottom':
            arrow_x, arrow_y = x + w // 2, y + h + 20
            cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), (255, 0, 255), 2)
        elif entry_direction == 'any':
            cv2.putText(vis_frame, "Any Direction", (x + w + 10, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Display instructions
        cv2.putText(vis_frame, "Configure Counting Region", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(vis_frame, "Press 'Enter' to confirm, 'Esc' to cancel", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(vis_frame, "Press 'd' to change entry direction", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Configure Counting Region', vis_frame)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13:  # Enter key
                # Save configuration
                self.counting_params['region'] = region
                self.counting_params['entry_direction'] = entry_direction
                
                # Initialize counter with new parameters
                self.counter = RegionCounter(
                    region=region,
                    entry_direction=entry_direction,
                    min_frames_in_region=self.counting_params.get('min_frames_in_region', 3),
                    cooldown_frames=self.counting_params.get('cooldown_frames', 10)
                )
                
                break
            elif key == 27:  # Esc key
                # Cancel configuration
                break
            elif key == ord('d'):
                # Change entry direction
                directions = ['any', 'left', 'right', 'top', 'bottom']
                current_index = directions.index(entry_direction)
                entry_direction = directions[(current_index + 1) % len(directions)]
                
                # Update visualization
                vis_frame = frame.copy()
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                
                # Draw entry direction arrow
                if entry_direction == 'left':
                    arrow_x, arrow_y = x - 20, y + h // 2
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x + 40, arrow_y), (255, 0, 255), 2)
                elif entry_direction == 'right':
                    arrow_x, arrow_y = x + w + 20, y + h // 2
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x - 40, arrow_y), (255, 0, 255), 2)
                elif entry_direction == 'top':
                    arrow_x, arrow_y = x + w // 2, y - 20
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y + 40), (255, 0, 255), 2)
                elif entry_direction == 'bottom':
                    arrow_x, arrow_y = x + w // 2, y + h + 20
                    cv2.arrowedLine(vis_frame, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), (255, 0, 255), 2)
                elif entry_direction == 'any':
                    cv2.putText(vis_frame, "Any Direction", (x + w + 10, y + h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Display instructions
                cv2.putText(vis_frame, "Configure Counting Region", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(vis_frame, "Press 'Enter' to confirm, 'Esc' to cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(vis_frame, "Press 'd' to change entry direction", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(vis_frame, f"Entry Direction: {entry_direction}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                cv2.imshow('Configure Counting Region', vis_frame)
        
        cv2.destroyWindow('Configure Counting Region')


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Cement Bag Counter')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file')
    parser.add_argument('-t', '--counter-type', choices=['line', 'region'], default='line',
                        help='Type of counter to use (default: line)')
    parser.add_argument('-c', '--configure', action='store_true',
                        help='Configure counting line/region before processing')
    parser.add_argument('-n', '--no-display', action='store_true',
                        help='Do not display video while processing')
    
    args = parser.parse_args()
    
    # Create counter
    counter = CementBagCounter(counter_type=args.counter_type)
    
    # Configure counting line/region if requested
    if args.configure:
        # Read first frame
        cap = cv2.VideoCapture(args.input)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            if args.counter_type == 'line':
                counter.configure_counting_line(frame)
            else:
                counter.configure_counting_region(frame)
        else:
            print(f"Error: Could not read video file: {args.input}")
            return
    
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
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
