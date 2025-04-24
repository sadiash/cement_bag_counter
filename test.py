#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Cement Bag Counter

This script tests the cement bag counter with different parameters
and provides performance metrics.
"""

import cv2
import numpy as np
import time
import os
import argparse
from typing import Dict, Any

from object_detection import BagDetector
from object_tracking import BagTracker
from counting import LineCounter, RegionCounter
from main import CementBagCounter


def test_detection_parameters(video_path: str, output_dir: str):
    """
    Test different detection parameters and save results.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read first 100 frames of video
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("Error: Could not read video file")
        return
        
    # Parameter combinations to test
    param_combinations = [
        {
            'history': 500,
            'var_threshold': 16,
            'min_area': 1000,
            'max_area': 50000,
            'aspect_ratio_range': (0.5, 2.0),
            'learning_rate': 0.01
        },
        {
            'history': 300,
            'var_threshold': 24,
            'min_area': 800,
            'max_area': 40000,
            'aspect_ratio_range': (0.7, 1.5),
            'learning_rate': 0.02
        },
        {
            'history': 700,
            'var_threshold': 12,
            'min_area': 1200,
            'max_area': 60000,
            'aspect_ratio_range': (0.4, 2.5),
            'learning_rate': 0.005
        }
    ]
    
    # Test each parameter combination
    results = []
    for i, params in enumerate(param_combinations):
        print(f"Testing parameter combination {i+1}/{len(param_combinations)}")
        
        # Create detector with parameters
        detector = BagDetector(
            history=params['history'],
            var_threshold=params['var_threshold'],
            min_area=params['min_area'],
            max_area=params['max_area'],
            aspect_ratio_range=params['aspect_ratio_range'],
            learning_rate=params['learning_rate']
        )
        
        # Process frames
        detection_counts = []
        processing_times = []
        
        for frame in frames:
            start_time = time.time()
            detections = detector.detect(frame)
            processing_time = time.time() - start_time
            
            detection_counts.append(len(detections))
            processing_times.append(processing_time)
            
        # Calculate metrics
        avg_detections = sum(detection_counts) / len(detection_counts)
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Save results
        results.append({
            'params': params,
            'avg_detections': avg_detections,
            'avg_processing_time': avg_processing_time
        })
        
        # Visualize last frame with detections
        vis_frame = detector.visualize(frames[-1], detector.detect(frames[-1]))
        
        # Add parameter information to visualization
        y_pos = 30
        cv2.putText(vis_frame, f"Parameter Set {i+1}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        
        for key, value in params.items():
            cv2.putText(vis_frame, f"{key}: {value}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25
            
        cv2.putText(vis_frame, f"Avg Detections: {avg_detections:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos += 25
        cv2.putText(vis_frame, f"Avg Processing Time: {avg_processing_time*1000:.2f} ms", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save visualization
        cv2.imwrite(os.path.join(output_dir, f"detection_params_{i+1}.jpg"), vis_frame)
    
    # Find best parameter combination
    best_params = None
    best_score = float('-inf')
    
    for result in results:
        # Score based on detection count and processing time
        # Higher detection count is better, lower processing time is better
        score = result['avg_detections'] - result['avg_processing_time'] * 100
        
        if score > best_score:
            best_score = score
            best_params = result['params']
    
    # Save best parameters
    with open(os.path.join(output_dir, "best_detection_params.txt"), "w") as f:
        f.write("Best Detection Parameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        
        # Find the result with the best parameters
        best_result = None
        for result in results:
            if result['params'] == best_params:
                best_result = result
                break
        
        # Write the results
        f.write(f"\nAvg Detections: {best_result['avg_detections']:.2f}\n")
        f.write(f"Avg Processing Time: {best_result['avg_processing_time']*1000:.2f} ms\n")
    
    print(f"Best detection parameters saved to {os.path.join(output_dir, 'best_detection_params.txt')}")
    
    return best_params


def test_tracking_parameters(video_path: str, output_dir: str, detection_params: Dict[str, Any] = None):
    """
    Test different tracking parameters and save results.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save results
        detection_params: Detection parameters to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read first 100 frames of video
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("Error: Could not read video file")
        return
        
    # Create detector with parameters
    detector = BagDetector(**(detection_params or {}))
    
    # Parameter combinations to test
    param_combinations = [
        {
            'max_age': 10,
            'min_hits': 3,
            'iou_threshold': 0.3
        },
        {
            'max_age': 15,
            'min_hits': 2,
            'iou_threshold': 0.4
        },
        {
            'max_age': 8,
            'min_hits': 4,
            'iou_threshold': 0.25
        }
    ]
    
    # Test each parameter combination
    results = []
    for i, params in enumerate(param_combinations):
        print(f"Testing parameter combination {i+1}/{len(param_combinations)}")
        
        # Create tracker with parameters
        tracker = BagTracker(
            max_age=params['max_age'],
            min_hits=params['min_hits'],
            iou_threshold=params['iou_threshold']
        )
        
        # Process frames
        track_counts = []
        processing_times = []
        
        for frame in frames:
            # Detect bags
            detections = detector.detect(frame)
            
            # Track bags
            start_time = time.time()
            tracks = tracker.update(detections)
            processing_time = time.time() - start_time
            
            track_counts.append(len(tracks))
            processing_times.append(processing_time)
            
        # Calculate metrics
        avg_tracks = sum(track_counts) / len(track_counts)
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Save results
        results.append({
            'params': params,
            'avg_tracks': avg_tracks,
            'avg_processing_time': avg_processing_time
        })
        
        # Visualize last frame with tracks
        detections = detector.detect(frames[-1])
        tracks = tracker.update(detections)
        vis_frame = detector.visualize(frames[-1], detections)
        vis_frame = tracker.visualize(vis_frame, tracks)
        
        # Add parameter information to visualization
        y_pos = 30
        cv2.putText(vis_frame, f"Parameter Set {i+1}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        
        for key, value in params.items():
            cv2.putText(vis_frame, f"{key}: {value}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25
            
        cv2.putText(vis_frame, f"Avg Tracks: {avg_tracks:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos += 25
        cv2.putText(vis_frame, f"Avg Processing Time: {avg_processing_time*1000:.2f} ms", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save visualization
        cv2.imwrite(os.path.join(output_dir, f"tracking_params_{i+1}.jpg"), vis_frame)
    
    # Find best parameter combination
    best_params = None
    best_score = float('-inf')
    
    for result in results:
        # Score based on track count and processing time
        # Higher track count is better, lower processing time is better
        score = result['avg_tracks'] - result['avg_processing_time'] * 100
        
        if score > best_score:
            best_score = score
            best_params = result['params']
    
    # Save best parameters
    with open(os.path.join(output_dir, "best_tracking_params.txt"), "w") as f:
        f.write("Best Tracking Parameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        
        # Find the result with the best parameters
        best_result = None
        for result in results:
            if result['params'] == best_params:
                best_result = result
                break
        
        # Write the results
        f.write(f"\nAvg Tracks: {best_result['avg_tracks']:.2f}\n")
        f.write(f"Avg Processing Time: {best_result['avg_processing_time']*1000:.2f} ms\n")
    
    print(f"Best tracking parameters saved to {os.path.join(output_dir, 'best_tracking_params.txt')}")
    
    return best_params


def test_counting_parameters(video_path: str, output_dir: str, detection_params: Dict[str, Any] = None, tracking_params: Dict[str, Any] = None):
    """
    Test different counting parameters and save results.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save results
        detection_params: Detection parameters to use
        tracking_params: Tracking parameters to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read video properties
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create detector and tracker with parameters
    detector = BagDetector(**(detection_params or {}))
    tracker = BagTracker(**(tracking_params or {}))
    
    # Line counter parameter combinations to test
    line_param_combinations = [
        {
            'line_position': ((0, height // 2), (width, height // 2)),
            'direction': 'top-to-bottom',
            'min_distance': 10,
            'cooldown_frames': 10
        },
        {
            'line_position': ((0, height // 3), (width, height // 3)),
            'direction': 'top-to-bottom',
            'min_distance': 15,
            'cooldown_frames': 15
        },
        {
            'line_position': ((0, 2 * height // 3), (width, 2 * height // 3)),
            'direction': 'top-to-bottom',
            'min_distance': 5,
            'cooldown_frames': 5
        }
    ]
    
    # Region counter parameter combinations to test
    region_param_combinations = [
        {
            'region': (width // 4, height // 4, width // 2, height // 2),
            'entry_direction': 'any',
            'min_frames_in_region': 3,
            'cooldown_frames': 10
        },
        {
            'region': (width // 3, height // 3, width // 3, height // 3),
            'entry_direction': 'top',
            'min_frames_in_region': 2,
            'cooldown_frames': 15
        },
        {
            'region': (width // 5, height // 5, 3 * width // 5, 3 * height // 5),
            'entry_direction': 'any',
            'min_frames_in_region': 5,
            'cooldown_frames': 5
        }
    ]
    
    # Test line counter parameters
    print("Testing line counter parameters...")
    line_results = []
    
    for i, params in enumerate(line_param_combinations):
        print(f"Testing parameter combination {i+1}/{len(line_param_combinations)}")
        
        # Create counter with parameters
        counter = CementBagCounter(
            counter_type='line',
            detection_params=detection_params,
            tracking_params=tracking_params,
            counting_params=params
        )
        
        # Process video
        final_count = counter.process_video(
            video_path,
            os.path.join(output_dir, f"line_counter_{i+1}.mp4"),
            display=False
        )
        
        # Save results
        line_results.append({
            'params': params,
            'final_count': final_count
        })
        
        # Save parameters
        with open(os.path.join(output_dir, f"line_counter_params_{i+1}.txt"), "w") as f:
            f.write(f"Line Counter Parameters {i+1}:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nFinal Count: {final_count}\n")
    
    # Test region counter parameters
    print("Testing region counter parameters...")
    region_results = []
    
    for i, params in enumerate(region_param_combinations):
        print(f"Testing parameter combination {i+1}/{len(region_param_combinations)}")
        
        # Create counter with parameters
        counter = CementBagCounter(
            counter_type='region',
            detection_params=detection_params,
            tracking_params=tracking_params,
            counting_params=params
        )
        
        # Process video
        final_count = counter.process_video(
            video_path,
            os.path.join(output_dir, f"region_counter_{i+1}.mp4"),
            display=False
        )
        
        # Save results
        region_results.append({
            'params': params,
            'final_count': final_count
        })
        
        # Save parameters
        with open(os.path.join(output_dir, f"region_counter_params_{i+1}.txt"), "w") as f:
            f.write(f"Region Counter Parameters {i+1}:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nFinal Count: {final_count}\n")
    
    # Compare line and region counter results
    line_counts = [result['final_count'] for result in line_results]
    region_counts = [result['final_count'] for result in region_results]
    
    avg_line_count = sum(line_counts) / len(line_counts) if line_counts else 0
    avg_region_count = sum(region_counts) / len(region_counts) if region_counts else 0
    
    with open(os.path.join(output_dir, "counting_comparison.txt"), "w") as f:
        f.write("Counting Comparison:\n")
        f.write(f"Average Line Counter Count: {avg_line_count:.2f}\n")
        f.write(f"Average Region Counter Count: {avg_region_count:.2f}\n")
        f.write("\nLine Counter Results:\n")
        for i, result in enumerate(line_results):
            f.write(f"  Set {i+1}: {result['final_count']}\n")
        f.write("\nRegion Counter Results:\n")
        for i, result in enumerate(region_results):
            f.write(f"  Set {i+1}: {result['final_count']}\n")
    
    print(f"Counting comparison saved to {os.path.join(output_dir, 'counting_comparison.txt')}")
    
    # Return best parameters based on consistency
    if abs(avg_line_count - avg_region_count) < 2:
        # Counts are similar, choose based on consistency
        line_variance = sum((c - avg_line_count) ** 2 for c in line_counts) / len(line_counts) if line_counts else float('inf')
        region_variance = sum((c - avg_region_count) ** 2 for c in region_counts) / len(region_counts) if region_counts else float('inf')
        
        if line_variance <= region_variance:
            # Line counter is more consistent
            best_counter_type = 'line'
            best_counting_params = line_results[line_counts.index(min(line_counts, key=lambda x: abs(x - avg_line_count)))]['params']
        else:
            # Region counter is more consistent
            best_counter_type = 'region'
            best_counting_params = region_results[region_counts.index(min(region_counts, key=lambda x: abs(x - avg_region_count)))]['params']
    else:
        # Counts are different, choose the higher one (assuming undercounting is more likely)
        if avg_line_count > avg_region_count:
            best_counter_type = 'line'
            best_counting_params = line_results[line_counts.index(max(line_counts))]['params']
        else:
            best_counter_type = 'region'
            best_counting_params = region_results[region_counts.index(max(region_counts))]['params']
    
    # Save best parameters
    with open(os.path.join(output_dir, "best_counting_params.txt"), "w") as f:
        f.write(f"Best Counter Type: {best_counter_type}\n")
        f.write("Best Counting Parameters:\n")
        for key, value in best_counting_params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Best counting parameters saved to {os.path.join(output_dir, 'best_counting_params.txt')}")
    
    return best_counter_type, best_counting_params


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Test Cement Bag Counter')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output-dir', default='test_results',
                        help='Directory to save test results (default: test_results)')
    parser.add_argument('-d', '--detection', action='store_true',
                        help='Test detection parameters')
    parser.add_argument('-t', '--tracking', action='store_true',
                        help='Test tracking parameters')
    parser.add_argument('-c', '--counting', action='store_true',
                        help='Test counting parameters')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Test all parameters')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test parameters
    detection_params = None
    tracking_params = None
    
    if args.all or args.detection:
        print("Testing detection parameters...")
        detection_params = test_detection_parameters(args.input, args.output_dir)
    
    if args.all or args.tracking:
        print("Testing tracking parameters...")
        tracking_params = test_tracking_parameters(args.input, args.output_dir, detection_params)
    
    if args.all or args.counting:
        print("Testing counting parameters...")
        best_counter_type, best_counting_params = test_counting_parameters(
            args.input, args.output_dir, detection_params, tracking_params
        )
    
    print("Testing complete!")


if __name__ == "__main__":
    main()
