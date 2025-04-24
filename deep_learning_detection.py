#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Learning Implementation for Cement Bag Detection

This script demonstrates how to use deep learning (specifically YOLO) 
for cement bag detection as an enhancement to the current system.
"""

import cv2
import numpy as np
import argparse
import os
import time
from typing import Dict, Any, List, Tuple

# For tracking and counting, we'll reuse our existing modules
from object_tracking import BagTracker
from counting import RegionCounter
from config import TRACKING_PARAMS, REGION_COUNTING_PARAMS


class DeepLearningDetector:
    """Cement bag detector using YOLO deep learning model."""
    
    def __init__(self, model_path: str, config_path: str, classes_path: str, 
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Initialize the deep learning detector.
        
        Args:
            model_path: Path to the YOLO weights file
            config_path: Path to the YOLO config file
            classes_path: Path to the classes file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        
        # Set preferred backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use DNN_TARGET_CUDA for GPU
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Generate random colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
        
        # Debug images
        self.debug_images = {}
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cement bags in the given frame using YOLO.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing detection information
        """
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # YOLO returns coordinates relative to the image
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Prepare detections
        detections = []
        
        for i in indices:
            if isinstance(i, list):  # OpenCV 4.5.4 and earlier returns a list
                i = i[0]
                
            box = boxes[i]
            x, y, w, h = box
            
            # Calculate centroid
            cx = x + w // 2
            cy = y + h // 2
            
            # Add detection
            detections.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'area': w * h,
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': self.classes[class_ids[i]],
                'detection_type': 'deep_learning'
            })
        
        return detections
    
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
        
        # Draw detections
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det['centroid']
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Get color for this class
            color = self.colors[class_id].tolist()
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cv2.circle(vis_frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame


class DeepLearningCementBagCounter:
    """Application for cement bag counting using deep learning."""
    
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 classes_path: str,
                 counter_type: str = 'region',
                 tracking_config: Dict = None,
                 counting_config: Dict = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize the deep learning cement bag counter application.
        
        Args:
            model_path: Path to the YOLO weights file
            config_path: Path to the YOLO config file
            classes_path: Path to the classes file
            counter_type: Type of counter to use ('line' or 'region')
            tracking_config: Configuration for the bag tracker
            counting_config: Configuration for the counter
            confidence_threshold: Minimum confidence for detections
        """
        # Initialize detector
        self.detector = DeepLearningDetector(
            model_path=model_path,
            config_path=config_path,
            classes_path=classes_path,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize tracker
        self.tracker = BagTracker(**(tracking_config or TRACKING_PARAMS))
        
        # Set up counter type and parameters
        self.counter_type = counter_type
        self.counter = None  # Will be initialized when processing first frame
        self.counting_config = counting_config or REGION_COUNTING_PARAMS
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # Keep track of last 30 frames for FPS calculation
        
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
        
        # Detect bags using deep learning
        detections = self.detector.detect(frame)
        
        # Track bags
        tracks = self.tracker.update(detections)
        
        # Count bags
        count = self.counter.update(tracks)
        
        # Create visualization
        vis_frame = self.detector.visualize(frame, detections)
        vis_frame = self.tracker.visualize(vis_frame, tracks)
        vis_frame = self.counter.visualize(vis_frame)
        
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
            from counting import LineCounter
            # Use line counter
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
            # Use region counter
            region = self.counting_config.get(
                'region',
                (50, 100, 350, 300)  # Default to left side of frame
            )
            
            self.counter = RegionCounter(
                region=region,
                entry_direction=self.counting_config.get('entry_direction', 'bottom'),
                min_frames_in_region=self.counting_config.get('min_frames_in_region', 2),
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
        
        print(f"Processing video with deep learning detection and {self.counter_type} counter")
        
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
                cv2.imshow('Deep Learning Cement Bag Counter', vis_frame)
                
                # Add progress information
                frame_count += 1
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"\rProcessing: {progress:.1f}% (Frame {frame_count}/{total_frames}) | Current count: {count}", end="")
                
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


def download_yolo_model(output_dir):
    """
    Download YOLOv4 model files if they don't exist.
    
    Args:
        output_dir: Directory to save model files
    """
    import urllib.request
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs for YOLOv4 files
    urls = {
        'config': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
        'weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
        'classes': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
    }
    
    # Download files
    for name, url in urls.items():
        output_path = os.path.join(output_dir, f"yolov4_{name}.txt" if name == 'classes' else f"yolov4_{name}")
        
        if not os.path.exists(output_path):
            print(f"Downloading {name} file...")
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded {name} file to {output_path}")
        else:
            print(f"{name.capitalize()} file already exists at {output_path}")
    
    return {
        'config': os.path.join(output_dir, "yolov4_config"),
        'weights': os.path.join(output_dir, "yolov4_weights"),
        'classes': os.path.join(output_dir, "yolov4_classes.txt")
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Deep Learning Cement Bag Counter')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file')
    parser.add_argument('-t', '--counter-type', choices=['line', 'region'], default='region',
                        help='Type of counter to use (default: region)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('-m', '--model-dir', default='./models',
                        help='Directory to store model files (default: ./models)')
    parser.add_argument('-n', '--no-display', action='store_true',
                        help='Do not display video while processing')
    
    args = parser.parse_args()
    
    # Download YOLO model if needed
    model_paths = download_yolo_model(args.model_dir)
    
    # Create counter
    counter = DeepLearningCementBagCounter(
        model_path=model_paths['weights'],
        config_path=model_paths['config'],
        classes_path=model_paths['classes'],
        counter_type=args.counter_type,
        confidence_threshold=args.confidence
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
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
