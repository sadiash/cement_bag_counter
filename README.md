# Cement Bag Counter

A computer vision prototype for detecting and counting cement bags moving on a conveyor belt from a top-down camera view.

## Overview

This prototype system processes videos of cement bags on a conveyor belt to:

1. Detect individual cement bags as they appear on the conveyor belt
2. Track each detected bag as it moves along the belt
3. Count the bags that pass a designated point or area in the frame
4. Visualize the detection, tracking, and counting process

The system is designed to be robust to variations in:
- Lighting conditions
- Bag appearance
- Conveyor belt speed (including starting and stopping)

## Features

- **Background Subtraction Detection**: Isolates moving objects from the static background
- **SORT Tracking Algorithm**: Maintains object identity across frames using Kalman filtering
- **Flexible Counting Options**: 
  - Line-based counting (bags crossing a virtual line)
  - Region-based counting (bags entering a defined area)
- **Interactive Configuration**: Set up counting lines/regions through a simple interface
- **Visualization**: Real-time display of detections, tracks, and counts
- **Performance Optimization**: Testing tools to find optimal parameters

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- SciPy
- FilterPy

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install opencv-python numpy scipy filterpy
```

## Project Structure

```
cement_bag_counter/
├── detection.py  # Bag detection using background subtraction
├── object_tracking.py   # Tracking using SORT algorithm
├── counting.py          # Line and region-based counting
├── main.py              # Main application integrating all components
└── README.md            # Documentation
```


## Configuring the Counting Area

### Line Counter

The line counter counts bags as they cross a virtual line. You can configure:

1. **Line Position**: Define start and end points of the line
2. **Direction**: Specify which direction of crossing should be counted
3. **Minimum Distance**: How close to the line an object must be to be considered crossing
4. **Cooldown Frames**: Frames to wait before counting the same object again

Example configuration in code:

```python
counter = LineCounter(
    line_position=((0, height // 2), (width, height // 2)),  # Horizontal line in the middle
    direction='top-to-bottom',  # Count objects moving from top to bottom
    min_distance=10,  # Must be within 10 pixels of the line
    cooldown_frames=10  # Wait 10 frames before counting the same object again
)
```

### Region Counter

The region counter counts bags as they enter a defined region. You can configure:

1. **Region**: Define the region as (x, y, width, height)
2. **Entry Direction**: Specify from which direction objects must enter to be counted
3. **Minimum Frames**: How many frames an object must be in the region to be counted
4. **Cooldown Frames**: Frames to wait before counting the same object again

Example configuration in code:

```python
counter = RegionCounter(
    region=(x, y, width, height),  # Region coordinates
    entry_direction='any',  # Count objects entering from any direction
    min_frames_in_region=3,  # Must be in region for at least 3 frames
    cooldown_frames=10  # Wait 10 frames before counting the same object again
)
```

## Technical Approach

### Detection

The detection module uses background subtraction to isolate moving objects from the static conveyor belt background:

1. **Background Model**: Adaptive background subtraction using OpenCV's MOG2 algorithm
2. **Morphological Operations**: Clean up the foreground mask to remove noise
3. **Contour Analysis**: Find and filter contours based on area and aspect ratio
4. **Adaptive Parameters**: Automatically adjust detection parameters based on results

### Tracking

The tracking module uses the SORT (Simple Online and Realtime Tracking) algorithm:

1. **Kalman Filter**: Predict object positions between frames
2. **Hungarian Algorithm**: Associate detections with existing tracks
3. **Track Management**: Create new tracks and delete old ones as needed

### Counting

Two counting methods are provided:

1. **Line Counter**: Count objects crossing a virtual line
   - Tracks object positions relative to the line
   - Determines crossing direction
   - Prevents double-counting

2. **Region Counter**: Count objects entering a defined region
   - Tracks objects entering and exiting the region
   - Filters by entry direction
   - Requires minimum time in region to be counted

## Limitations and Future Improvements

### Current Limitations

- Limited to background subtraction for detection, which may struggle with:
  - Static objects (stopped conveyor belt)
  - Significant lighting changes
  - Similar-colored bags and background
- No specific cement bag detector trained on domain data
- Basic tracking that may struggle with dense packing or occlusions
- No support for multiple counting lines/regions simultaneously

### Future Improvements

- **Deep Learning Detection**: Train a specialized detector for cement bags
- **DeepSORT**: Integrate appearance features for more robust tracking
- **Multi-Camera Support**: Combine counts from multiple camera views
- **Speed Estimation**: Calculate conveyor belt speed for adaptive parameters
- **Anomaly Detection**: Identify damaged or improperly positioned bags
- **Edge Deployment**: Optimize for deployment on edge devices
- **Web Interface**: Provide a user-friendly web interface for configuration and monitoring


## Usage

### Basic Usage

python usage.py video/video1.mp4 -d -t region -o video/output1.mp4 


