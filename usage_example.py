#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage example for Cement Bag Counter with configuration file

This script demonstrates how to use the Cement Bag Counter with the
optimized parameters from the configuration file.
"""

import cv2
import argparse
from main import CementBagCounter
from config import (
    DETECTION_PARAMS,
    TRACKING_PARAMS,
    LINE_COUNTING_PARAMS,
    REGION_COUNTING_PARAMS,
    BRIGHT_LIGHTING_PARAMS,
    SMALL_BAG_PARAMS,
    LARGE_BAG_PARAMS,
    FAST_CONVEYOR_PARAMS
)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Cement Bag Counter with Configuration')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Path to output video file')
    parser.add_argument('-t', '--counter-type', choices=['line', 'region'], default='line',
                        help='Type of counter to use (default: line)')
    parser.add_argument('-c', '--conditions', choices=['standard', 'bright', 'small', 'large', 'fast'],
                        default='standard', help='Lighting and bag conditions (default: standard)')
    parser.add_argument('-n', '--no-display', action='store_true',
                        help='Do not display video while processing')
    
    args = parser.parse_args()
    
    # Select detection parameters based on conditions
    if args.conditions == 'bright':
        detection_params = BRIGHT_LIGHTING_PARAMS
    elif args.conditions == 'small':
        detection_params = {**DETECTION_PARAMS, **SMALL_BAG_PARAMS}
    elif args.conditions == 'large':
        detection_params = {**DETECTION_PARAMS, **LARGE_BAG_PARAMS}
    elif args.conditions == 'fast':
        detection_params = {**DETECTION_PARAMS, **FAST_CONVEYOR_PARAMS}
        tracking_params = {**TRACKING_PARAMS, **FAST_CONVEYOR_PARAMS}
    else:
        detection_params = DETECTION_PARAMS
        
    # Select tracking parameters
    if args.conditions == 'fast':
        tracking_params = {**TRACKING_PARAMS, **FAST_CONVEYOR_PARAMS}
    else:
        tracking_params = TRACKING_PARAMS
        
    # Select counting parameters based on counter type
    if args.counter_type == 'line':
        counting_params = LINE_COUNTING_PARAMS
    else:
        counting_params = REGION_COUNTING_PARAMS
    
    # Create counter with selected parameters
    counter = CementBagCounter(
        counter_type=args.counter_type,
        detection_params=detection_params,
        tracking_params=tracking_params,
        counting_params=counting_params
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
