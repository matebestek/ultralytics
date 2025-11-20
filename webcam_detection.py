#!/usr/bin/env python3
"""
Real-time object detection from webcam using Ultralytics YOLO.

This script captures video from your laptop camera and performs real-time object detection
using a pre-trained YOLO model. Press 'q' to quit the application.

Requirements:
    - ultralytics
    - opencv-python
    - A working webcam/camera
"""

import cv2
from ultralytics import YOLO


def main():
    """Run real-time object detection on webcam feed."""
    # Load a pre-trained YOLOv11 model
    model = YOLO("yolo11n.pt")  # nano model for faster inference
    
    # Open the default camera (usually index 0 for built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution (optional - adjust based on your camera capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting webcam object detection...")
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Run YOLO inference on the frame
        results = model(frame, verbose=False)  # Set verbose=False to reduce console output
        
        # Visualize the results on the frame with custom settings
        annotated_frame = results[0].plot(
            conf=True,           # Show confidence scores
            labels=True,         # Show class labels
            line_width=2         # Bounding box line width
        )
        
        # Count detections and add text overlay
        try:
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(annotated_frame, f"Detections: {num_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except:
            pass
        
        cv2.putText(annotated_frame, "Press 'q' to quit", 
                   (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('YOLO Webcam Detection - Real-time Object Detection', annotated_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == "__main__":
    main()