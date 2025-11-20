#!/usr/bin/env python3
"""
Uniform detection from webcam using Ultralytics YOLO.

This script captures video from your laptop camera and performs real-time detection
focusing on uniforms, people, and uniform-related items. It can detect:
- People (potential uniform wearers)
- Ties (common uniform accessory)
- Sports equipment (sports uniforms)
- And highlight person detections for uniform analysis

Press 'q' to quit the application.

Requirements:
    - ultralytics
    - opencv-python
    - A working webcam/camera
"""

import cv2
import numpy as np
from ultralytics import YOLO


class UniformDetector:
    def __init__(self):
        """Initialize the uniform detection system."""
        # Load a pre-trained YOLOv11 model
        self.model = YOLO("yolo11n.pt")  # nano model for faster inference
        
        # Define uniform-related classes from COCO dataset
        self.uniform_related_classes = {
            0: "person",           # Main target for uniform detection
            27: "tie",            # Dress uniform accessory
            32: "sports ball",    # Sports uniform context
            37: "tennis racket",  # Sports uniform context
            38: "bottle",         # Could be uniform accessories
            39: "wine glass",     # Formal context
            41: "cup",            # Service uniform context
            67: "cell phone",     # Modern uniform accessories
            73: "laptop",         # Professional uniform context
            74: "mouse",          # Professional context
            75: "remote",         # Service uniform context
            76: "keyboard",       # Professional context
        }
        
        # Colors for different detection types
        self.uniform_colors = {
            "person": (0, 255, 0),        # Green for people
            "tie": (255, 0, 0),           # Red for ties
            "sports": (0, 0, 255),        # Blue for sports items
            "professional": (255, 255, 0), # Yellow for professional items
            "other": (128, 128, 128)      # Gray for other items
        }
        
        self.frame_count = 0
        self.person_detections = []
        
    def classify_detection_type(self, class_name):
        """Classify detection into uniform-related categories."""
        if class_name == "person":
            return "person"
        elif class_name == "tie":
            return "tie"
        elif class_name in ["sports ball", "tennis racket"]:
            return "sports"
        elif class_name in ["laptop", "mouse", "keyboard", "cell phone"]:
            return "professional"
        else:
            return "other"
    
    def analyze_person_for_uniform(self, person_box, frame):
        """Analyze a detected person region for uniform characteristics."""
        x1, y1, x2, y2 = map(int, person_box[:4])
        
        # Crop person region
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return "Unknown", 0.0
        
        # Simple color analysis for uniform detection
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common uniform colors
        uniform_colors = {
            "Blue Uniform": ([100, 50, 50], [130, 255, 255]),     # Blue range
            "White/Light Uniform": ([0, 0, 200], [180, 30, 255]), # White/light colors
            "Dark Uniform": ([0, 0, 0], [180, 255, 80]),          # Dark colors
            "Green Uniform": ([35, 50, 50], [85, 255, 255]),      # Green range
        }
        
        best_match = "Casual Clothes"
        best_score = 0.0
        
        total_pixels = person_region.shape[0] * person_region.shape[1]
        
        for uniform_type, (lower, upper) in uniform_colors.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            uniform_pixels = cv2.countNonZero(mask)
            score = uniform_pixels / total_pixels
            
            if score > best_score and score > 0.3:  # At least 30% of the region
                best_match = uniform_type
                best_score = score
        
        return best_match, best_score
    
    def process_frame(self, frame):
        """Process a single frame for uniform detection."""
        self.frame_count += 1
        
        # Run YOLO inference on the frame
        results = self.model(frame, verbose=False)
        
        # Get detections
        detections = results[0].boxes
        
        uniform_count = 0
        person_count = 0
        
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = self.model.names[int(class_id)]
                
                # Filter for uniform-related detections
                if int(class_id) in self.uniform_related_classes:
                    detection_type = self.classify_detection_type(class_name)
                    color = self.uniform_colors.get(detection_type, self.uniform_colors["other"])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Special handling for person detections
                    if class_name == "person":
                        person_count += 1
                        
                        # Analyze for uniform characteristics
                        uniform_type, uniform_score = self.analyze_person_for_uniform(box, frame)
                        
                        if "Uniform" in uniform_type:
                            uniform_count += 1
                            label = f"UNIFORM DETECTED: {uniform_type} ({conf:.2f})"
                            label_color = (0, 0, 255)  # Red for uniform detection
                        else:
                            label = f"Person: {uniform_type} ({conf:.2f})"
                            label_color = (0, 255, 0)  # Green for regular person
                        
                        # Draw label with background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(frame, (int(x1), int(y1) - 25), 
                                    (int(x1) + label_size[0], int(y1)), label_color, -1)
                        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Regular detection label
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add status information
        cv2.putText(frame, f"People Detected: {person_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Potential Uniforms: {uniform_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'q' to quit | Press 's' to save screenshot", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run the uniform detection system."""
        # Open the default camera (usually index 0 for built-in webcam)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting uniform detection...")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("\nLooking for:")
        print("- People (analyzed for uniform characteristics)")
        print("- Ties (uniform accessories)")
        print("- Professional/sports contexts")
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame for uniform detection
            processed_frame = self.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('Uniform Detection System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"uniform_detection_screenshot_{self.frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot saved as {filename}")
        
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Uniform detection stopped")


def main():
    """Main function to run uniform detection."""
    detector = UniformDetector()
    detector.run()


if __name__ == "__main__":
    main()