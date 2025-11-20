#!/usr/bin/env python3
"""
Redarstvo Uniform Detection with Multiple Camera Support and Test Mode
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import json
import time

class RedarstvoDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.uniform_profiles = {}
        self.debug_mode = True
        
    def load_uniform_profiles(self):
        """Load the analyzed uniform profiles."""
        try:
            with open("uniform_config.json", 'r') as f:
                self.uniform_profiles = json.load(f)
            print(f"✅ Loaded {len(self.uniform_profiles)} uniform profiles")
            return True
        except:
            print("❌ No uniform profiles found. Please run analysis first.")
            return False
    
    def test_camera_access(self):
        """Test different camera indices to find working camera."""
        print("🔍 Testing camera access...")
        
        for camera_id in [0, 1, 2]:
            print(f"Trying camera {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {camera_id} working! Resolution: {frame.shape}")
                    cap.release()
                    return camera_id
                else:
                    print(f"❌ Camera {camera_id} opened but can't read frames")
            else:
                print(f"❌ Camera {camera_id} failed to open")
            
            cap.release()
            time.sleep(0.5)
        
        print("❌ No working cameras found")
        return None
    
    def classify_uniform_simple(self, person_region):
        """Simplified uniform classification focused on Redarstvo patterns."""
        if person_region.size == 0:
            return "No Match", 0.0
        
        # Focus on torso area
        h, w = person_region.shape[:2]
        if h > 60 and w > 40:
            torso_region = person_region[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        else:
            torso_region = person_region
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # Analyze color characteristics
        total_pixels = torso_region.shape[0] * torso_region.shape[1]
        
        # Count dark pixels (typical uniform base)
        dark_mask = hsv[:, :, 2] < 70  # Very dark
        dark_count = np.sum(dark_mask)
        dark_ratio = dark_count / total_pixels
        
        # Count blue-ish pixels (Redarstvo often has blue elements)
        blue_mask = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 30)
        blue_count = np.sum(blue_mask)
        blue_ratio = blue_count / total_pixels
        
        # Count gray/neutral pixels
        gray_mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 200)
        gray_count = np.sum(gray_mask)
        gray_ratio = gray_count / total_pixels
        
        # Count light pixels (badges, reflective elements)
        light_mask = hsv[:, :, 2] > 200
        light_count = np.sum(light_mask)
        light_ratio = light_count / total_pixels
        
        # Calculate Redarstvo uniform score
        score = 0.0
        
        # Dark clothing base (should be significant)
        if 0.2 <= dark_ratio <= 0.8:
            score += 0.3
        elif 0.1 <= dark_ratio <= 0.9:
            score += 0.2
        
        # Blue elements (characteristic of many uniforms)
        if blue_ratio > 0.02:  # At least 2% blue
            score += 0.25
        
        # Gray/neutral elements
        if gray_ratio > 0.1:  # At least 10% gray
            score += 0.2
        
        # Light accents (badges, reflective strips)
        if 0.05 <= light_ratio <= 0.3:  # 5-30% light elements
            score += 0.15
        
        # Bonus for balanced color distribution (not all one color)
        color_variety = dark_ratio + blue_ratio + gray_ratio + light_ratio
        if 0.4 <= color_variety <= 0.9:
            score += 0.1
        
        # Debug information
        debug_info = {
            'dark_ratio': dark_ratio,
            'blue_ratio': blue_ratio,
            'gray_ratio': gray_ratio,
            'light_ratio': light_ratio,
            'total_score': score
        }
        
        if self.debug_mode:
            print(f"Debug: Dark={dark_ratio:.2f}, Blue={blue_ratio:.2f}, Gray={gray_ratio:.2f}, Light={light_ratio:.2f}, Score={score:.2f}")
        
        return "Redarstvo_Uniform" if score > 0.4 else "No Match", score
    
    def run_detection(self):
        """Run detection with improved camera handling."""
        if not self.load_uniform_profiles():
            return
        
        # Test camera access
        camera_id = self.test_camera_access()
        if camera_id is None:
            print("❌ Cannot access camera. Please check camera permissions and connections.")
            return
        
        # Open camera with better settings
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"\n🎯 Redarstvo Uniform Detection Active!")
        print(f"Using camera {camera_id}")
        print("Controls:")
        print("  q = quit")
        print("  s = save screenshot") 
        print("  d = toggle debug mode")
        print("  r = reset detection")
        
        frame_count = 0
        detection_history = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Run YOLO detection
            try:
                results = self.model(frame, verbose=False)
                detections = results[0].boxes
            except Exception as e:
                print(f"YOLO detection error: {e}")
                continue
            
            redarstvo_count = 0
            person_count = 0
            
            if detections is not None:
                for i, box in enumerate(detections.data):
                    x1, y1, x2, y2, conf, class_id = box
                    class_name = self.model.names[int(class_id)]
                    
                    if class_name == "person" and conf > 0.5:
                        person_count += 1
                        
                        # Extract person region with safety checks
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            person_region = frame[y1:y2, x1:x2]
                            
                            # Classify uniform
                            uniform_type, uniform_confidence = self.classify_uniform_simple(person_region)
                            
                            # Determine if Redarstvo uniform detected
                            is_redarstvo = uniform_confidence > 0.4 and uniform_type != "No Match"
                            
                            if is_redarstvo:
                                color = (0, 0, 255)  # Red
                                label = f"🚨 REDARSTVO UNIFORM ({uniform_confidence:.2f})"
                                redarstvo_count += 1
                                thickness = 4
                                
                                # Add to detection history
                                detection_history.append(frame_count)
                                if len(detection_history) > 100:  # Keep last 100 detections
                                    detection_history.pop(0)
                                    
                            else:
                                color = (0, 255, 0)  # Green
                                label = f"Person ({conf:.2f})"
                                thickness = 2
                            
                            # Draw detection box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Add label with background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - 35), 
                                        (x1 + label_size[0] + 10, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Debug info if enabled
                            if self.debug_mode and uniform_confidence > 0.1:
                                debug_text = f"Confidence: {uniform_confidence:.3f}"
                                cv2.putText(frame, debug_text, (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Status display
            if redarstvo_count > 0:
                status_color = (0, 0, 255)
                status_text = f"🚨 REDARSTVO DETECTED: {redarstvo_count}"
            else:
                status_color = (0, 255, 0)
                status_text = f"🟢 No uniforms detected"
            
            cv2.putText(frame, status_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
            
            cv2.putText(frame, f"People: {person_count} | Frame: {frame_count}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Debug: {'ON' if self.debug_mode else 'OFF'} | Camera: {camera_id}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, "q=quit | s=save | d=debug | r=reset", 
                       (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Redarstvo Uniform Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"redarstvo_detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('r'):
                detection_history.clear()
                print("🔄 Detection history reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        total_detections = len(detection_history)
        print(f"\n📊 Detection Summary:")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Redarstvo uniform detections: {total_detections}")
        print("Detection stopped")

def main():
    detector = RedarstvoDetector()
    print("=== Redarstvo Uniform Detection System ===")
    print("Enhanced with better camera support and detection logic")
    detector.run_detection()

if __name__ == "__main__":
    main()