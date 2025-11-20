#!/usr/bin/env python3
"""
Robust Webcam Redarstvo Detection System
Enhanced error handling for Windows camera systems
"""

import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import sys

class RobustWebcamDetector:
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
        except FileNotFoundError:
            print("❌ No uniform profiles found. Please run analysis first.")
            return False
        except Exception as e:
            print(f"❌ Error loading uniform profiles: {e}")
            return False
    
    def test_camera_safely(self, camera_id):
        """Safely test a camera with proper error handling."""
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Use DirectShow for Windows
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)  # Fallback to default backend
            
            if cap.isOpened():
                # Give camera time to initialize
                time.sleep(0.5)
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    height, width = frame.shape[:2]
                    cap.release()
                    return True, f"{width}x{height}"
                else:
                    cap.release()
                    return False, "No frames"
            else:
                return False, "Can't open"
                
        except Exception as e:
            return False, f"Error: {str(e)[:30]}"
    
    def find_working_cameras(self):
        """Find all working cameras with robust detection."""
        print("🔍 Scanning for webcams...")
        
        working_cameras = []
        
        # Test cameras 0-5 (most common range)
        for i in range(6):
            print(f"   Testing camera {i}...", end=" ", flush=True)
            
            is_working, info = self.test_camera_safely(i)
            
            if is_working:
                working_cameras.append({
                    'id': i,
                    'resolution': info,
                    'status': 'Working'
                })
                print(f"✅ {info}")
            else:
                print(f"❌ {info}")
        
        return working_cameras
    
    def setup_camera(self, camera_id):
        """Setup camera with Windows-optimized settings."""
        print(f"📹 Connecting to camera {camera_id}...")
        
        # Try DirectShow first (Windows optimized)
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("   DirectShow failed, trying default backend...")
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return None
        
        # Wait for camera to initialize
        time.sleep(1)
        
        # Set camera properties
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to enable auto-exposure and autofocus if supported
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not set all camera properties: {e}")
        
        # Test if we can read frames
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✅ Camera ready: {width}x{height}")
            return cap
        else:
            print("❌ Camera opened but can't read frames")
            cap.release()
            return None
    
    def classify_uniform_simple(self, person_region):
        """Simplified but effective uniform classification."""
        if person_region.size == 0:
            return "No Match", 0.0
        
        h, w = person_region.shape[:2]
        
        # Focus on torso area
        if h > 60 and w > 40:
            torso = person_region[int(h*0.25):int(h*0.75), int(w*0.15):int(w*0.85)]
        else:
            torso = person_region
        
        if torso.size == 0:
            return "No Match", 0.0
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total_pixels = torso.shape[0] * torso.shape[1]
        
        # Simple but effective uniform detection
        # Dark clothing (uniforms are typically dark)
        dark_pixels = np.sum(hsv[:, :, 2] < 80) / total_pixels
        
        # Blue elements (common in many uniforms)
        blue_pixels = np.sum(
            (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 30)
        ) / total_pixels
        
        # Gray/neutral elements
        gray_pixels = np.sum(
            (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 180)
        ) / total_pixels
        
        # Light elements (badges, reflective strips)
        light_pixels = np.sum(hsv[:, :, 2] > 200) / total_pixels
        
        # Calculate uniform probability
        score = 0.0
        
        # Primary scoring
        if 0.2 <= dark_pixels <= 0.8:  # Appropriate amount of dark clothing
            score += 0.4
        
        if blue_pixels > 0.01:  # Blue elements present
            score += 0.3
        
        if 0.1 <= gray_pixels <= 0.6:  # Good amount of neutral colors
            score += 0.2
        
        if 0.01 <= light_pixels <= 0.25:  # Some light elements (badges, etc.)
            score += 0.1
        
        if self.debug_mode:
            print(f"Debug [{h}x{w}]: Dark={dark_pixels:.2f}, Blue={blue_pixels:.3f}, Score={score:.2f}")
        
        return "Redarstvo_Uniform" if score > 0.35 else "No Match", score
    
    def run_detection(self):
        """Run the webcam detection system."""
        print("📹 === Robust Webcam Redarstvo Detection ===")
        
        # Load uniform profiles
        if not self.load_uniform_profiles():
            return
        
        # Find working cameras
        cameras = self.find_working_cameras()
        
        if not cameras:
            print("\n❌ No working cameras found!")
            print("\n💡 Troubleshooting tips:")
            print("1. Make sure your webcam is properly connected")
            print("2. Close other programs that might be using the camera")
            print("3. Try reconnecting the webcam")
            print("4. Check Windows camera permissions")
            return
        
        # Select camera
        if len(cameras) == 1:
            camera_id = cameras[0]['id']
            print(f"\n✅ Using camera {camera_id} ({cameras[0]['resolution']})")
        else:
            print(f"\n📹 Found {len(cameras)} cameras:")
            for i, cam in enumerate(cameras):
                print(f"{i+1}. Camera {cam['id']} - {cam['resolution']}")
            
            try:
                choice = input("Select camera (1-{}) or Enter for first: ".format(len(cameras))).strip()
                if choice:
                    camera_id = cameras[int(choice) - 1]['id']
                else:
                    camera_id = cameras[0]['id']
                print(f"✅ Selected camera {camera_id}")
            except:
                camera_id = cameras[0]['id']
                print(f"✅ Using default camera {camera_id}")
        
        # Setup camera
        cap = self.setup_camera(camera_id)
        if cap is None:
            return
        
        print(f"\n🎯 Detection Starting!")
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  s = save screenshot")  
        print("  d = toggle debug")
        print("  r = reset counter")
        
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Camera disconnected")
                    break
                
                frame_count += 1
                height, width = frame.shape[:2]
                
                # YOLO detection
                try:
                    results = self.model(frame, verbose=False)
                    detections = results[0].boxes
                except Exception as e:
                    if self.debug_mode:
                        print(f"YOLO error: {e}")
                    continue
                
                current_detections = 0
                people_count = 0
                
                if detections is not None:
                    for box in detections.data:
                        x1, y1, x2, y2, conf, class_id = box
                        class_name = self.model.names[int(class_id)]
                        
                        if class_name == "person" and conf > 0.4:
                            people_count += 1
                            
                            # Get person region
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            if x2 > x1 and y2 > y1:
                                person_region = frame[y1:y2, x1:x2]
                                
                                # Classify uniform
                                uniform_type, uniform_score = self.classify_uniform_simple(person_region)
                                
                                is_uniform = uniform_score > 0.35 and uniform_type != "No Match"
                                
                                if is_uniform:
                                    color = (0, 0, 255)  # Red
                                    label = f"🚨 UNIFORM ({uniform_score:.2f})"
                                    current_detections += 1
                                    detection_count += 1
                                    thickness = 3
                                else:
                                    color = (0, 255, 0)  # Green
                                    label = f"Person ({conf:.2f})"
                                    thickness = 2
                                
                                # Draw detection
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Add label
                                cv2.rectangle(frame, (x1, y1-30), (x1+200, y1), color, -1)
                                cv2.putText(frame, label, (x1+5, y1-8), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Status display
                if current_detections > 0:
                    status = f"🚨 UNIFORMS DETECTED: {current_detections}"
                    status_color = (0, 0, 255)
                else:
                    status = "🟢 Monitoring..."
                    status_color = (0, 255, 0)
                
                cv2.putText(frame, status, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                cv2.putText(frame, f"People: {people_count} | Total: {detection_count} | Frame: {frame_count}", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Camera: {camera_id} | Debug: {'ON' if self.debug_mode else 'OFF'}", 
                           (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Webcam Uniform Detection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"webcam_uniform_{detection_count}_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('r'):
                    detection_count = 0
                    print("🔄 Counter reset")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 Final Results:")
            print(f"   📹 Camera: {camera_id}")
            print(f"   🎬 Frames: {frame_count}")
            print(f"   🚨 Detections: {detection_count}")

def main():
    detector = RobustWebcamDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()