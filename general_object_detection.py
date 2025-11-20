#!/usr/bin/env python3
"""
General Object Detection System
Basic YOLO-based object detection without uniform-specific features
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

class GeneralObjectDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.debug_mode = False
        
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
        print("🔍 Scanning for cameras...")
        
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
    
    def run_detection(self):
        """Run general object detection without uniform-specific features."""
        print("📹 === General Object Detection System ===")
        
        # Find working cameras
        cameras = self.find_working_cameras()
        
        if not cameras:
            print("\n❌ No working cameras found!")
            print("\n💡 Troubleshooting tips:")
            print("1. Make sure your camera is properly connected")
            print("2. Close other programs that might be using the camera")
            print("3. Try reconnecting the camera")
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
            except ValueError:
                camera_id = cameras[0]['id']
                print(f"✅ Using default camera {camera_id}")
        
        # Setup camera
        cap = self.setup_camera(camera_id)
        if cap is None:
            return
        
        print("\n🎯 Object Detection Active!")
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  s = save screenshot")  
        print("  d = toggle debug info")
        print("  c = toggle confidence display")
        
        frame_count = 0
        show_confidence = True
        
        # YOLO class names we're interested in
        interesting_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog', 'cat']
        
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
                
                detected_objects = {}
                
                if detections is not None:
                    for box in detections.data:
                        x1, y1, x2, y2, conf, class_id = box
                        class_name = self.model.names[int(class_id)]
                        
                        # Only show objects with reasonable confidence
                        if conf > 0.3:
                            # Count detected objects
                            if class_name in detected_objects:
                                detected_objects[class_name] += 1
                            else:
                                detected_objects[class_name] = 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            # Choose color based on object type
                            if class_name == 'person':
                                color = (0, 255, 0)  # Green for people
                            elif class_name in ['car', 'truck', 'bus']:
                                color = (255, 0, 0)  # Blue for vehicles
                            elif class_name in ['dog', 'cat']:
                                color = (0, 255, 255)  # Yellow for animals
                            else:
                                color = (255, 255, 0)  # Cyan for other objects
                            
                            # Draw bounding box
                            thickness = 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Add label
                            if show_confidence:
                                label = f"{class_name} ({conf:.2f})"
                            else:
                                label = class_name
                            
                            # Background for text
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                            cv2.putText(frame, label, (x1+5, y1-8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Status display
                status_y = 50
                cv2.putText(frame, f"🎯 General Object Detection", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show detected objects count
                status_y += 40
                if detected_objects:
                    object_text = " | ".join([f"{obj}: {count}" for obj, count in detected_objects.items()])
                    cv2.putText(frame, f"Detected: {object_text}", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No objects detected", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                # Frame info
                status_y += 30
                cv2.putText(frame, f"Frame: {frame_count} | Camera: {camera_id}", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Debug info
                if self.debug_mode:
                    status_y += 25
                    cv2.putText(frame, f"Resolution: {width}x{height} | Debug: ON", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Controls info
                cv2.putText(frame, "q=quit | s=save | d=debug | c=confidence", 
                           (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('General Object Detection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"detection_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('c'):
                    show_confidence = not show_confidence
                    print(f"📊 Confidence: {'ON' if show_confidence else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n📊 Final Results:")
            print(f"   📹 Camera: {camera_id}")
            print(f"   🎬 Frames processed: {frame_count}")
            print("   🎯 General object detection completed")

def main():
    print("📹 === General Object Detection System ===")
    print("Basic YOLO object detection for common objects")
    print("Detects: people, vehicles, animals, and more")
    print("\n🚀 Starting detection...")
    
    detector = GeneralObjectDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()