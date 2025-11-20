#!/usr/bin/env python3
"""
Enhanced Object Detection System with Zoom Control
YOLO-based object detection with interactive zoom slider
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

class ZoomableObjectDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.debug_mode = False
        self.zoom_factor = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 3.0
        self.zoom_center_x = 0.5  # Center of zoom (0.0-1.0)
        self.zoom_center_y = 0.5  # Center of zoom (0.0-1.0)
        
        # Brightness control
        self.brightness = 0  # Range: -100 to +100
        self.brightness_min = -100
        self.brightness_max = 100
        
    def zoom_callback(self, val):
        """Callback function for zoom slider."""
        self.zoom_factor = self.zoom_min + (val / 100.0) * (self.zoom_max - self.zoom_min)
    
    def brightness_callback(self, val):
        """Callback function for brightness slider."""
        self.brightness = self.brightness_min + (val / 100.0) * (self.brightness_max - self.brightness_min)
    
    def apply_zoom(self, frame):
        """Apply zoom to the frame."""
        if self.zoom_factor == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        
        # Calculate the size of the zoomed region
        zoomed_width = int(width / self.zoom_factor)
        zoomed_height = int(height / self.zoom_factor)
        
        # Calculate the top-left corner of the zoom region
        center_x = int(width * self.zoom_center_x)
        center_y = int(height * self.zoom_center_y)
        
        x1 = max(0, center_x - zoomed_width // 2)
        y1 = max(0, center_y - zoomed_height // 2)
        x2 = min(width, x1 + zoomed_width)
        y2 = min(height, y1 + zoomed_height)
        
        # Adjust if we're near the edges
        if x2 - x1 < zoomed_width:
            x1 = max(0, x2 - zoomed_width)
        if y2 - y1 < zoomed_height:
            y1 = max(0, y2 - zoomed_height)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    def apply_brightness(self, frame):
        """Apply brightness adjustment to the frame."""
        if self.brightness == 0:
            return frame
        
        # Convert brightness from -100 to +100 range to a multiplier
        # Brightness adjustment using additive method for better control
        brightness_adjustment = self.brightness
        
        # Create brightness adjusted frame
        adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_adjustment)
        
        return adjusted
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zoom center adjustment."""
        if event == cv2.EVENT_LBUTTONDOWN:
            height, width = param.shape[:2]
            self.zoom_center_x = x / width
            self.zoom_center_y = y / height
    
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
    
    def setup_zoom_controls(self, window_name):
        """Setup zoom and brightness slider controls."""
        # Create zoom slider (0-100, maps to zoom_min to zoom_max)
        initial_zoom_value = int(((1.0 - self.zoom_min) / (self.zoom_max - self.zoom_min)) * 100)
        cv2.createTrackbar('Zoom', window_name, initial_zoom_value, 100, self.zoom_callback)
        
        # Create brightness slider (0-100, maps to brightness_min to brightness_max)
        # Default brightness is 0, which is at position 50 (middle of 0-100 range)
        initial_brightness_value = int(((0 - self.brightness_min) / (self.brightness_max - self.brightness_min)) * 100)
        cv2.createTrackbar('Brightness', window_name, initial_brightness_value, 100, self.brightness_callback)
        
        # Instructions for controls
        print("\n🔍 Video Controls:")
        print(f"   📏 Zoom range: {self.zoom_min}x to {self.zoom_max}x")
        print(f"   💡 Brightness range: {self.brightness_min} to {self.brightness_max}")
        print("   🎯 Click on video to set zoom center")
        print("   📐 Use sliders to adjust zoom and brightness levels")
    
    def run_detection(self):
        """Run object detection with zoom functionality."""
        print("📹 === Enhanced Object Detection with Zoom ===")
        
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
        
        print("\n🎯 Object Detection with Zoom Active!")
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  s = save screenshot")  
        print("  d = toggle debug info")
        print("  c = toggle confidence display")
        print("  r = reset zoom to 1.0x")
        print("  b = reset brightness to 0")
        print("  Mouse click = set zoom center")
        
        frame_count = 0
        show_confidence = True
        window_name = 'Zoomable Object Detection'
        window_created = False
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Camera disconnected")
                    break
                
                frame_count += 1
                original_frame = frame.copy()
                
                # Create window and setup controls only once, after we have the first frame
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    self.setup_zoom_controls(window_name)
                    cv2.setMouseCallback(window_name, self.mouse_callback, original_frame)
                    window_created = True
                
                # Apply zoom
                zoomed_frame = self.apply_zoom(frame)
                
                # Apply brightness adjustment
                zoomed_frame = self.apply_brightness(zoomed_frame)
                
                height, width = zoomed_frame.shape[:2]
                
                # YOLO detection on zoomed frame
                try:
                    results = self.model(zoomed_frame, verbose=False)
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
                            cv2.rectangle(zoomed_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Add label
                            if show_confidence:
                                label = f"{class_name} ({conf:.2f})"
                            else:
                                label = class_name
                            
                            # Background for text
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(zoomed_frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                            cv2.putText(zoomed_frame, label, (x1+5, y1-8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Status display
                status_y = 50
                cv2.putText(zoomed_frame, "🎯 Zoomable Object Detection", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show detected objects count
                status_y += 40
                if detected_objects:
                    object_text = " | ".join([f"{obj}: {count}" for obj, count in detected_objects.items()])
                    cv2.putText(zoomed_frame, f"Detected: {object_text}", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(zoomed_frame, "No objects detected", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                # Zoom and brightness info
                status_y += 30
                cv2.putText(zoomed_frame, f"Zoom: {self.zoom_factor:.1f}x | Brightness: {self.brightness:+.0f}", 
                           (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                status_y += 20
                cv2.putText(zoomed_frame, f"Center: ({self.zoom_center_x:.2f}, {self.zoom_center_y:.2f})", 
                           (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Frame info
                status_y += 25
                cv2.putText(zoomed_frame, f"Frame: {frame_count} | Camera: {camera_id}", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Debug info
                if self.debug_mode:
                    status_y += 25
                    cv2.putText(zoomed_frame, f"Resolution: {width}x{height} | Debug: ON", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Controls info
                cv2.putText(zoomed_frame, "q=quit | s=save | d=debug | c=confidence | r=reset zoom | b=reset brightness | click=zoom center", 
                           (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow(window_name, zoomed_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"zoomed_detection_{frame_count}_zoom{self.zoom_factor:.1f}x.jpg"
                    cv2.imwrite(filename, zoomed_frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('c'):
                    show_confidence = not show_confidence
                    print(f"📊 Confidence: {'ON' if show_confidence else 'OFF'}")
                elif key == ord('r'):
                    self.zoom_factor = 1.0
                    self.zoom_center_x = 0.5
                    self.zoom_center_y = 0.5
                    # Reset slider to middle position
                    slider_value = int(((1.0 - self.zoom_min) / (self.zoom_max - self.zoom_min)) * 100)
                    cv2.setTrackbarPos('Zoom', window_name, slider_value)
                    print("🔄 Zoom reset to 1.0x, center reset")
                elif key == ord('b'):
                    self.brightness = 0
                    # Reset brightness slider to middle position (0 brightness)
                    brightness_value = int(((0 - self.brightness_min) / (self.brightness_max - self.brightness_min)) * 100)
                    cv2.setTrackbarPos('Brightness', window_name, brightness_value)
                    print("💡 Brightness reset to 0")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n📊 Final Results:")
            print(f"   📹 Camera: {camera_id}")
            print(f"   🎬 Frames processed: {frame_count}")
            print(f"   🔍 Final zoom level: {self.zoom_factor:.1f}x")
            print("   🎯 Zoomable object detection completed")

def main():
    print("📹 === Enhanced Object Detection with Zoom & Brightness Control ===")
    print("YOLO object detection with interactive zoom and brightness functionality")
    print("Features:")
    print("  🔍 Zoom slider (0.5x to 3.0x)")
    print("  💡 Brightness slider (-100 to +100)")
    print("  🎯 Click to set zoom center")
    print("  📐 Real-time zoom and brightness adjustment")
    print("  🎮 Full object detection capabilities")
    print("\n🚀 Starting detection...")
    
    detector = ZoomableObjectDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()