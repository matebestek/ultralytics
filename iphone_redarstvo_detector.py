#!/usr/bin/env python3
"""
Redarstvo Uniform Detection for iPhone Camera
Supports both USB connection and WiFi IP camera apps
"""

import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import requests
from urllib.parse import urlparse

class iPhoneRedarstvoDetector:
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
    
    def test_iphone_connection_methods(self):
        """Test different ways to connect to iPhone camera."""
        print("🔍 Testing iPhone camera connection methods...")
        
        connection_methods = []
        
        # Method 1: USB connection (iPhone as webcam via Continuity Camera)
        print("1. Testing USB/Continuity Camera connection...")
        for camera_id in range(10):  # Test more camera indices for iPhone
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ Found camera at index {camera_id} - Resolution: {frame.shape}")
                    connection_methods.append(('usb', camera_id, f"USB Camera {camera_id}"))
                cap.release()
            time.sleep(0.2)
        
        # Method 2: IP Camera apps (user will need to provide IP)
        print("\n2. IP Camera setup available (requires IP Webcam app)")
        connection_methods.append(('ip', None, "IP Camera (manual setup)"))
        
        return connection_methods
    
    def connect_to_iphone_camera(self):
        """Connect to iPhone camera using best available method."""
        methods = self.test_iphone_connection_methods()
        
        if not methods:
            print("❌ No iPhone camera connections found.")
            print("\n📱 iPhone Camera Setup Options:")
            print("1. Use iPhone as webcam (iOS 16+):")
            print("   - Connect iPhone via USB")
            print("   - Enable 'iPhone as webcam' in macOS Ventura+ or use third-party software on Windows")
            print("\n2. Use IP Webcam app:")
            print("   - Install 'IP Webcam' app from App Store")
            print("   - Start the app and note the IP address")
            print("   - Run this script again")
            return None, None
        
        print(f"\n📱 Found {len(methods)} connection method(s):")
        for i, (method_type, camera_id, description) in enumerate(methods):
            print(f"{i+1}. {description}")
        
        # Auto-select first USB method if available
        usb_methods = [m for m in methods if m[0] == 'usb']
        if usb_methods:
            method_type, camera_id, description = usb_methods[0]
            print(f"✅ Auto-selecting: {description}")
            
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher res for iPhone
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            return cap, f"iPhone USB Camera {camera_id}"
        
        # If no USB, offer IP camera setup
        print("\n💡 No USB camera found. Let's set up IP camera:")
        ip_address = input("Enter your iPhone's IP Webcam address (e.g., http://192.168.1.100:8080): ")
        
        if ip_address:
            # Try to connect to IP camera
            try:
                if not ip_address.startswith('http'):
                    ip_address = f"http://{ip_address}"
                
                video_url = f"{ip_address}/video"
                cap = cv2.VideoCapture(video_url)
                
                if cap.isOpened():
                    return cap, f"iPhone IP Camera ({ip_address})"
                else:
                    print("❌ Could not connect to IP camera")
            except Exception as e:
                print(f"❌ IP camera connection error: {e}")
        
        return None, None
    
    def classify_uniform_iphone_optimized(self, person_region):
        """iPhone-optimized uniform classification with better image quality handling."""
        if person_region.size == 0:
            return "No Match", 0.0
        
        # iPhone cameras have better quality, so we can be more detailed
        h, w = person_region.shape[:2]
        
        # Use larger torso region for better accuracy with iPhone quality
        if h > 100 and w > 60:
            torso_region = person_region[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]
        else:
            torso_region = person_region
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # More detailed analysis for higher quality iPhone images
        total_pixels = torso_region.shape[0] * torso_region.shape[1]
        
        # Enhanced color detection for iPhone quality
        # Dark uniform base (typical security/police uniforms)
        dark_mask = hsv[:, :, 2] < 60
        very_dark_mask = hsv[:, :, 2] < 40
        dark_ratio = np.sum(dark_mask) / total_pixels
        very_dark_ratio = np.sum(very_dark_mask) / total_pixels
        
        # Blue elements (common in many uniforms, including Redarstvo)
        blue_mask = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 40)
        blue_ratio = np.sum(blue_mask) / total_pixels
        
        # Navy/dark blue (specific to many security uniforms)
        navy_mask = (hsv[:, :, 0] >= 110) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] < 100)
        navy_ratio = np.sum(navy_mask) / total_pixels
        
        # Gray/neutral elements
        gray_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 180)
        gray_ratio = np.sum(gray_mask) / total_pixels
        
        # Light elements (badges, reflective strips, name tags)
        light_mask = hsv[:, :, 2] > 210
        bright_mask = hsv[:, :, 2] > 240
        light_ratio = np.sum(light_mask) / total_pixels
        bright_ratio = np.sum(bright_mask) / total_pixels
        
        # Calculate Redarstvo uniform probability
        score = 0.0
        
        # Primary uniform indicators
        if 0.25 <= dark_ratio <= 0.85:  # Significant dark clothing
            score += 0.25
        
        if very_dark_ratio >= 0.15:  # Very dark elements (typical uniform)
            score += 0.15
        
        # Blue uniform elements
        if blue_ratio > 0.015:  # At least 1.5% blue elements
            score += 0.2
        
        if navy_ratio > 0.01:  # Navy blue specific elements
            score += 0.15
        
        # Professional uniform characteristics
        if 0.15 <= gray_ratio <= 0.6:  # Good amount of gray/neutral
            score += 0.15
        
        # Badge/insignia indicators
        if 0.02 <= light_ratio <= 0.25:  # Light elements present but not overwhelming
            score += 0.1
        
        if bright_ratio > 0.005:  # Small bright elements (badges, reflective strips)
            score += 0.1
        
        # Bonus scoring for uniform-like color distribution
        uniform_colors = dark_ratio + blue_ratio + navy_ratio + gray_ratio
        if 0.4 <= uniform_colors <= 0.9:  # Good mix of uniform colors
            score += 0.1
        
        # Color contrast bonus (uniforms often have contrasting elements)
        contrast_score = abs(light_ratio - dark_ratio)
        if 0.2 <= contrast_score <= 0.7:
            score += 0.05
        
        # Debug information for iPhone detection
        debug_info = {
            'dark_ratio': dark_ratio,
            'very_dark_ratio': very_dark_ratio,
            'blue_ratio': blue_ratio,
            'navy_ratio': navy_ratio,
            'gray_ratio': gray_ratio,
            'light_ratio': light_ratio,
            'bright_ratio': bright_ratio,
            'total_score': score,
            'image_quality': f"{h}x{w}"
        }
        
        if self.debug_mode:
            print(f"iPhone Debug [{h}x{w}]: Dark={dark_ratio:.3f}, Blue={blue_ratio:.3f}, Navy={navy_ratio:.3f}, Score={score:.3f}")
        
        return "Redarstvo_Uniform" if score > 0.35 else "No Match", score  # Slightly lower threshold for iPhone
    
    def run_iphone_detection(self):
        """Run detection optimized for iPhone camera."""
        if not self.load_uniform_profiles():
            return
        
        print("📱 Setting up iPhone camera connection...")
        cap, camera_description = self.connect_to_iphone_camera()
        
        if cap is None:
            return
        
        print(f"\n🎯 iPhone Redarstvo Detection Active!")
        print(f"📱 Camera: {camera_description}")
        print(f"📊 Loaded {len(self.uniform_profiles)} uniform profiles")
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  s = save high-res screenshot")
        print("  d = toggle debug mode")
        print("  r = reset detection")
        print("  f = toggle full screen")
        
        frame_count = 0
        detection_history = []
        fullscreen = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame from iPhone, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # iPhone cameras often have higher resolution, so we can work with that
            original_height, original_width = frame.shape[:2]
            
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
                    
                    if class_name == "person" and conf > 0.4:  # Lower threshold for iPhone quality
                        person_count += 1
                        
                        # Extract person region with safety checks
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            person_region = frame[y1:y2, x1:x2]
                            
                            # Classify uniform with iPhone optimization
                            uniform_type, uniform_confidence = self.classify_uniform_iphone_optimized(person_region)
                            
                            # Determine if Redarstvo uniform detected
                            is_redarstvo = uniform_confidence > 0.35 and uniform_type != "No Match"
                            
                            if is_redarstvo:
                                color = (0, 0, 255)  # Red
                                label = f"🚨 REDARSTVO UNIFORM ({uniform_confidence:.2f})"
                                redarstvo_count += 1
                                thickness = 4
                                
                                # Add to detection history
                                detection_history.append(frame_count)
                                if len(detection_history) > 200:  # Keep more history for iPhone
                                    detection_history.pop(0)
                                    
                            else:
                                color = (0, 255, 0)  # Green
                                label = f"Person ({conf:.2f})"
                                thickness = 2
                            
                            # Draw detection box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Add label with background (scaled for iPhone resolution)
                            font_scale = max(0.6, min(1.2, original_width / 1000))  # Scale font with resolution
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - 40), 
                                        (x1 + label_size[0] + 15, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 8, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                            
                            # Debug info if enabled
                            if self.debug_mode and uniform_confidence > 0.1:
                                debug_text = f"Conf: {uniform_confidence:.3f} | Region: {x2-x1}x{y2-y1}"
                                cv2.putText(frame, debug_text, (x1, y2 + 25), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Status display (scaled for iPhone)
            font_scale = max(0.7, min(1.4, original_width / 800))
            
            if redarstvo_count > 0:
                status_color = (0, 0, 255)
                status_text = f"🚨 REDARSTVO DETECTED: {redarstvo_count}"
            else:
                status_color = (0, 255, 0)
                status_text = "🟢 No uniforms detected"
            
            cv2.putText(frame, status_text, (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 3)
            
            cv2.putText(frame, f"📱 People: {person_count} | Frame: {frame_count}", 
                       (15, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"📊 Resolution: {original_width}x{original_height} | Debug: {'ON' if self.debug_mode else 'OFF'}", 
                       (15, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, "q=quit | s=save | d=debug | f=fullscreen", 
                       (15, original_height - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (255, 255, 255), 1)
            
            # Display frame
            window_name = 'iPhone Redarstvo Detection'
            if fullscreen:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"iphone_redarstvo_{frame_count}_{original_width}x{original_height}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 High-res screenshot saved: {filename}")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('r'):
                detection_history.clear()
                print("🔄 Detection history reset")
            elif key == ord('f'):
                fullscreen = not fullscreen
                print(f"🖥️ Fullscreen: {'ON' if fullscreen else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        total_detections = len(detection_history)
        print(f"\n📊 iPhone Detection Summary:")
        print(f"   📱 Camera: {camera_description}")
        print(f"   🎬 Total frames: {frame_count}")
        print(f"   🚨 Redarstvo detections: {total_detections}")
        print(f"   📏 Resolution: {original_width}x{original_height}")
        print("Detection stopped")

def main():
    detector = iPhoneRedarstvoDetector()
    print("📱 === iPhone Redarstvo Uniform Detection System ===")
    print("Optimized for iPhone camera quality and resolution")
    print("\n📱 iPhone Setup Instructions:")
    print("1. USB Method (Recommended):")
    print("   - Connect iPhone via USB cable")
    print("   - Enable iPhone as webcam (iOS 16+ feature)")
    print("   - Or use third-party software like Camo, EpocCam")
    print("\n2. WiFi Method:")
    print("   - Install 'IP Webcam' app on iPhone")
    print("   - Connect to same WiFi network")
    print("   - Start app and note the IP address")
    print("\n🚀 Starting detection...")
    
    detector.run_iphone_detection()

if __name__ == "__main__":
    main()