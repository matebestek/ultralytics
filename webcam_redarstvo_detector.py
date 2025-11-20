#!/usr/bin/env python3
"""
Webcam Redarstvo Uniform Detection System
Optimized for external web cameras with enhanced detection algorithms
"""

import cv2
import numpy as np
import json
from ultralytics import YOLO
import time

class WebcamRedarstvoDetector:
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
    
    def test_webcam_connections(self):
        """Test and find available webcam connections."""
        print("🔍 Testing webcam connections...")
        
        available_cameras = []
        
        # Test multiple camera indices (0-10 should cover most setups)
        for camera_id in range(11):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Try to read a frame to verify camera is actually working
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    
                    # Test if we can set higher resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret2, frame2 = cap.read()
                    if ret2 and frame2 is not None:
                        height, width = frame2.shape[:2]
                    
                    camera_info = {
                        'id': camera_id,
                        'resolution': f"{width}x{height}",
                        'working': True
                    }
                    available_cameras.append(camera_info)
                    print(f"   ✅ Camera {camera_id}: {width}x{height}")
                else:
                    print(f"   ❌ Camera {camera_id}: Can't read frames")
                cap.release()
            else:
                print(f"   ❌ Camera {camera_id}: Not available")
            
            time.sleep(0.1)  # Small delay to prevent system overload
        
        return available_cameras
    
    def select_webcam(self, available_cameras):
        """Let user select which webcam to use."""
        if not available_cameras:
            print("❌ No working cameras found!")
            print("\n💡 Webcam Troubleshooting:")
            print("1. Check that webcam is properly connected")
            print("2. Ensure webcam drivers are installed")
            print("3. Close other applications using the camera")
            print("4. Try a different USB port")
            return None
        
        if len(available_cameras) == 1:
            camera = available_cameras[0]
            print(f"✅ Auto-selecting Camera {camera['id']} ({camera['resolution']})")
            return camera['id']
        
        print(f"\n📹 Found {len(available_cameras)} cameras:")
        for i, camera in enumerate(available_cameras):
            print(f"{i+1}. Camera {camera['id']} - {camera['resolution']}")
        
        while True:
            try:
                choice = input(f"\nSelect camera (1-{len(available_cameras)}) or Enter for Camera 0: ").strip()
                
                if choice == "":
                    return 0  # Default to camera 0
                
                choice_num = int(choice) - 1
                if 0 <= choice_num < len(available_cameras):
                    selected = available_cameras[choice_num]
                    print(f"✅ Selected Camera {selected['id']} ({selected['resolution']})")
                    return selected['id']
                else:
                    print("❌ Invalid choice, try again")
            except ValueError:
                print("❌ Invalid input, try again")
            except KeyboardInterrupt:
                return None
    
    def setup_webcam(self, camera_id):
        """Initialize webcam with optimal settings."""
        print(f"📹 Setting up webcam {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return None
        
        # Try to set optimal resolution and frame rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Set additional webcam properties for better quality
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
        
        # Test the setup
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✅ Webcam ready: {width}x{height}")
            return cap
        else:
            print("❌ Failed to read from webcam")
            cap.release()
            return None
    
    def classify_uniform_webcam_optimized(self, person_region):
        """Webcam-optimized uniform classification."""
        if person_region.size == 0:
            return "No Match", 0.0
        
        h, w = person_region.shape[:2]
        
        # Focus on torso region for uniform detection
        if h > 80 and w > 50:
            # Adjust torso region for webcam perspective
            torso_region = person_region[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        else:
            torso_region = person_region
        
        if torso_region.size == 0:
            return "No Match", 0.0
        
        # Convert to HSV for reliable color analysis
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        total_pixels = torso_region.shape[0] * torso_region.shape[1]
        
        # Enhanced color analysis for webcam quality
        # Dark uniform elements (typical security/police uniforms)
        dark_mask = hsv[:, :, 2] < 70  # Slightly higher threshold for webcam
        very_dark_mask = hsv[:, :, 2] < 50
        dark_ratio = np.sum(dark_mask) / total_pixels
        very_dark_ratio = np.sum(very_dark_mask) / total_pixels
        
        # Blue uniform elements (common in Redarstvo uniforms)
        blue_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] > 35)
        blue_ratio = np.sum(blue_mask) / total_pixels
        
        # Navy/dark blue specific detection
        navy_mask = (hsv[:, :, 0] >= 105) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 25) & (hsv[:, :, 2] < 120)
        navy_ratio = np.sum(navy_mask) / total_pixels
        
        # Gray/neutral tones (uniform fabric, badges)
        gray_mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 200)
        gray_ratio = np.sum(gray_mask) / total_pixels
        
        # Light elements (badges, name tags, reflective elements)
        light_mask = hsv[:, :, 2] > 200
        bright_mask = hsv[:, :, 2] > 230
        light_ratio = np.sum(light_mask) / total_pixels
        bright_ratio = np.sum(bright_mask) / total_pixels
        
        # Calculate Redarstvo uniform probability
        score = 0.0
        
        # Primary dark uniform base
        if 0.2 <= dark_ratio <= 0.8:  # Good amount of dark clothing
            score += 0.3
        
        if very_dark_ratio >= 0.1:  # Strong dark elements
            score += 0.2
        
        # Blue uniform components
        if blue_ratio > 0.01:  # At least 1% blue elements
            score += 0.25
        
        if navy_ratio > 0.008:  # Navy blue specifics
            score += 0.15
        
        # Professional uniform characteristics
        if 0.1 <= gray_ratio <= 0.7:  # Good neutral tones
            score += 0.15
        
        # Badge/insignia detection
        if 0.015 <= light_ratio <= 0.3:  # Light elements present
            score += 0.1
        
        if bright_ratio > 0.003:  # Small bright elements
            score += 0.05
        
        # Uniform color distribution bonus
        total_uniform_colors = dark_ratio + blue_ratio + navy_ratio + gray_ratio
        if 0.35 <= total_uniform_colors <= 0.85:
            score += 0.1
        
        # Color contrast (uniforms have contrasting elements)
        if abs(light_ratio - dark_ratio) > 0.15:
            score += 0.05
        
        # Debug output for webcam
        if self.debug_mode:
            print(f"Webcam Debug [{h}x{w}]: Dark={dark_ratio:.3f}, Blue={blue_ratio:.3f}, Navy={navy_ratio:.3f}, Score={score:.3f}")
        
        return "Redarstvo_Uniform" if score > 0.4 else "No Match", score
    
    def run_webcam_detection(self):
        """Run webcam detection with enhanced algorithms."""
        print("📹 === Webcam Redarstvo Uniform Detection System ===")
        
        if not self.load_uniform_profiles():
            return
        
        # Find and setup webcam
        available_cameras = self.test_webcam_connections()
        camera_id = self.select_webcam(available_cameras)
        
        if camera_id is None:
            return
        
        cap = self.setup_webcam(camera_id)
        if cap is None:
            return
        
        print(f"\n🎯 Webcam Detection Active!")
        print(f"📹 Camera: {camera_id}")
        print(f"📊 Uniform profiles: {len(self.uniform_profiles)}")
        print("\n🎮 Controls:")
        print("  q = quit detection")
        print("  s = save screenshot")
        print("  d = toggle debug mode")
        print("  r = reset detection counter")
        print("  + = increase detection sensitivity")
        print("  - = decrease detection sensitivity")
        
        frame_count = 0
        detection_count = 0
        detection_sensitivity = 0.4  # Adjustable threshold
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Failed to read from webcam")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                original_height, original_width = frame.shape[:2]
                
                # Run YOLO detection
                try:
                    results = self.model(frame, verbose=False)
                    detections = results[0].boxes
                except Exception as e:
                    if self.debug_mode:
                        print(f"Detection error: {e}")
                    continue
                
                current_detections = 0
                person_count = 0
                
                if detections is not None:
                    for box in detections.data:
                        x1, y1, x2, y2, conf, class_id = box
                        class_name = self.model.names[int(class_id)]
                        
                        if class_name == "person" and conf > 0.3:  # Lower threshold for webcam
                            person_count += 1
                            
                            # Extract and validate person region
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:
                                person_region = frame[y1:y2, x1:x2]
                                
                                # Classify uniform
                                uniform_type, uniform_confidence = self.classify_uniform_webcam_optimized(person_region)
                                
                                # Check if Redarstvo uniform detected
                                is_redarstvo = uniform_confidence > detection_sensitivity and uniform_type != "No Match"
                                
                                if is_redarstvo:
                                    color = (0, 0, 255)  # Red for uniform detection
                                    label = f"🚨 REDARSTVO ({uniform_confidence:.2f})"
                                    current_detections += 1
                                    detection_count += 1
                                    thickness = 3
                                else:
                                    color = (0, 255, 0)  # Green for regular person
                                    label = f"Person ({conf:.2f})"
                                    thickness = 2
                                
                                # Draw detection box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Add label with background
                                font_scale = max(0.5, min(1.0, original_width / 1000))
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                                cv2.rectangle(frame, (x1, y1 - 35), 
                                            (x1 + label_size[0] + 10, y1), color, -1)
                                cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                                
                                # Debug information
                                if self.debug_mode and uniform_confidence > 0.1:
                                    debug_text = f"Size: {x2-x1}x{y2-y1} | Conf: {uniform_confidence:.3f}"
                                    cv2.putText(frame, debug_text, (x1, y2 + 20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Status overlay
                font_scale = max(0.6, min(1.2, original_width / 800))
                
                if current_detections > 0:
                    status_color = (0, 0, 255)
                    status_text = f"🚨 REDARSTVO DETECTED: {current_detections}"
                else:
                    status_color = (0, 255, 0)
                    status_text = "🟢 Monitoring..."
                
                cv2.putText(frame, status_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 2)
                
                cv2.putText(frame, f"📹 People: {person_count} | Total Detections: {detection_count}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Frame: {frame_count} | Sensitivity: {detection_sensitivity:.2f} | Debug: {'ON' if self.debug_mode else 'OFF'}", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), 1)
                
                cv2.putText(frame, "q=quit | s=save | d=debug | +/- sensitivity", 
                           (20, original_height - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Webcam Redarstvo Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"webcam_detection_{frame_count}_{detection_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('r'):
                    detection_count = 0
                    print("🔄 Detection counter reset")
                elif key == ord('+') or key == ord('='):
                    detection_sensitivity = min(0.8, detection_sensitivity + 0.05)
                    print(f"📈 Sensitivity increased to {detection_sensitivity:.2f}")
                elif key == ord('-'):
                    detection_sensitivity = max(0.2, detection_sensitivity - 0.05)
                    print(f"📉 Sensitivity decreased to {detection_sensitivity:.2f}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Detection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final summary
            print(f"\n📊 Webcam Detection Summary:")
            print(f"   📹 Camera: {camera_id}")
            print(f"   🎬 Total frames: {frame_count}")
            print(f"   🚨 Redarstvo detections: {detection_count}")
            print(f"   📏 Resolution: {original_width}x{original_height}")
            print(f"   🎯 Final sensitivity: {detection_sensitivity:.2f}")
            print("Detection stopped")

def main():
    detector = WebcamRedarstvoDetector()
    print("📹 === Webcam Redarstvo Detection System ===")
    print("Optimized for external web cameras")
    print("Enhanced uniform detection with adjustable sensitivity")
    print("\n🚀 Starting webcam detection...")
    
    detector.run_webcam_detection()

if __name__ == "__main__":
    main()