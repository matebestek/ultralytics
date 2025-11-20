#!/usr/bin/env python3
"""
Enhanced Object Detection System with Logo Recognition
YOLO-based object detection with zoom, brightness, and specific logo detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

class LogoEnhancedDetector:
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
        
        # Logo detection parameters
        self.logo_template = None
        self.logo_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Multiple scales
        self.logo_threshold = 0.6  # Confidence threshold for logo detection
        self.logo_detections = []  # Store logo detections for confirmation
        self.confirmation_window_open = False
        
        # Load logo template
        self.load_logo_template()
        
    def load_logo_template(self):
        """Load and preprocess the logo template."""
        logo_path = "javna_parkirisca_uniforma_logo.jpeg"
        
        if not os.path.exists(logo_path):
            print(f"❌ Logo template not found: {logo_path}")
            return False
            
        try:
            # Load logo image
            self.logo_template = cv2.imread(logo_path)
            if self.logo_template is None:
                print(f"❌ Failed to load logo image: {logo_path}")
                return False
                
            # Convert to grayscale for better template matching
            self.logo_template_gray = cv2.cvtColor(self.logo_template, cv2.COLOR_BGR2GRAY)
            
            # Get template dimensions
            self.logo_h, self.logo_w = self.logo_template_gray.shape
            
            print(f"✅ Logo template loaded: {self.logo_w}x{self.logo_h} pixels")
            print(f"📋 Logo scales: {len(self.logo_scales)} different sizes")
            print(f"🎯 Detection threshold: {self.logo_threshold}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading logo template: {e}")
            return False
    
    def detect_logo_in_person(self, frame, person_bbox):
        """Detect logo within a person's bounding box using template matching."""
        if self.logo_template is None:
            return []
            
        x1, y1, x2, y2 = person_bbox
        
        # Extract person region with some padding
        padding = 10
        person_x1 = max(0, x1 - padding)
        person_y1 = max(0, y1 - padding)
        person_x2 = min(frame.shape[1], x2 + padding)
        person_y2 = min(frame.shape[0], y2 + padding)
        
        person_roi = frame[person_y1:person_y2, person_x1:person_x2]
        
        if person_roi.size == 0:
            return []
            
        # Convert person ROI to grayscale
        person_gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        logo_detections = []
        
        # Try different scales of the logo template
        for scale in self.logo_scales:
            # Resize template
            scaled_w = int(self.logo_w * scale)
            scaled_h = int(self.logo_h * scale)
            
            # Skip if template is larger than the person ROI
            if scaled_w > person_gray.shape[1] or scaled_h > person_gray.shape[0]:
                continue
                
            scaled_template = cv2.resize(self.logo_template_gray, (scaled_w, scaled_h))
            
            # Template matching
            result = cv2.matchTemplate(person_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where matching exceeds threshold
            locations = np.where(result >= self.logo_threshold)
            
            for pt in zip(*locations[::-1]):  # Switch x and y
                # Calculate confidence
                confidence = result[pt[1], pt[0]]
                
                # Convert coordinates back to full frame
                logo_x1 = person_x1 + pt[0]
                logo_y1 = person_y1 + pt[1]
                logo_x2 = logo_x1 + scaled_w
                logo_y2 = logo_y1 + scaled_h
                
                logo_detections.append({
                    'bbox': (logo_x1, logo_y1, logo_x2, logo_y2),
                    'confidence': confidence,
                    'scale': scale,
                    'person_bbox': (x1, y1, x2, y2)
                })
        
        # Non-maximum suppression to remove overlapping detections
        if logo_detections:
            logo_detections = self.non_max_suppression_logos(logo_detections)
            
        return logo_detections
    
    def non_max_suppression_logos(self, detections, overlap_threshold=0.3):
        """Apply non-maximum suppression to logo detections."""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        for i, det in enumerate(detections):
            bbox1 = det['bbox']
            
            # Check if this detection overlaps too much with any kept detection
            should_keep = True
            for kept_det in keep:
                bbox2 = kept_det['bbox']
                
                # Calculate IoU (Intersection over Union)
                iou = self.calculate_iou(bbox1, bbox2)
                
                if iou > overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det)
        
        return keep
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def show_logo_confirmation(self, frame, logo_detection):
        """Show confirmation dialog for detected logo."""
        # Extract logo region from frame
        bbox = logo_detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Add padding for better visualization
        padding = 20
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(frame.shape[1], x2 + padding)
        y2_padded = min(frame.shape[0], y2 + padding)
        
        logo_region = frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()
        
        if logo_region.size == 0:
            return False
        
        # Draw bounding box around detected logo
        cv2.rectangle(logo_region, 
                     (x1 - x1_padded, y1 - y1_padded), 
                     (x2 - x1_padded, y2 - y1_padded), 
                     (0, 255, 0), 3)
        
        # Add confidence text
        conf_text = f"Logo Confidence: {logo_detection['confidence']:.2f}"
        cv2.putText(logo_region, conf_text, (5, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize for better visibility if too small
        if logo_region.shape[0] < 150 or logo_region.shape[1] < 150:
            scale_factor = max(150 / logo_region.shape[0], 150 / logo_region.shape[1])
            new_width = int(logo_region.shape[1] * scale_factor)
            new_height = int(logo_region.shape[0] * scale_factor)
            logo_region = cv2.resize(logo_region, (new_width, new_height), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Show confirmation window
        window_name = f"Logo Detection - Confidence: {logo_detection['confidence']:.2f}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, logo_region)
        
        print(f"\n🔍 Logo detected with confidence: {logo_detection['confidence']:.2f}")
        print("📝 Please confirm if this is the correct logo:")
        print("   ✅ Press 'y' to confirm (correct logo)")
        print("   ❌ Press 'n' to reject (false positive)")
        print("   ⏭️  Press any other key to skip")
        
        # Wait for user input
        self.confirmation_window_open = True
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y') or key == ord('Y'):
                print("✅ Logo detection confirmed!")
                cv2.destroyWindow(window_name)
                self.confirmation_window_open = False
                return True
            elif key == ord('n') or key == ord('N'):
                print("❌ Logo detection rejected")
                cv2.destroyWindow(window_name)
                self.confirmation_window_open = False
                return False
            elif key != 255:  # Any other key
                print("⏭️ Logo detection skipped")
                cv2.destroyWindow(window_name)
                self.confirmation_window_open = False
                return False
    
    def zoom_callback(self, val):
        """Callback function for zoom slider."""
        self.zoom_factor = self.zoom_min + (val / 100.0) * (self.zoom_max - self.zoom_min)
    
    def brightness_callback(self, val):
        """Callback function for brightness slider."""
        self.brightness = self.brightness_min + (val / 100.0) * (self.brightness_max - self.brightness_min)
    
    def apply_zoom(self, frame):
        """Apply zoom to the frame."""
        if abs(self.zoom_factor - 1.0) < 0.01:  # Use small tolerance for float comparison
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
        if event == cv2.EVENT_LBUTTONDOWN and not self.confirmation_window_open:
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
    
    def setup_controls(self, window_name):
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
        print("   🏷️  Logo detection active with confirmation dialogs")
    
    def run_detection(self):
        """Run object detection with zoom, brightness, and logo detection functionality."""
        print("📹 === Enhanced Object Detection with Logo Recognition ===")
        
        # Check if logo template is loaded
        if self.logo_template is None:
            print("\n⚠️ Logo template not available - running without logo detection")
        
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
        
        print("\n🎯 Object Detection with Logo Recognition Active!")
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  s = save screenshot")  
        print("  d = toggle debug info")
        print("  c = toggle confidence display")
        print("  r = reset zoom to 1.0x")
        print("  b = reset brightness to 0")
        print("  l = toggle logo detection")
        print("  Mouse click = set zoom center")
        
        frame_count = 0
        show_confidence = True
        logo_detection_enabled = True
        window_name = 'Logo Enhanced Object Detection'
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
                    self.setup_controls(window_name)
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
                person_bboxes = []
                logo_detections_current = []
                
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
                            
                            # Store person bounding boxes for logo detection
                            if class_name == 'person':
                                person_bboxes.append((x1, y1, x2, y2))
                            
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
                
                # Logo detection within person bounding boxes
                if logo_detection_enabled and self.logo_template is not None and person_bboxes:
                    for person_bbox in person_bboxes:
                        logos_in_person = self.detect_logo_in_person(zoomed_frame, person_bbox)
                        
                        for logo_det in logos_in_person:
                            # Draw logo detection
                            lx1, ly1, lx2, ly2 = logo_det['bbox']
                            cv2.rectangle(zoomed_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)  # Red for logo
                            
                            # Add logo label
                            logo_label = f"LOGO ({logo_det['confidence']:.2f})"
                            cv2.putText(zoomed_frame, logo_label, (lx1, ly1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Add to detected objects
                            if 'person_with_logo' in detected_objects:
                                detected_objects['person_with_logo'] += 1
                            else:
                                detected_objects['person_with_logo'] = 1
                            
                            logo_detections_current.append(logo_det)
                
                # Status display
                status_y = 50
                cv2.putText(zoomed_frame, "🎯 Logo Enhanced Detection", (20, status_y), 
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
                
                # Logo detection status
                status_y += 25
                logo_status = "ON" if logo_detection_enabled else "OFF"
                logo_color = (0, 255, 0) if logo_detection_enabled else (0, 0, 255)
                cv2.putText(zoomed_frame, f"Logo Detection: {logo_status}", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, logo_color, 2)
                
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
                cv2.putText(zoomed_frame, "q=quit | s=save | d=debug | c=confidence | r=reset zoom | b=reset brightness | l=logo toggle", 
                           (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow(window_name, zoomed_frame)
                
                # Show logo confirmation dialogs
                if logo_detections_current and not self.confirmation_window_open:
                    for logo_det in logo_detections_current:
                        confirmed = self.show_logo_confirmation(zoomed_frame, logo_det)
                        if confirmed:
                            print(f"🎯 Logo confirmed at frame {frame_count}")
                        break  # Only show one confirmation at a time
                
                # Handle keys (only if no confirmation window is open)
                if not self.confirmation_window_open:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"logo_detection_{frame_count}_zoom{self.zoom_factor:.1f}x.jpg"
                        cv2.imwrite(filename, zoomed_frame)
                        print(f"📸 Saved: {filename}")
                    elif key == ord('d'):
                        self.debug_mode = not self.debug_mode
                        print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
                    elif key == ord('c'):
                        show_confidence = not show_confidence
                        print(f"📊 Confidence: {'ON' if show_confidence else 'OFF'}")
                    elif key == ord('l'):
                        logo_detection_enabled = not logo_detection_enabled
                        print(f"🏷️ Logo Detection: {'ON' if logo_detection_enabled else 'OFF'}")
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
            print("   🎯 Logo enhanced object detection completed")

def main():
    print("📹 === Enhanced Object Detection with Logo Recognition ===")
    print("YOLO object detection with zoom, brightness, and specific logo detection")
    print("Features:")
    print("  🔍 Zoom slider (0.5x to 3.0x)")
    print("  💡 Brightness slider (-100 to +100)")
    print("  🏷️ Logo detection with confirmation dialogs")
    print("  🎯 Click to set zoom center")
    print("  📐 Real-time zoom and brightness adjustment")
    print("  🎮 Full object detection capabilities")
    print("\n🚀 Starting detection...")
    
    detector = LogoEnhancedDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()