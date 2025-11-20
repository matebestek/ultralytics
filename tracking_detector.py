#!/usr/bin/env python3
"""
Enhanced Object Detection System with Object Tracking
YOLO-based object detection with zoom, brightness, and multi-object tracking in separate windows
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import math

class TrackingEnhancedDetector:
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
        
        # Tracking parameters
        self.tracking_enabled = True
        self.next_object_id = 1
        self.tracked_objects = {}  # {object_id: {bbox, class_name, confidence, trail, tracker}}
        self.tracking_threshold = 100  # Maximum distance for object matching
        self.max_trail_length = 30
        self.tracker_type = 'CSRT'  # Options: 'CSRT', 'KCF', 'BOOSTING', 'MIL', 'TLD', 'MEDIANFLOW'
        
        # Separate tracking windows
        self.tracking_windows = {}  # {object_id: window_info}
        self.window_size = (400, 300)
        self.tracking_zoom_factor = 2.0
        
        # Colors for different object IDs
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        
    def get_color_for_id(self, object_id):
        """Get consistent color for object ID."""
        return self.colors[object_id % len(self.colors)]
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding box centers."""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        return math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    
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
    
    def create_tracker(self):
        """Create a new tracker instance."""
        if self.tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif self.tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracker_type == 'MIL':
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif self.tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        else:
            return cv2.TrackerCSRT_create()
    
    def update_tracking(self, frame, detections):
        """Update object tracking with new detections."""
        if not self.tracking_enabled:
            return
        
        current_detections = []
        
        # Process YOLO detections
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                
                if conf > 0.5:  # Higher threshold for tracking
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    class_name = self.model.names[int(class_id)]
                    
                    current_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_name': class_name,
                        'confidence': float(conf)
                    })
        
        # Update existing trackers
        to_remove = []
        for obj_id, obj_info in self.tracked_objects.items():
            if 'tracker' in obj_info:
                success, bbox = obj_info['tracker'].update(frame)
                
                if success:
                    # Convert bbox to integer coordinates
                    x, y, w, h = [int(v) for v in bbox]
                    new_bbox = (x, y, x + w, y + h)
                    
                    # Update object info
                    obj_info['bbox'] = new_bbox
                    obj_info['trail'].append(((x + w//2), (y + h//2)))
                    
                    # Limit trail length
                    if len(obj_info['trail']) > self.max_trail_length:
                        obj_info['trail'].popleft()
                else:
                    # Tracker failed, mark for removal
                    to_remove.append(obj_id)
        
        # Remove failed trackers
        for obj_id in to_remove:
            if obj_id in self.tracked_objects:
                del self.tracked_objects[obj_id]
            if obj_id in self.tracking_windows:
                cv2.destroyWindow(f"Tracker_{obj_id}")
                del self.tracking_windows[obj_id]
        
        # Match new detections with existing tracks
        matched_detections = set()
        
        for detection in current_detections:
            det_bbox = detection['bbox']
            best_match_id = None
            best_score = float('inf')
            
            # Find best matching existing object
            for obj_id, obj_info in self.tracked_objects.items():
                if obj_info['class_name'] == detection['class_name']:
                    distance = self.calculate_distance(det_bbox, obj_info['bbox'])
                    iou = self.calculate_iou(det_bbox, obj_info['bbox'])
                    
                    # Combined score (lower is better)
                    score = distance - (iou * 100)  # IoU helps reduce score
                    
                    if score < best_score and distance < self.tracking_threshold:
                        best_score = score
                        best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing track
                matched_detections.add(best_match_id)
                obj_info = self.tracked_objects[best_match_id]
                obj_info['bbox'] = det_bbox
                obj_info['confidence'] = detection['confidence']
                
                # Reinitialize tracker with new bbox
                x1, y1, x2, y2 = det_bbox
                tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = self.create_tracker()
                tracker.init(frame, tracker_bbox)
                obj_info['tracker'] = tracker
            else:
                # Create new track
                obj_id = self.next_object_id
                self.next_object_id += 1
                
                x1, y1, x2, y2 = det_bbox
                tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = self.create_tracker()
                tracker.init(frame, tracker_bbox)
                
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2
                
                self.tracked_objects[obj_id] = {
                    'bbox': det_bbox,
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'trail': deque([(center_x, center_y)], maxlen=self.max_trail_length),
                    'tracker': tracker,
                    'frames_tracked': 0
                }
                
                matched_detections.add(obj_id)
        
        # Update frame counts
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['frames_tracked'] += 1
    
    def create_tracking_window(self, frame, obj_id, obj_info):
        """Create or update tracking window for specific object."""
        window_name = f"Tracker_{obj_id}"
        
        # Get object bounding box
        x1, y1, x2, y2 = obj_info['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Calculate tracking window size
        obj_width = x2 - x1
        obj_height = y2 - y1
        
        # Expand the region around the object
        expand_factor = self.tracking_zoom_factor
        track_width = int(obj_width * expand_factor)
        track_height = int(obj_height * expand_factor)
        
        # Calculate tracking region bounds
        track_x1 = max(0, center_x - track_width // 2)
        track_y1 = max(0, center_y - track_height // 2)
        track_x2 = min(frame.shape[1], center_x + track_width // 2)
        track_y2 = min(frame.shape[0], center_y + track_height // 2)
        
        # Extract tracking region
        track_region = frame[track_y1:track_y2, track_x1:track_x2].copy()
        
        if track_region.size == 0:
            return
        
        # Adjust bounding box coordinates to tracking region
        adj_x1 = x1 - track_x1
        adj_y1 = y1 - track_y1
        adj_x2 = x2 - track_x1
        adj_y2 = y2 - track_y1
        
        # Draw object bounding box
        color = self.get_color_for_id(obj_id)
        cv2.rectangle(track_region, (adj_x1, adj_y1), (adj_x2, adj_y2), color, 3)
        
        # Draw trail in tracking region
        if len(obj_info['trail']) > 1:
            trail_points = []
            for point in obj_info['trail']:
                trail_x = point[0] - track_x1
                trail_y = point[1] - track_y1
                
                # Only add points within tracking region
                if 0 <= trail_x < track_region.shape[1] and 0 <= trail_y < track_region.shape[0]:
                    trail_points.append((trail_x, trail_y))
            
            # Draw trail lines
            for i in range(1, len(trail_points)):
                thickness = max(1, int((i / len(trail_points)) * 5))
                cv2.line(track_region, trail_points[i-1], trail_points[i], color, thickness)
        
        # Add object information
        info_text = f"{obj_info['class_name']} ID:{obj_id}"
        conf_text = f"Conf: {obj_info['confidence']:.2f}"
        frames_text = f"Frames: {obj_info['frames_tracked']}"
        
        cv2.putText(track_region, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(track_region, conf_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(track_region, frames_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Resize for consistent window size
        track_region = cv2.resize(track_region, self.window_size, interpolation=cv2.INTER_LINEAR)
        
        # Create or update window
        if obj_id not in self.tracking_windows:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            self.tracking_windows[obj_id] = {
                'window_name': window_name,
                'last_update': time.time()
            }
        
        cv2.imshow(window_name, track_region)
        self.tracking_windows[obj_id]['last_update'] = time.time()
    
    def cleanup_old_windows(self, max_age=5.0):
        """Clean up tracking windows for objects that are no longer tracked."""
        current_time = time.time()
        to_remove = []
        
        for obj_id, window_info in self.tracking_windows.items():
            if obj_id not in self.tracked_objects:
                age = current_time - window_info['last_update']
                if age > max_age:
                    cv2.destroyWindow(window_info['window_name'])
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracking_windows[obj_id]
    
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
        print(f"   📱 Object tracking with separate windows ({self.tracker_type})")
    
    def run_detection(self):
        """Run object detection with tracking functionality."""
        print("📹 === Enhanced Object Detection with Multi-Object Tracking ===")
        
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
        
        print("\n🎯 Object Detection with Multi-Object Tracking Active!")
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  s = save screenshot")  
        print("  d = toggle debug info")
        print("  c = toggle confidence display")
        print("  t = toggle tracking")
        print("  w = close all tracking windows")
        print("  r = reset zoom to 1.0x")
        print("  b = reset brightness to 0")
        print("  Mouse click = set zoom center")
        
        frame_count = 0
        show_confidence = True
        window_name = 'Multi-Object Tracking Detection'
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
                
                # Update tracking
                self.update_tracking(zoomed_frame, detections)
                
                detected_objects = {}
                
                # Draw tracked objects
                for obj_id, obj_info in self.tracked_objects.items():
                    x1, y1, x2, y2 = obj_info['bbox']
                    class_name = obj_info['class_name']
                    confidence = obj_info['confidence']
                    
                    # Count detected objects
                    if class_name in detected_objects:
                        detected_objects[class_name] += 1
                    else:
                        detected_objects[class_name] = 1
                    
                    # Get color for this object ID
                    color = self.get_color_for_id(obj_id)
                    
                    # Draw bounding box
                    thickness = 3
                    cv2.rectangle(zoomed_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw trail
                    if len(obj_info['trail']) > 1:
                        trail_points = list(obj_info['trail'])
                        for i in range(1, len(trail_points)):
                            thickness_trail = max(1, int((i / len(trail_points)) * 5))
                            cv2.line(zoomed_frame, trail_points[i-1], trail_points[i], color, thickness_trail)
                    
                    # Add label with ID
                    if show_confidence:
                        label = f"{class_name} ID:{obj_id} ({confidence:.2f})"
                    else:
                        label = f"{class_name} ID:{obj_id}"
                    
                    # Background for text
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(zoomed_frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                    cv2.putText(zoomed_frame, label, (x1+5, y1-8), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Create/update tracking window for this object
                    if self.tracking_enabled:
                        self.create_tracking_window(zoomed_frame, obj_id, obj_info)
                
                # Clean up old tracking windows
                self.cleanup_old_windows()
                
                # Status display
                status_y = 50
                cv2.putText(zoomed_frame, "🎯 Multi-Object Tracking Detection", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show detected objects count
                status_y += 40
                if detected_objects:
                    object_text = " | ".join([f"{obj}: {count}" for obj, count in detected_objects.items()])
                    cv2.putText(zoomed_frame, f"Tracked: {object_text}", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(zoomed_frame, "No objects tracked", (20, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                # Tracking info
                status_y += 30
                active_tracks = len(self.tracked_objects)
                tracking_status = "ON" if self.tracking_enabled else "OFF"
                tracking_color = (0, 255, 0) if self.tracking_enabled else (0, 0, 255)
                cv2.putText(zoomed_frame, f"Tracking: {tracking_status} | Active: {active_tracks} | Algo: {self.tracker_type}", 
                           (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracking_color, 2)
                
                # Zoom and brightness info
                status_y += 25
                cv2.putText(zoomed_frame, f"Zoom: {self.zoom_factor:.1f}x | Brightness: {self.brightness:+.0f}", 
                           (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
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
                cv2.putText(zoomed_frame, "q=quit | s=save | t=tracking | w=close windows | r=reset zoom | b=reset brightness", 
                           (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show main frame
                cv2.imshow(window_name, zoomed_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"tracking_detection_{frame_count}_zoom{self.zoom_factor:.1f}x.jpg"
                    cv2.imwrite(filename, zoomed_frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('c'):
                    show_confidence = not show_confidence
                    print(f"📊 Confidence: {'ON' if show_confidence else 'OFF'}")
                elif key == ord('t'):
                    self.tracking_enabled = not self.tracking_enabled
                    print(f"📱 Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
                    if not self.tracking_enabled:
                        # Clear all tracking
                        self.tracked_objects.clear()
                elif key == ord('w'):
                    # Close all tracking windows
                    for obj_id, window_info in self.tracking_windows.items():
                        cv2.destroyWindow(window_info['window_name'])
                    self.tracking_windows.clear()
                    print("🪟 All tracking windows closed")
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
            print(f"   📱 Objects tracked: {len(self.tracked_objects)}")
            print("   🎯 Multi-object tracking detection completed")

def main():
    print("📹 === Enhanced Object Detection with Multi-Object Tracking ===")
    print("YOLO object detection with zoom, brightness, and advanced tracking")
    print("Features:")
    print("  🔍 Zoom slider (0.5x to 3.0x)")
    print("  💡 Brightness slider (-100 to +100)")
    print("  📱 Multi-object tracking with separate windows")
    print("  🎨 Object trails and persistent IDs")
    print("  🎯 Click to set zoom center")
    print("  📐 Real-time zoom and brightness adjustment")
    print("  🎮 Full object detection capabilities")
    print("\n🚀 Starting detection...")
    
    detector = TrackingEnhancedDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()