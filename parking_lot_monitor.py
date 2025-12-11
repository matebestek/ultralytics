#!/usr/bin/env python3
"""
Parking Lot Monitoring System
Real-time detection of parking space occupancy with notifications when spaces become empty
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import json
import os
from collections import deque
import math
import threading

class ParkingLotMonitor:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.debug_mode = False
        
        # Brightness control
        self.brightness = 0
        self.brightness_min = -100
        self.brightness_max = 100
        
        # Parking spaces configuration
        self.parking_spaces = []  # List of parking space polygons
        self.space_states = {}  # {space_id: {'occupied': bool, 'last_change': timestamp, 'history': deque}}
        self.setup_mode = True  # Mode for setting up parking spaces
        self.current_polygon = []  # Points for current polygon being drawn
        self.next_space_id = 1  # Track next available space ID
        
        # Detection parameters
        self.occupancy_threshold = 0.15  # Minimum overlap ratio to consider space occupied (lowered for better detection)
        self.state_change_delay = 2.0  # Seconds before confirming state change
        self.history_length = 30  # Frames to keep in history
        self.confidence_threshold = 0.25  # Low confidence threshold to catch parked cars in shadow/angles
        
        # Notification settings
        self.notifications_enabled = True
        self.notification_sound = True
        self.log_file = "parking_log.json"
        self.events_log = []
        
        # Colors
        self.color_empty = (0, 255, 0)  # Green for empty
        self.color_occupied = (0, 0, 255)  # Red for occupied
        self.color_pending = (0, 255, 255)  # Yellow for pending change
        self.color_setup = (255, 255, 255)  # White for setup mode
        
        # Statistics
        self.total_empty_events = 0
        self.total_occupied_events = 0
        
        # Popup notification settings
        self.popup_duration = 5  # seconds
        self.active_popups = []  # Track active popup windows
        
        # My car monitoring
        self.my_car_space_id = None  # ID of parking space with user's car
        self.my_car_monitoring_enabled = False
        self.proximity_threshold = 100  # pixels - distance threshold for proximity alerts
        
        # Car movement tracking - track ALL cars
        self.tracked_cars = {}  # {car_id: {'position': (x,y), 'stationary_frames': count, 'moving': bool}}
        self.stationary_threshold = 15  # Frames car must be still to be considered stationary
        self.movement_threshold = 50  # Pixels movement to consider car is moving (increased to filter jitter)
        self.position_history_size = 5  # Number of frames to average for smoothing
        self.car_id_next = 0  # Counter for assigning car IDs
        self.selected_cars = set()  # Set of car IDs that user wants to track
        self.selection_mode = True  # If True, only notify for selected cars
        self.current_detections = []  # Store current frame detections for click selection
        
        self.person_near_my_car = False
        self.vehicle_near_my_car = False
        
        # Pattern learning and analytics
        self.analytics_file = "parking_analytics.json"
        self.pattern_data = {}  # {space_id: {'occupancy_events': [], 'empty_events': [], 'durations': []}}
        self.load_analytics()
        
        # Load saved configuration if exists
        self.config_file = "parking_config.json"
        self.load_configuration()

        # Camera hot-plug support
        self.available_cameras = []
        self.camera_lock = threading.Lock()
        self.camera_scanner_thread = None
        self.stop_camera_scanner = False

    def camera_scanner_loop(self, interval=5):
        """Background thread that rescans cameras periodically to detect hot-plug events."""
        while not self.stop_camera_scanner:
            try:
                cams = self.find_working_cameras()
                with self.camera_lock:
                    # replace available cameras list
                    self.available_cameras = cams
            except Exception:
                pass
            time.sleep(interval)
    
    def load_analytics(self):
        """Load parking analytics and pattern data."""
        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r') as f:
                    self.pattern_data = json.load(f)
                print(f"\u2705 Loaded analytics for {len(self.pattern_data)} parking spaces")
            except Exception as e:
                print(f"\u26a0\ufe0f Could not load analytics: {e}")
                self.pattern_data = {}
        else:
            self.pattern_data = {}
    
    def save_analytics(self):
        """Save parking analytics and pattern data."""
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.pattern_data, f, indent=2)
        except Exception as e:
            print(f"\u26a0\ufe0f Could not save analytics: {e}")
    
    def record_occupancy_event(self, space_id, event_type, timestamp=None):
        """Record parking space occupancy event for pattern analysis."""
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize space analytics if not exists
        if str(space_id) not in self.pattern_data:
            self.pattern_data[str(space_id)] = {
                'occupancy_events': [],
                'empty_events': [],
                'last_change': None,
                'total_occupied_duration': 0,
                'total_empty_duration': 0,
                'hourly_stats': {}  # Hour of day -> {occupied_count, empty_count}
            }
        
        space_data = self.pattern_data[str(space_id)]
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day_of_week = dt.strftime('%A')
        
        event_record = {
            'timestamp': timestamp,
            'datetime': dt.isoformat(),
            'hour': hour,
            'day_of_week': day_of_week
        }
        
        # Calculate duration since last change
        if space_data['last_change'] is not None:
            duration = timestamp - space_data['last_change']
            event_record['duration'] = duration
            
            # Update total durations
            if event_type == 'occupied':
                space_data['total_empty_duration'] += duration
            else:
                space_data['total_occupied_duration'] += duration
        
        # Record event
        if event_type == 'occupied':
            space_data['occupancy_events'].append(event_record)
        else:
            space_data['empty_events'].append(event_record)
        
        # Update hourly statistics
        if str(hour) not in space_data['hourly_stats']:
            space_data['hourly_stats'][str(hour)] = {'occupied': 0, 'empty': 0}
        space_data['hourly_stats'][str(hour)][event_type] += 1
        
        space_data['last_change'] = timestamp
        
        # Keep only last 1000 events per type
        space_data['occupancy_events'] = space_data['occupancy_events'][-1000:]
        space_data['empty_events'] = space_data['empty_events'][-1000:]
        
        # Save analytics periodically (every 10 events)
        total_events = len(space_data['occupancy_events']) + len(space_data['empty_events'])
        if total_events % 10 == 0:
            self.save_analytics()
    
    def get_pattern_predictions(self, space_id):
        """Get pattern-based predictions for when space is likely to become empty."""
        if str(space_id) not in self.pattern_data:
            return None
        
        space_data = self.pattern_data[str(space_id)]
        now = datetime.now()
        current_hour = now.hour
        
        # Calculate average durations
        occupied_events = space_data['occupancy_events']
        empty_events = space_data['empty_events']
        
        if len(occupied_events) < 5:
            return None  # Not enough data
        
        # Calculate average occupied duration
        durations = [e.get('duration', 0) for e in empty_events if 'duration' in e]
        avg_occupied_duration = sum(durations) / len(durations) if durations else 0
        
        # Get hourly patterns
        hourly_stats = space_data['hourly_stats']
        
        # Find peak empty hours
        empty_hours = []
        for hour_str, stats in hourly_stats.items():
            hour = int(hour_str)
            empty_count = stats['empty']
            if empty_count > 0:
                empty_hours.append((hour, empty_count))
        
        empty_hours.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'avg_occupied_duration_minutes': avg_occupied_duration / 60 if avg_occupied_duration > 0 else None,
            'total_occupied_events': len(occupied_events),
            'total_empty_events': len(empty_events),
            'peak_empty_hours': empty_hours[:3],  # Top 3 hours
            'current_hour_empty_probability': hourly_stats.get(str(current_hour), {}).get('empty', 0) / max(1, len(empty_events)) * 100
        }
    
    def display_analytics_summary(self):
        """Display analytics summary for all parking spaces."""
        print("\n" + "="*60)
        print("📊 PARKING PATTERN ANALYTICS")
        print("="*60)
        
        if not self.pattern_data:
            print("No pattern data available yet.")
            return
        
        for space_id_str, data in self.pattern_data.items():
            space_id = int(space_id_str)
            predictions = self.get_pattern_predictions(space_id)
            
            if predictions is None:
                continue
            
            print(f"\n🅿️ Space #{space_id}:")
            print(f"   Total Events: {predictions['total_occupied_events']} occupied, {predictions['total_empty_events']} empty")
            
            if predictions['avg_occupied_duration_minutes']:
                print(f"   Avg Occupied Duration: {predictions['avg_occupied_duration_minutes']:.1f} minutes")
            
            if predictions['peak_empty_hours']:
                print(f"   Peak Empty Hours: ", end="")
                for hour, count in predictions['peak_empty_hours']:
                    print(f"{hour:02d}:00 ({count}x), ", end="")
                print()
            
            print(f"   Current Hour Empty Probability: {predictions['current_hour_empty_probability']:.1f}%")
        
        print("\n" + "="*60)
    
    def load_configuration(self):
        """Load parking space configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.parking_spaces = [
                        {'id': space['id'], 'polygon': np.array(space['polygon'], dtype=np.int32)}
                        for space in config.get('spaces', [])
                    ]
                    
                    # Load my car space ID
                    self.my_car_space_id = config.get('my_car_space_id', None)
                    if self.my_car_space_id:
                        self.my_car_monitoring_enabled = True
                        print(f"🚗 My car is in parking space #{self.my_car_space_id}")
                    
                    # Initialize states for loaded spaces
                    for space in self.parking_spaces:
                        self.space_states[space['id']] = {
                            'occupied': False,
                            'last_change': time.time(),
                            'pending_state': None,
                            'pending_since': None,
                            'history': deque(maxlen=self.history_length)
                        }
                    
                    if self.parking_spaces:
                        self.setup_mode = False
                        print(f"✅ Loaded {len(self.parking_spaces)} parking spaces from {self.config_file}")
                        
                        # Update next_space_id to be higher than any existing ID
                        if self.parking_spaces:
                            max_id = max(space['id'] for space in self.parking_spaces)
                            self.next_space_id = max_id + 1
                    else:
                        print(f"⚠️ No parking spaces found in {self.config_file}")
            except Exception as e:
                print(f"❌ Error loading configuration: {e}")
    
    def save_configuration(self):
        """Save parking space configuration to file."""
        config = {
            'spaces': [
                {'id': space['id'], 'polygon': space['polygon'].tolist()}
                for space in self.parking_spaces
            ],
            'my_car_space_id': self.my_car_space_id,
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Saved {len(self.parking_spaces)} parking spaces to {self.config_file}")
            if self.my_car_space_id:
                print(f"   🚗 My car space: #{self.my_car_space_id}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def log_event(self, event_type, space_id, details=None):
        """Log parking events to file and memory."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'space_id': space_id,
            'details': details or {}
        }
        
        self.events_log.append(event)
        
        # Save to file
        try:
            events_to_save = self.events_log[-100:]  # Keep last 100 events
            with open(self.log_file, 'w') as f:
                json.dump(events_to_save, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving log: {e}")
        
        return event
    
    def show_popup_notification(self, space_id, notification_type="empty", details=""):
        """Show a popup notification window on top when a parking space becomes empty or security alert."""
        def create_popup():
            # Create notification window
            popup_window = f"{notification_type.upper()}_PARKING_{space_id}"
            
            # Create a bright notification image
            popup_height = 250
            popup_width = 600
            notification_img = np.zeros((popup_height, popup_width, 3), dtype=np.uint8)
            
            # Background color based on notification type
            if notification_type == "empty":
                notification_img[:] = (0, 180, 0)  # Green for empty
                main_text = "EMPTY PARKING"
            elif notification_type == "security":
                notification_img[:] = (0, 0, 200)  # Red for security alert
                main_text = "SECURITY ALERT"
            elif notification_type == "shake":
                notification_img[:] = (0, 100, 255)  # Orange for shake/impact
                main_text = "IMPACT DETECTED"
            elif notification_type == "movement":
                notification_img[:] = (200, 150, 0)  # Blue for car movement
                main_text = "CAR MOVING"
            else:
                notification_img[:] = (0, 180, 0)
                main_text = "NOTIFICATION"
            
            # Add white border
            cv2.rectangle(notification_img, (5, 5), (popup_width-5, popup_height-5), (255, 255, 255), 10)
            
            # Add text
            space_text = f"Space #{space_id}"
            if details:
                detail_text = details
            else:
                detail_text = ""
            time_text = datetime.now().strftime("%H:%M:%S")
            
            # Main text - large and bold (using SIMPLEX with thick lines)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0 if notification_type != "empty" else 2.5
            thickness = 6
            (text_width, text_height), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
            text_x = (popup_width - text_width) // 2
            text_y = 80
            
            # Add shadow effect
            cv2.putText(notification_img, main_text, (text_x + 3, text_y + 3),
                       font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(notification_img, main_text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Space number
            font_scale_small = 1.2
            thickness_small = 3
            (text_width_small, text_height_small), _ = cv2.getTextSize(space_text, font, font_scale_small, thickness_small)
            text_x_small = (popup_width - text_width_small) // 2
            text_y_small = text_y + 50
            
            cv2.putText(notification_img, space_text, (text_x_small + 2, text_y_small + 2),
                       font, font_scale_small, (0, 0, 0), thickness_small + 1)
            cv2.putText(notification_img, space_text, (text_x_small, text_y_small),
                       font, font_scale_small, (255, 255, 255), thickness_small)
            
            # Details text
            if detail_text:
                font_scale_detail = 0.8
                thickness_detail = 2
                (text_width_detail, _), _ = cv2.getTextSize(detail_text, font, font_scale_detail, thickness_detail)
                text_x_detail = (popup_width - text_width_detail) // 2
                text_y_detail = text_y_small + 40
                cv2.putText(notification_img, detail_text, (text_x_detail, text_y_detail),
                           font, font_scale_detail, (255, 255, 255), thickness_detail)
            
            # Timestamp
            font_scale_time = 0.7
            thickness_time = 2
            cv2.putText(notification_img, time_text, (20, popup_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_time, (255, 255, 255), thickness_time)
            
            # Create window and set it to be always on top
            cv2.namedWindow(popup_window, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(popup_window, cv2.WND_PROP_TOPMOST, 1)
            cv2.resizeWindow(popup_window, popup_width, popup_height)
            
            # Show the notification
            start_time = time.time()
            while (time.time() - start_time) < self.popup_duration:
                try:
                    cv2.imshow(popup_window, notification_img)
                    
                    # Check if window was closed manually
                    if cv2.getWindowProperty(popup_window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    break  # Window was closed
                
                # Allow ESC or any key to close
                key = cv2.waitKey(100) & 0xFF
                if key != 255:  # Any key pressed
                    break
            
            # Close the popup safely
            try:
                cv2.destroyWindow(popup_window)
            except:
                pass  # Window already destroyed or doesn't exist
            
            # Remove from active popups list
            if popup_window in self.active_popups:
                self.active_popups.remove(popup_window)
        
        # Run popup in a separate thread so it doesn't block the main monitoring
        popup_thread = threading.Thread(target=create_popup, daemon=True)
        popup_thread.start()
        
        # Track active popup
        self.active_popups.append(f"{notification_type.upper()}_PARKING_{space_id}")

    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing parking spaces and selecting cars."""
        # Car selection in monitor mode
        if not self.setup_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # First check if clicking on a tracked car
                clicked_car_id = None
                for car_id, tracked in self.tracked_cars.items():
                    x1, y1, x2, y2 = tracked['bbox']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        clicked_car_id = car_id
                        break
                
                if clicked_car_id is not None:
                    # Toggle selection of tracked car
                    if clicked_car_id in self.selected_cars:
                        self.selected_cars.remove(clicked_car_id)
                        print(f"❌ Deselected car ID:{clicked_car_id}")
                    else:
                        self.selected_cars.add(clicked_car_id)
                        print(f"✅ Selected car ID:{clicked_car_id} for tracking")
                        print(f"   Total selected: {len(self.selected_cars)}")
                else:
                    # Check if clicking on an untracked detection
                    for detection in self.current_detections:
                        x1, y1, x2, y2 = detection['bbox']
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            # Add this car to tracking
                            car_id = self.car_id_next
                            self.car_id_next += 1
                            
                            cx, cy = detection['center']
                            self.tracked_cars[car_id] = {
                                'position': (cx, cy),
                                'bbox': detection['bbox'],
                                'class': detection['class'],
                                'stationary_frames': 0,
                                'stationary': False,
                                'baseline_position': (cx, cy),
                                'position_history': deque([(cx, cy)], maxlen=self.position_history_size),
                                'last_seen': time.time(),
                                'matched_this_frame': True
                            }
                            self.selected_cars.add(car_id)
                            print(f"🆕 Added car ID:{car_id} to tracking at ({cx}, {cy})")
                            print(f"   Type: {detection['class']}")
                            break
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right-click to remove car from tracking
                clicked_car_id = None
                for car_id, tracked in self.tracked_cars.items():
                    x1, y1, x2, y2 = tracked['bbox']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        clicked_car_id = car_id
                        break
                
                if clicked_car_id is not None:
                    # Remove from tracking
                    del self.tracked_cars[clicked_car_id]
                    # Also remove from selected if it was selected
                    if clicked_car_id in self.selected_cars:
                        self.selected_cars.remove(clicked_car_id)
                    print(f"🗑️ Removed car ID:{clicked_car_id} from tracking")
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on existing space with Ctrl held
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Delete parking space if clicking inside it
                space_to_delete = None
                space_index = None
                for idx, space in enumerate(self.parking_spaces):
                    if self.point_in_polygon((x, y), space['polygon']):
                        space_to_delete = space
                        space_index = idx
                        break
                
                if space_to_delete is not None:
                    space_id = space_to_delete['id']
                    # Remove by index to avoid array comparison issues
                    self.parking_spaces.pop(space_index)
                    if space_id in self.space_states:
                        del self.space_states[space_id]
                    print(f"🗑️ Deleted parking space #{space_id}")
                else:
                    print(f"⚠️ No parking space found at ({x}, {y})")
            else:
                # Add point to current polygon
                self.current_polygon.append([x, y])
                print(f"📍 Added point {len(self.current_polygon)}: ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete current polygon and create parking space
            if len(self.current_polygon) >= 3:
                space_id = self.next_space_id
                polygon = np.array(self.current_polygon, dtype=np.int32)
                
                self.parking_spaces.append({
                    'id': space_id,
                    'polygon': polygon
                })
                
                # Initialize state for new space
                self.space_states[space_id] = {
                    'occupied': False,
                    'last_change': time.time(),
                    'pending_state': None,
                    'pending_since': None,
                    'history': deque(maxlen=self.history_length)
                }
                
                print(f"✅ Created parking space #{space_id} with {len(self.current_polygon)} points")
                self.current_polygon = []
                self.next_space_id += 1  # Increment for next space
            else:
                print(f"⚠️ Need at least 3 points to create a parking space (current: {len(self.current_polygon)})")
    
    def calculate_polygon_iou(self, polygon1, polygon2):
        """Calculate intersection over union between two polygons."""
        # Create masks for both polygons
        mask1 = np.zeros((1000, 1000), dtype=np.uint8)
        mask2 = np.zeros((1000, 1000), dtype=np.uint8)
        
        cv2.fillPoly(mask1, [polygon1], 255)
        cv2.fillPoly(mask2, [polygon2], 255)
        
        # Calculate intersection and union
        intersection = cv2.bitwise_and(mask1, mask2)
        union = cv2.bitwise_or(mask1, mask2)
        
        intersection_area = cv2.countNonZero(intersection)
        union_area = cv2.countNonZero(union)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def calculate_bbox_polygon_overlap(self, bbox, polygon):
        """Calculate how much of a bounding box overlaps with a polygon."""
        x1, y1, x2, y2 = bbox
        
        # Create bounding box polygon
        bbox_polygon = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)
        
        # Create masks with sufficient size
        max_y = max(y2 + 10, int(polygon[:, 1].max()) + 10)
        max_x = max(x2 + 10, int(polygon[:, 0].max()) + 10)
        frame_shape = (max_y, max_x)
        
        mask_bbox = np.zeros(frame_shape, dtype=np.uint8)
        mask_polygon = np.zeros(frame_shape, dtype=np.uint8)
        
        cv2.fillPoly(mask_bbox, [bbox_polygon], 255)
        cv2.fillPoly(mask_polygon, [polygon], 255)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(mask_bbox, mask_polygon)
        intersection_area = cv2.countNonZero(intersection)
        
        # Calculate overlap relative to parking space area (not bbox area)
        # This prevents small vehicles from being missed in large spaces
        polygon_area = cv2.countNonZero(mask_polygon)
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if polygon_area == 0 or bbox_area == 0:
            return 0.0
        
        # Use the ratio of intersection to the smaller of the two areas
        # This helps detect partial occupancy
        overlap_ratio = intersection_area / min(polygon_area, bbox_area)
        
        return overlap_ratio
    
    def check_parking_occupancy(self, frame, detections):
        """Check which parking spaces are occupied by vehicles."""
        current_time = time.time()
        
        # Get vehicle detections with lower confidence threshold
        vehicle_bboxes = []
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = self.model.names[int(class_id)]
                
                # Consider cars, trucks, buses, motorcycles as vehicles
                # Lowered confidence threshold to catch more vehicles
                if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > self.confidence_threshold:
                    vehicle_bboxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        # Check each parking space
        for space in self.parking_spaces:
            space_id = space['id']
            polygon = space['polygon']
            
            # Check if any vehicle significantly overlaps with this space
            max_overlap = 0.0
            best_vehicle = None
            for i, bbox in enumerate(vehicle_bboxes):
                x1, y1, x2, y2, conf = bbox
                overlap = self.calculate_bbox_polygon_overlap((x1, y1, x2, y2), polygon)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_vehicle = (i, conf)
            
            # Determine current occupancy
            is_occupied = max_overlap > self.occupancy_threshold
            
            # Update history
            state = self.space_states[space_id]
            state['history'].append(is_occupied)
            
            # Store debug info
            if self.debug_mode and best_vehicle:
                vehicle_idx, vehicle_conf = best_vehicle
                state['debug_info'] = {
                    'overlap': max_overlap,
                    'threshold': self.occupancy_threshold,
                    'vehicle_conf': vehicle_conf,
                    'occupied': is_occupied
                }
            
            # Check for state change
            current_state = state['occupied']
            
            if is_occupied != current_state:
                # State change detected
                if state['pending_state'] == is_occupied:
                    # Same pending state, check if enough time has passed
                    if current_time - state['pending_since'] >= self.state_change_delay:
                        # Confirm state change
                        old_state = current_state
                        state['occupied'] = is_occupied
                        state['last_change'] = current_time
                        state['pending_state'] = None
                        state['pending_since'] = None
                        
                        # Record pattern event for analytics
                        event_type_pattern = 'occupied' if is_occupied else 'empty'
                        self.record_occupancy_event(space_id, event_type_pattern, current_time)
                        
                        # Log event
                        if is_occupied:
                            event_type = 'space_occupied'
                            self.total_occupied_events += 1
                            print(f"🚗 Parking space #{space_id} is now OCCUPIED")
                        else:
                            event_type = 'space_emptied'
                            self.total_empty_events += 1
                            print(f"✨ Parking space #{space_id} is now EMPTY!")
                            
                            # Show popup notification for empty space
                            if self.notifications_enabled:
                                self.show_popup_notification(space_id, "empty")
                            
                            # Play notification sound (simple beep)
                            if self.notification_sound:
                                print('\a')  # Terminal beep
                        
                        self.log_event(event_type, space_id, {
                            'previous_state': 'occupied' if old_state else 'empty',
                            'new_state': 'occupied' if is_occupied else 'empty',
                            'overlap': max_overlap
                        })
                else:
                    # New pending state
                    state['pending_state'] = is_occupied
                    state['pending_since'] = current_time
            else:
                    # No state change, clear pending
                    if state['pending_state'] is not None:
                        state['pending_state'] = None
                        state['pending_since'] = None
    
    def track_car_movement(self, detections):
        """Track all cars and detect when stationary cars start moving."""
        if detections is None:
            return
        
        # Store current detections for click selection and get all detected cars
        self.current_detections = []
        current_cars = []
        for box in detections.data:
            x1, y1, x2, y2, conf, class_id = box
            class_name = self.model.names[int(class_id)]
            
            if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > self.confidence_threshold:
                car_cx = (int(x1) + int(x2)) // 2
                car_cy = (int(y1) + int(y2)) // 2
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                detection_info = {
                    'center': (car_cx, car_cy),
                    'bbox': bbox,
                    'class': class_name
                }
                self.current_detections.append(detection_info)
                current_cars.append(detection_info)
        
        # Match current cars with tracked cars (only update existing tracked cars)
        matched_ids = set()
        for car in current_cars:
            car_cx, car_cy = car['center']
            
            # Find closest tracked car (within reasonable distance)
            best_match_id = None
            best_distance = float('inf')
            
            for car_id, tracked in self.tracked_cars.items():
                if tracked.get('matched_this_frame'):
                    continue
                    
                tracked_x, tracked_y = tracked['position']
                distance = math.sqrt((car_cx - tracked_x)**2 + (car_cy - tracked_y)**2)
                
                # Only match if car hasn't moved too far (likely same car)
                if distance < 150 and distance < best_distance:
                    best_distance = distance
                    best_match_id = car_id
            
            if best_match_id is not None:
                # Update existing tracked car
                tracked = self.tracked_cars[best_match_id]
                
                # Add current position to history buffer
                if 'position_history' not in tracked:
                    tracked['position_history'] = deque(maxlen=self.position_history_size)
                    tracked['position_history'].append(tracked['position'])
                tracked['position_history'].append((car_cx, car_cy))
                
                # Calculate smoothed position (average of recent positions)
                if len(tracked['position_history']) >= 3:
                    avg_x = sum(p[0] for p in tracked['position_history']) / len(tracked['position_history'])
                    avg_y = sum(p[1] for p in tracked['position_history']) / len(tracked['position_history'])
                    smoothed_pos = (int(avg_x), int(avg_y))
                else:
                    smoothed_pos = (car_cx, car_cy)
                
                # Calculate displacement from baseline position (when became stationary)
                if 'baseline_position' not in tracked:
                    tracked['baseline_position'] = smoothed_pos
                
                baseline_x, baseline_y = tracked['baseline_position']
                displacement = math.sqrt((smoothed_pos[0] - baseline_x)**2 + (smoothed_pos[1] - baseline_y)**2)
                
                # Check if car was stationary and is now moving
                if tracked['stationary'] and displacement > self.movement_threshold:
                    # Only notify if selection mode is off OR car is selected
                    should_notify = not self.selection_mode or best_match_id in self.selected_cars
                    
                    if should_notify:
                        print(f"🚗 CAR STARTED MOVING! ID:{best_match_id} moved {displacement:.1f}px")
                        print(f"   Type: {car['class']}, From: {tracked['baseline_position']} To: {smoothed_pos}")
                        
                        if self.notifications_enabled:
                            self.show_popup_notification(
                                best_match_id,
                                "movement",
                                f"{car['class'].upper()} started moving"
                            )
                        
                        self.log_event('car_started_moving', best_match_id, {
                            'displacement': displacement,
                            'from_position': tracked['baseline_position'],
                            'to_position': smoothed_pos,
                            'vehicle_type': car['class'],
                            'selected': best_match_id in self.selected_cars
                        })
                        print('\a\a')  # Double beep
                    
                    tracked['stationary'] = False
                    tracked['stationary_frames'] = 0
                    tracked['baseline_position'] = smoothed_pos  # Reset baseline
                elif displacement < 15:  # Increased tolerance for detection jitter
                    tracked['stationary_frames'] += 1
                    if tracked['stationary_frames'] >= self.stationary_threshold:
                        if not tracked['stationary']:
                            # Just became stationary, set new baseline
                            tracked['baseline_position'] = smoothed_pos
                        tracked['stationary'] = True
                else:  # Moderate movement, not stationary
                    tracked['stationary_frames'] = 0
                    tracked['stationary'] = False
                    tracked['baseline_position'] = smoothed_pos  # Update baseline while moving
                
                tracked['position'] = smoothed_pos
                tracked['bbox'] = car['bbox']
                tracked['class'] = car['class']
                tracked['last_seen'] = time.time()
                tracked['matched_this_frame'] = True
                matched_ids.add(best_match_id)
            # Note: Cars not in tracked_cars are ignored - user must click to add them
        
        # Clear match flags for next frame
        for car_id in self.tracked_cars:
            self.tracked_cars[car_id]['matched_this_frame'] = False
        
        # Remove cars that haven't been seen for a while (5 seconds)
        current_time = time.time()
        cars_to_remove = []
        for car_id, tracked in self.tracked_cars.items():
            if current_time - tracked['last_seen'] > 5.0:
                cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            print(f"❌ Lost track of car ID:{car_id}")
            del self.tracked_cars[car_id]
            # Also remove from selected cars if it was selected
            if car_id in self.selected_cars:
                self.selected_cars.remove(car_id)
                print(f"   Removed from selected cars")
    
    def monitor_my_car_security(self, frame, detections):
        """Monitor security threats to user's car."""
        if not self.my_car_monitoring_enabled or self.my_car_space_id is None:
            return
        
        if self.my_car_space_id not in self.space_states:
            return
        
        # Only monitor if my car space is occupied
        if not self.space_states[self.my_car_space_id]['occupied']:
            return
        
        # Get my car's parking space polygon
        my_car_space = None
        for space in self.parking_spaces:
            if space['id'] == self.my_car_space_id:
                my_car_space = space
                break
        
        if my_car_space is None:
            return
        
        polygon = my_car_space['polygon']
        
        # Calculate center of my car's parking space
        M = cv2.moments(polygon)
        if M['m00'] != 0:
            space_cx = int(M['m10'] / M['m00'])
            space_cy = int(M['m01'] / M['m00'])
        else:
            space_cx = polygon[0][0]
            space_cy = polygon[0][1]
        
        # Find car in my space
        my_car_bbox = None
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = self.model.names[int(class_id)]
                
                if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > self.confidence_threshold:
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    overlap = self.calculate_bbox_polygon_overlap(bbox, polygon)
                    
                    if overlap > self.occupancy_threshold:
                        my_car_bbox = bbox
                        break
        
        # Check for people near my car
        person_detected_near = False
        vehicle_detected_near = False
        
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = self.model.names[int(class_id)]
                
                if conf > self.confidence_threshold:
                    obj_cx = (int(x1) + int(x2)) // 2
                    obj_cy = (int(y1) + int(y2)) // 2
                    
                    # Calculate distance to my car's space center
                    distance = math.sqrt((obj_cx - space_cx)**2 + (obj_cy - space_cy)**2)
                    
                    # Check if it's the car in my space (ignore it)
                    if my_car_bbox:
                        mcx1, mcy1, mcx2, mcy2 = my_car_bbox
                        if (abs(obj_cx - (mcx1 + mcx2)//2) < 50 and 
                            abs(obj_cy - (mcy1 + mcy2)//2) < 50):
                            continue  # This is my car, skip it
                    
                    # Person proximity alert
                    if class_name == 'person' and distance < self.proximity_threshold:
                        person_detected_near = True
                        if not self.person_near_my_car:
                            print(f"⚠️ PERSON NEAR MY CAR! Distance: {distance:.1f}px")
                            if self.notifications_enabled:
                                self.show_popup_notification(
                                    self.my_car_space_id, 
                                    "security", 
                                    f"Person detected: {distance:.0f}px away"
                                )
                            self.log_event('my_car_person_proximity', self.my_car_space_id, {
                                'distance': distance,
                                'threshold': self.proximity_threshold
                            })
                            print('\a\a')  # Double beep for person
                    
                    # Vehicle proximity alert
                    if class_name in ['car', 'truck', 'bus', 'motorcycle'] and distance < self.proximity_threshold:
                        vehicle_detected_near = True
                        if not self.vehicle_near_my_car:
                            print(f"⚠️ VEHICLE TOO CLOSE TO MY CAR! Distance: {distance:.1f}px")
                            if self.notifications_enabled:
                                self.show_popup_notification(
                                    self.my_car_space_id, 
                                    "security", 
                                    f"Vehicle too close: {distance:.0f}px"
                                )
                            self.log_event('my_car_vehicle_proximity', self.my_car_space_id, {
                                'distance': distance,
                                'threshold': self.proximity_threshold,
                                'vehicle_type': class_name
                            })
                            print('\a\a')  # Double beep for vehicle
        
        # Update proximity states
        self.person_near_my_car = person_detected_near
        self.vehicle_near_my_car = vehicle_detected_near
        """Monitor security threats to user's car."""
        if not self.my_car_monitoring_enabled or self.my_car_space_id is None:
            return
        
        if self.my_car_space_id not in self.space_states:
            return
        
        # Only monitor if my car space is occupied
        if not self.space_states[self.my_car_space_id]['occupied']:
            return
        
        # Get my car's parking space polygon
        my_car_space = None
        for space in self.parking_spaces:
            if space['id'] == self.my_car_space_id:
                my_car_space = space
                break
        
        if my_car_space is None:
            return
        
        polygon = my_car_space['polygon']
        
        # Calculate center of my car's parking space
        M = cv2.moments(polygon)
        if M['m00'] != 0:
            space_cx = int(M['m10'] / M['m00'])
            space_cy = int(M['m01'] / M['m00'])
        else:
            space_cx = polygon[0][0]
            space_cy = polygon[0][1]
        
        # Find car in my space and track its position for shake detection
        my_car_bbox = None
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = self.model.names[int(class_id)]
                
                if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > self.confidence_threshold:
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    overlap = self.calculate_bbox_polygon_overlap(bbox, polygon)
                    
                    if overlap > self.occupancy_threshold:
                        my_car_bbox = bbox
                        break
        
        # Check for people near my car
        person_detected_near = False
        vehicle_detected_near = False
        
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = self.model.names[int(class_id)]
                
                if conf > self.confidence_threshold:
                    obj_cx = (int(x1) + int(x2)) // 2
                    obj_cy = (int(y1) + int(y2)) // 2
                    
                    # Calculate distance to my car's space center
                    distance = math.sqrt((obj_cx - space_cx)**2 + (obj_cy - space_cy)**2)
                    
                    # Check if it's the car in my space (ignore it)
                    if my_car_bbox:
                        mcx1, mcy1, mcx2, mcy2 = my_car_bbox
                        if (abs(obj_cx - (mcx1 + mcx2)//2) < 50 and 
                            abs(obj_cy - (mcy1 + mcy2)//2) < 50):
                            continue  # This is my car, skip it
                    
                    # Person proximity alert
                    if class_name == 'person' and distance < self.proximity_threshold:
                        person_detected_near = True
                        if not self.person_near_my_car:
                            print(f"⚠️ PERSON NEAR MY CAR! Distance: {distance:.1f}px")
                            if self.notifications_enabled:
                                self.show_popup_notification(
                                    self.my_car_space_id, 
                                    "security", 
                                    f"Person detected: {distance:.0f}px away"
                                )
                            self.log_event('my_car_person_proximity', self.my_car_space_id, {
                                'distance': distance,
                                'threshold': self.proximity_threshold
                            })
                            print('\a\a')  # Double beep for person
                    
                    # Vehicle proximity alert
                    if class_name in ['car', 'truck', 'bus', 'motorcycle'] and distance < self.proximity_threshold:
                        vehicle_detected_near = True
                        if not self.vehicle_near_my_car:
                            print(f"⚠️ VEHICLE TOO CLOSE TO MY CAR! Distance: {distance:.1f}px")
                            if self.notifications_enabled:
                                self.show_popup_notification(
                                    self.my_car_space_id, 
                                    "security", 
                                    f"Vehicle too close: {distance:.0f}px"
                                )
                            self.log_event('my_car_vehicle_proximity', self.my_car_space_id, {
                                'distance': distance,
                                'threshold': self.proximity_threshold,
                                'vehicle_type': class_name
                            })
                            print('\a\a')  # Double beep for vehicle
        
        # Update proximity states
        self.person_near_my_car = person_detected_near
        self.vehicle_near_my_car = vehicle_detected_near
    
    def draw_parking_spaces(self, frame):
        """Draw parking spaces on the frame with occupancy status."""
        overlay = frame.copy()
        
        for space in self.parking_spaces:
            space_id = space['id']
            polygon = space['polygon']
            state = self.space_states[space_id]
            
            # Get pattern predictions for this space
            predictions = self.get_pattern_predictions(space_id)
            empty_probability = predictions['current_hour_empty_probability'] if predictions else 0
            
            # Determine color based on state and probability
            if state.get('pending_state') is not None:
                color = self.color_pending
                status_text = "PENDING"
            elif state['occupied']:
                # Color-code occupied spaces by probability of becoming empty
                if empty_probability > 50:
                    color = (0, 200, 0)  # Bright green - likely to empty soon
                    status_text = "OCCUPIED (LIKELY EMPTY SOON)"
                elif empty_probability > 20:
                    color = (0, 200, 200)  # Yellow-green - moderate chance
                    status_text = "OCCUPIED (MAY EMPTY SOON)"
                else:
                    color = self.color_occupied
                    status_text = "OCCUPIED"
            else:
                color = self.color_empty
                status_text = "EMPTY"
            
            # Special highlighting for my car's space
            if space_id == self.my_car_space_id and self.my_car_monitoring_enabled:
                color = (255, 215, 0)  # Gold color for my car
                status_text = "MY CAR"
            
            # Draw filled polygon with transparency
            cv2.fillPoly(overlay, [polygon], color)
            
            # Draw polygon outline
            cv2.polylines(frame, [polygon], True, color, 3)
            
            # Calculate center of polygon for text
            M = cv2.moments(polygon)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = polygon[0][0]
                cy = polygon[0][1]
            
            # Draw space ID and status
            text = f"#{space_id}"
            cv2.putText(frame, text, (cx - 20, cy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (cx - 40, cy + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show pattern predictions if available
            if predictions and predictions['total_occupied_events'] > 0:
                y_offset = 40
                
                # Show probability
                prob_text = f"P(empty): {empty_probability:.0f}%"
                cv2.putText(frame, prob_text, (cx - 50, cy + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if empty_probability > 50 else (255, 255, 255), 1)
                y_offset += 15
                
                # Show average duration
                if predictions['avg_occupied_duration_minutes']:
                    duration = predictions['avg_occupied_duration_minutes']
                    dur_text = f"Avg: {duration:.0f}min"
                    cv2.putText(frame, dur_text, (cx - 50, cy + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += 15
                
                # Show peak empty hours (first 2 only for space)
                if predictions['peak_empty_hours']:
                    peak_hours = predictions['peak_empty_hours'][:2]
                    hours_str = ",".join([f"{h:02d}:00" for h, _ in peak_hours])
                    peak_text = f"Peak: {hours_str}"
                    cv2.putText(frame, peak_text, (cx - 50, cy + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
            
            # Show debug info if enabled
            elif self.debug_mode and 'debug_info' in state:
                debug = state['debug_info']
                debug_text = f"Overlap: {debug['overlap']:.3f}"
                cv2.putText(frame, debug_text, (cx - 50, cy + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def draw_setup_overlay(self, frame):
        """Draw setup mode overlay showing current polygon."""
        if self.current_polygon:
            # Draw current polygon points
            for i, point in enumerate(self.current_polygon):
                cv2.circle(frame, tuple(point), 5, self.color_setup, -1)
                cv2.putText(frame, str(i + 1), (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_setup, 2)
            
            # Draw lines between points
            if len(self.current_polygon) > 1:
                for i in range(len(self.current_polygon) - 1):
                    cv2.line(frame, tuple(self.current_polygon[i]), 
                            tuple(self.current_polygon[i + 1]), self.color_setup, 2)
                
                # Draw line from last point to first if we have at least 3 points
                if len(self.current_polygon) >= 3:
                    cv2.line(frame, tuple(self.current_polygon[-1]), 
                            tuple(self.current_polygon[0]), (128, 128, 128), 1)
        
        # Draw existing parking spaces
        for space in self.parking_spaces:
            polygon = space['polygon']
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
            
            # Draw space ID
            M = cv2.moments(polygon)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(frame, f"#{space['id']}", (cx - 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def brightness_callback(self, val):
        """Callback function for brightness slider."""
        self.brightness = self.brightness_min + (val / 100.0) * (self.brightness_max - self.brightness_min)
    
    def apply_brightness(self, frame):
        """Apply brightness adjustment to the frame."""
        if self.brightness == 0:
            return frame
        
        adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=self.brightness)
        return adjusted
    
    def test_camera_safely(self, camera_id):
        """Safely test a camera with proper error handling."""
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                time.sleep(0.5)
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
        """Find all working cameras."""
        print("🔍 Scanning for cameras...")
        working_cameras = []
        
        for i in range(6):
            print(f"   Testing camera {i}...", end=" ", flush=True)
            is_working, info = self.test_camera_safely(i)
            
            if is_working:
                working_cameras.append({'id': i, 'resolution': info, 'status': 'Working'})
                print(f"✅ {info}")
            else:
                print(f"❌ {info}")
        
        return working_cameras
    
    def setup_camera(self, camera_id):
        """Setup camera with optimized settings."""
        print(f"📹 Connecting to camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("   DirectShow failed, trying default backend...")
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return None
        
        time.sleep(1)
        
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception as e:
            print(f"⚠️ Warning: Could not set all camera properties: {e}")
        
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
        """Setup brightness slider control."""
        initial_brightness_value = int(((0 - self.brightness_min) / (self.brightness_max - self.brightness_min)) * 100)
        cv2.createTrackbar('Brightness', window_name, initial_brightness_value, 100, self.brightness_callback)
    
    def run_monitor(self):
        """Run the parking lot monitoring system."""
        print("🅿️ === Parking Lot Monitoring System ===")
        
        # Find cameras
        cameras = self.find_working_cameras()
        if not cameras:
            print("\n❌ No working cameras found!")
            return

        # Initialize available cameras and start hot-plug scanner
        with self.camera_lock:
            self.available_cameras = cameras

        # Start background scanner to detect newly attached/removed cameras
        self.stop_camera_scanner = False
        self.camera_scanner_thread = threading.Thread(target=self.camera_scanner_loop, daemon=True)
        self.camera_scanner_thread.start()

        # Select first camera by default and track index for runtime switching
        camera_index = 0
        with self.camera_lock:
            camera_id = self.available_cameras[camera_index]['id']
            res = self.available_cameras[camera_index].get('resolution', '')
            total_cams = len(self.available_cameras)

        print(f"\n✅ Using camera {camera_id} ({res})")
        if total_cams > 1:
            print(f"   (Found {total_cams} cameras total) - use '[' or '-' for previous, ']' or '+' for next camera")

        cap = self.setup_camera(camera_id)
        if cap is None:
            return
        
        print("\n🅿️ Parking Lot Monitor Active!")
        
        if self.setup_mode:
            print("\n🎨 SETUP MODE - Define Parking Spaces:")
            print("  Left Click = Add point to parking space")
            print("  Right Click = Complete parking space")
            print("  Ctrl + Left Click = Delete parking space (click inside it)")
            print("  's' = Save configuration and start monitoring")
            print("  'c' = Clear current polygon")
            print("  'u' = Undo last parking space")
        
        print("\n🎮 Controls:")
        print("  q = quit")
        print("  p = toggle setup mode")
        print("  s = save configuration")
        print("  d = toggle debug info")
        print("  n = toggle notifications")
        print("  b = reset brightness")
        print("  t = toggle selection mode (only selected cars)")
        print("  x = clear all car selections")
        print("  a = show parking pattern analytics")
        print("  m = set/unset my car space (type space number)")
        print("  [ / ] or - / + = previous / next camera")
        print("  1-9 = toggle occupancy for parking space #1-#9")
        print("  0 then number = toggle space #10+ (e.g., 0 then 1 then 2 = space #12)")
        
        print("\n🚗 Car Movement Tracking:")
        print("  - Gray boxes = Untracked cars (click to add)")
        print("  - Left Click on gray car = add to tracking")
        print("  - Left Click on tracked car = select/deselect for notifications")
        print("  - Right Click on car = remove from tracking completely")
        print("  - Green boxes = Selected stationary cars")
        print("  - Red boxes = Selected moving cars")
        print("  - Yellow/Cyan = Tracked but not selected")
        print("  - 't' = toggle selection mode (only notify for selected cars)")
        print("  - 'x' = clear all car selections")
        
        print("\n📊 Detection Confidence (thin boxes):")
        print("  - Red outline = Low confidence (0.25-0.35) - may be parked car")
        print("  - Purple outline = Medium confidence (0.35-0.50)")
        print("  - Blue outline = High confidence (>0.50)")
        
        frame_count = 0
        window_name = 'Parking Lot Monitor'
        window_created = False
        space_number_input = ""  # For multi-digit space numbers
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Camera disconnected")
                    break
                
                frame_count += 1
                original_frame = frame.copy()
                
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    self.setup_controls(window_name)
                    cv2.setMouseCallback(window_name, self.mouse_callback)
                    window_created = True
                
                # Apply brightness
                frame = self.apply_brightness(frame)
                height, width = frame.shape[:2]
                
                # Run detection if not in setup mode
                if not self.setup_mode:
                    try:
                        # Enhance image for better detection of parked cars
                        enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                        
                        # Run detection with increased sensitivity
                        results = self.model(
                            enhanced_frame,
                            conf=self.confidence_threshold,
                            iou=0.5,  # Allow more overlapping detections
                            verbose=False,
                            imgsz=640,
                            max_det=100  # Detect up to 100 objects
                        )
                        detections = results[0].boxes
                        
                        # Check parking occupancy
                        self.check_parking_occupancy(frame, detections)
                        
                        # Track all car movements
                        self.track_car_movement(detections)
                        
                        # Monitor my car security (kept for backward compatibility)
                        self.monitor_my_car_security(frame, detections)
                        
                        # Draw tracked cars with IDs and status
                        for car_id, tracked in self.tracked_cars.items():
                            x1, y1, x2, y2 = tracked['bbox']
                            is_selected = car_id in self.selected_cars
                            
                            # Color based on selection and status
                            if is_selected:
                                # Selected cars: bright green (stationary) or bright red (moving)
                                if tracked['stationary']:
                                    color = (0, 255, 0)  # Bright green for selected stationary
                                    status = "SELECTED-STATIONARY"
                                else:
                                    color = (0, 0, 255)  # Bright red for selected moving
                                    status = "SELECTED-MOVING"
                                thickness = 4  # Thicker border for selected
                            else:
                                # Unselected cars: yellow (stationary) or cyan (moving)
                                if tracked['stationary']:
                                    color = (0, 255, 255)  # Yellow for stationary
                                    status = "STATIONARY"
                                else:
                                    color = (255, 255, 0)  # Cyan for moving
                                    status = "MOVING"
                                thickness = 2  # Thinner border for unselected
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                            label = f"ID:{car_id} {tracked['class']} {status}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Draw center point (larger for selected)
                            cx, cy = tracked['position']
                            circle_size = 8 if is_selected else 5
                            cv2.circle(frame, (cx, cy), circle_size, color, -1)
                        
                        # Draw clickable areas for untracked detections (semi-transparent)
                        overlay = frame.copy()
                        for detection in self.current_detections:
                            x1, y1, x2, y2 = detection['bbox']
                            
                            # Check if this detection is already tracked
                            is_tracked = False
                            for car_id, tracked in self.tracked_cars.items():
                                tx1, ty1, tx2, ty2 = tracked['bbox']
                                # Check for significant overlap
                                if abs(x1 - tx1) < 20 and abs(y1 - ty1) < 20:
                                    is_tracked = True
                                    break
                            
                            if not is_tracked:
                                # Draw semi-transparent gray box for untracked cars (clickable)
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 2)
                                cv2.putText(overlay, f"Click to track", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                        
                        # Blend overlay with original frame
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        
                        # Draw vehicle detections (very light overlay with confidence scores)
                        if self.debug_mode and detections is not None:
                            for box in detections.data:
                                x1, y1, x2, y2, conf, class_id = box
                                class_name = self.model.names[int(class_id)]
                                
                                if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > self.confidence_threshold:
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                    # Color code by confidence: red for low, purple for medium, blue for high
                                    if conf < 0.35:
                                        det_color = (0, 0, 255)  # Red for low confidence
                                    elif conf < 0.50:
                                        det_color = (255, 0, 255)  # Purple for medium
                                    else:
                                        det_color = (255, 100, 0)  # Blue for high confidence
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), det_color, 1)
                                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 30),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, det_color, 1)
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Detection error: {e}")
                    
                    # Draw parking spaces (always draw in monitoring mode)
                    frame = self.draw_parking_spaces(frame)
                else:
                    # Draw setup overlay
                    frame = self.draw_setup_overlay(frame)
                
                # Status display
                status_y = 30
                
                if self.setup_mode:
                    cv2.putText(frame, "🎨 SETUP MODE - Define Parking Spaces", (20, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    status_y += 35
                    cv2.putText(frame, f"Spaces defined: {len(self.parking_spaces)} | Current points: {len(self.current_polygon)}",
                               (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "🅿️ Parking Lot Monitor", (20, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Display statistics
                    status_y += 40
                    empty_count = sum(1 for s in self.space_states.values() if not s['occupied'])
                    occupied_count = len(self.parking_spaces) - empty_count
                    
                    cv2.putText(frame, f"Spaces: {len(self.parking_spaces)} | Empty: {empty_count} | Occupied: {occupied_count}",
                               (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    status_y += 30
                    cv2.putText(frame, f"Events - Empty: {self.total_empty_events} | Occupied: {self.total_occupied_events}",
                               (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Car tracking statistics
                    status_y += 30
                    stationary_count = sum(1 for t in self.tracked_cars.values() if t['stationary'])
                    moving_count = len(self.tracked_cars) - stationary_count
                    cv2.putText(frame, f"Tracked Cars: {len(self.tracked_cars)} | Stationary: {stationary_count} | Moving: {moving_count}",
                               (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Selected cars info
                    status_y += 25
                    mode_text = "Selection Mode: ON" if self.selection_mode else "Selection Mode: OFF"
                    color = (0, 255, 0) if self.selection_mode else (128, 128, 128)
                    cv2.putText(frame, f"{mode_text} | Selected: {len(self.selected_cars)}",
                               (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Frame info
                status_y += 30
                cv2.putText(frame, f"Frame: {frame_count} | Camera: {camera_id}",
                           (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Controls info
                if self.setup_mode:
                    controls = "Left-Click=Add | Right-Click=Complete | Ctrl+Click=Delete | 's'=Save | 'c'=Clear | 'u'=Undo | 'p'=Monitor | 'q'=Quit"
                else:
                    controls = "q=quit | p=setup | s=save | d=debug | n=notifications | b=brightness | 1-9=toggle space"
                    if space_number_input:
                        controls += f" | Input: {space_number_input}"
                
                cv2.putText(frame, controls, (20, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Small on-screen prompt for camera switching
                try:
                    camera_hint = "Camera: [ / ]  or  - / +  to switch"
                    cv2.putText(frame, camera_hint, (max(20, width - 420), 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                except Exception:
                    pass
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_configuration()
                    if self.setup_mode:
                        self.setup_mode = False
                        print("✅ Switched to monitoring mode")
                elif key == ord('p'):
                    self.setup_mode = not self.setup_mode
                    if self.setup_mode:
                        print("🎨 Switched to setup mode")
                    else:
                        print("📊 Switched to monitoring mode")
                elif key == ord('c') and self.setup_mode:
                    self.current_polygon = []
                    print("🗑️ Cleared current polygon")
                elif key == ord('u') and self.setup_mode:
                    if self.parking_spaces:
                        removed = self.parking_spaces.pop()
                        del self.space_states[removed['id']]
                        print(f"↩️ Removed parking space #{removed['id']}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('n'):
                    self.notifications_enabled = not self.notifications_enabled
                    print(f"🔔 Notifications: {'ON' if self.notifications_enabled else 'OFF'}")
                elif key == ord('b'):
                    self.brightness = 0
                    brightness_value = int(((0 - self.brightness_min) / (self.brightness_max - self.brightness_min)) * 100)
                    cv2.setTrackbarPos('Brightness', window_name, brightness_value)
                    print("💡 Brightness reset to 0")
                elif key in (ord(']'), ord('+'), ord('l')):
                    # Switch to next camera (also accept 'l' as alternative)
                    with self.camera_lock:
                        cams = list(self.available_cameras)
                    if len(cams) > 1:
                        print("🔁 Switching to next camera...")
                        # Remember previous state
                        prev_index = camera_index
                        prev_cam_id = cams[prev_index]['id'] if prev_index < len(cams) else None

                        # Advance index safely
                        camera_index = (camera_index + 1) % len(cams)
                        next_cam = cams[camera_index]
                        camera_id = next_cam['id']
                        res = next_cam.get('resolution', '')
                        print(f"📹 Trying camera {camera_id} ({res})")

                        # Attempt to open new camera
                        try:
                            if 'cap' in locals() and cap is not None:
                                try:
                                    cap.release()
                                except:
                                    pass
                        except NameError:
                            pass

                        new_cap = self.setup_camera(camera_id)
                        if new_cap is None:
                            print("⚠️ Failed to open new camera, reverting to previous camera")
                            camera_index = prev_index
                            # Try to reopen previous camera
                            try:
                                prev_cam_id = cams[camera_index]['id']
                                cap = self.setup_camera(prev_cam_id)
                            except Exception:
                                cap = None
                                print("❌ Could not reopen previous camera")
                        else:
                            cap = new_cap
                    else:
                        print("⚠️ Not enough cameras to switch")
                elif key in (ord('['), ord('-'), ord('h')):
                    # Switch to previous camera (also accept 'h' as alternative)
                    with self.camera_lock:
                        cams = list(self.available_cameras)
                    if len(cams) > 1:
                        print("🔁 Switching to previous camera...")
                        prev_index = camera_index

                        # Move back safely
                        camera_index = (camera_index - 1) % len(cams)
                        prev_cam = cams[camera_index]
                        camera_id = prev_cam['id']
                        res = prev_cam.get('resolution', '')
                        print(f"📹 Trying camera {camera_id} ({res})")

                        try:
                            if 'cap' in locals() and cap is not None:
                                try:
                                    cap.release()
                                except:
                                    pass
                        except NameError:
                            pass

                        new_cap = self.setup_camera(camera_id)
                        if new_cap is None:
                            print("⚠️ Failed to open previous camera, reverting to original")
                            camera_index = prev_index
                            try:
                                orig_cam_id = cams[camera_index]['id']
                                cap = self.setup_camera(orig_cam_id)
                            except Exception:
                                cap = None
                                print("❌ Could not reopen original camera")
                        else:
                            cap = new_cap
                    else:
                        print("⚠️ Not enough cameras to switch")
                elif key == ord('t'):
                    # Toggle selection mode
                    self.selection_mode = not self.selection_mode
                    mode_status = "ON (only selected cars)" if self.selection_mode else "OFF (all cars)"
                    print(f"🎯 Selection Mode: {mode_status}")
                    print(f"   Currently selected: {len(self.selected_cars)} cars")
                elif key == ord('x'):
                    # Clear all car selections
                    cleared_count = len(self.selected_cars)
                    self.selected_cars.clear()
                    print(f"🧹 Cleared {cleared_count} car selections")
                elif key == ord('a'):
                    # Display analytics summary
                    print("\n" + "="*80)
                    print("📊 PARKING PATTERN ANALYTICS")
                    print("="*80)
                    self.display_analytics_summary()
                    print("="*80 + "\n")
                elif key == ord('m'):
                    # Set my car space
                    print("\n🚗 Enter parking space number for YOUR car (or 0 to disable):")
                    try:
                        my_space = input("   Space #: ").strip()
                        if my_space:
                            space_num = int(my_space)
                            if space_num == 0:
                                self.my_car_space_id = None
                                self.my_car_monitoring_enabled = False
                                print("   ✅ My car monitoring disabled")
                            elif space_num in self.space_states:
                                self.my_car_space_id = space_num
                                self.my_car_monitoring_enabled = True
                                print(f"   ✅ Set space #{space_num} as MY CAR")
                                print(f"   🔒 Security monitoring active:")
                                print(f"      - Person proximity: {self.proximity_threshold}px")
                                print(f"      - Vehicle proximity: {self.proximity_threshold}px")
                                self.save_configuration()
                            else:
                                print(f"   ⚠️ Parking space #{space_num} does not exist")
                    except ValueError:
                        print("   ⚠️ Invalid input")
                elif key >= ord('0') and key <= ord('9'):
                    # Handle number input for parking space toggle
                    digit = chr(key)
                    space_number_input += digit
                    
                    # Try to parse the space number
                    try:
                        space_num = int(space_number_input)
                        
                        # Check if this space exists
                        if space_num in self.space_states:
                            # Toggle the space occupancy
                            current_state = self.space_states[space_num]['occupied']
                            self.space_states[space_num]['occupied'] = not current_state
                            self.space_states[space_num]['last_change'] = time.time()
                            self.space_states[space_num]['pending_state'] = None
                            self.space_states[space_num]['pending_since'] = None
                            
                            new_state = "OCCUPIED" if not current_state else "EMPTY"
                            old_state = "OCCUPIED" if current_state else "EMPTY"
                            print(f"🔄 Manually toggled parking space #{space_num}: {old_state} → {new_state}")
                            
                            # Log the manual override
                            self.log_event('manual_override', space_num, {
                                'previous_state': old_state.lower(),
                                'new_state': new_state.lower(),
                                'method': 'manual_keyboard'
                            })
                            
                            # Reset input
                            space_number_input = ""
                        elif space_num > max(self.space_states.keys()) if self.space_states else 0:
                            # Number too large, wait for more input if it's reasonable
                            if space_num > 99:
                                print(f"⚠️ No parking space #{space_num}")
                                space_number_input = ""
                    except ValueError:
                        # Not a valid number yet, keep accumulating
                        pass
                elif key == 27:  # ESC key
                    # Clear the space number input
                    if space_number_input:
                        print(f"❌ Cleared input: {space_number_input}")
                        space_number_input = ""
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n📊 Final Statistics:")
            print(f"   📹 Camera: {camera_id}")
            print(f"   🎬 Frames processed: {frame_count}")
            print(f"   🅿️ Parking spaces: {len(self.parking_spaces)}")
            print(f"   ✨ Empty events: {self.total_empty_events}")
            print(f"   🚗 Occupied events: {self.total_occupied_events}")
            print(f"   📝 Events logged to: {self.log_file}")
            print("   🅿️ Parking lot monitoring completed")

def main():
    print("🅿️ === Parking Lot Monitoring System ===")
    print("Real-time parking space occupancy detection")
    print("\nFeatures:")
    print("  🎨 Interactive parking space setup")
    print("  🚗 Vehicle detection (car, truck, bus, motorcycle)")
    print("  ✨ Real-time empty space notifications")
    print("  📊 Occupancy statistics and logging")
    print("  💡 Brightness adjustment")
    print("  💾 Save/load configuration")
    print("\n🚀 Starting monitor...")
    
    monitor = ParkingLotMonitor()
    monitor.run_monitor()

if __name__ == "__main__":
    main()
