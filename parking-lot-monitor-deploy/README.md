# Parking Lot Monitor - Deployment Package

## Overview
Advanced parking lot monitoring system with YOLO11 object detection, real-time car tracking, and movement alerts.

## Features
- ✅ Real-time parking space occupancy detection
- ✅ Manual car selection and tracking
- ✅ Movement detection with notifications
- ✅ Interactive parking space definition
- ✅ Configuration persistence (JSON)
- ✅ Multi-camera support (auto-selects first)
- ✅ Popup notifications for events
- ✅ Position smoothing to prevent false alerts

## Requirements
- Python 3.10 or higher
- Webcam or IP camera
- Windows/Linux/macOS

## Installation

### 1. Clone or Extract Package
```bash
cd parking-lot-monitor
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model
The `yolo11n.pt` model should be in the same directory. If missing:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## Quick Start

### Run the Application
```bash
python parking_lot_monitor.py
```

The system automatically:
- Detects and uses the first available camera
- Starts in setup mode for defining parking spaces
- Loads previous configuration if available

## Usage Guide

### Initial Setup (Define Parking Spaces)

1. **Draw Parking Spaces:**
   - Left-click to add points around a parking space
   - Right-click to complete the space
   - Repeat for all spaces

2. **Edit Spaces:**
   - Ctrl + Left-click inside a space to delete it
   - 'c' = Clear current polygon
   - 'u' = Undo last space

3. **Save Configuration:**
   - Press 's' to save and start monitoring

### Monitoring Mode

**Car Tracking:**
- Gray boxes = Detected but untracked cars
- Left-click on gray car = Add to tracking
- Left-click on tracked car = Select/deselect for alerts
- Right-click on car = Remove from tracking

**Color Coding:**
- 🟢 Green = Selected & stationary (will alert if moves)
- 🔴 Red = Selected & moving
- 🟡 Yellow = Tracked but not selected & stationary
- 🔵 Cyan = Tracked but not selected & moving

**Keyboard Controls:**
- `q` = Quit
- `p` = Toggle setup mode
- `s` = Save configuration
- `d` = Toggle debug info
- `n` = Toggle notifications
- `t` = Toggle selection mode (only alert for selected cars)
- `x` = Clear all car selections
- `b` = Reset brightness
- `m` = Set "my car" space (type number)
- `1-9` = Toggle parking space occupancy manually

### Advanced Features

**Detection Confidence (Debug Mode):**
- Red outline = Low confidence (0.25-0.35) - may be parked car
- Purple outline = Medium confidence (0.35-0.50)
- Blue outline = High confidence (>0.50)

**Movement Detection:**
- 15 frames stationary threshold
- 50 pixel movement threshold
- 5-frame position smoothing
- Baseline position tracking

**My Car Security:**
- Person proximity alerts (100px)
- Vehicle proximity alerts (100px)
- Gold highlighting for your parking space

## Configuration Files

### `parking_config.json`
Stores parking space definitions and settings:
```json
{
  "spaces": [...],
  "my_car_space_id": 1,
  "next_space_id": 10
}
```

### `parking_log.json`
Event logging (last 100 events):
- Empty/occupied changes
- Car movement alerts
- Manual overrides
- Security events

## Troubleshooting

### Camera Not Detected
- Check camera permissions
- Try different camera index
- Ensure no other application is using the camera

### Cars Not Detected
- Adjust brightness slider
- Lower confidence threshold (edit line 38: `self.confidence_threshold = 0.20`)
- Better lighting conditions

### False Movement Alerts
- Increase movement threshold (line 69: `self.movement_threshold = 60`)
- Increase stationary threshold (line 68: `self.stationary_threshold = 20`)

### Performance Issues
- Reduce image size (line 1279: `imgsz=320`)
- Increase detection interval
- Use lighter YOLO model

## Technical Specifications

**Detection:**
- Model: YOLO11n (lightweight, fast)
- Confidence: 0.25 threshold
- Classes: car, truck, bus, motorcycle
- IoU: 0.5 for overlapping detections

**Tracking:**
- Proximity-based matching (150px)
- Position smoothing (5-frame average)
- Auto-cleanup (5 seconds timeout)
- Persistent ID assignment

**Performance:**
- ~30 FPS on modern CPU
- ~60 FPS with GPU acceleration
- Resolution: 640x480 (adjustable)

## Deployment Notes

### For Production:
1. Set `self.setup_mode = False` to start in monitoring mode
2. Place `parking_config.json` with pre-defined spaces
3. Disable debug mode and popup notifications if needed
4. Consider running as a system service

### For Remote Monitoring:
- Integrate with MQTT for remote notifications
- Add database logging instead of JSON
- Implement REST API for status queries
- Stream output to web interface

## File Structure
```
parking-lot-monitor/
├── parking_lot_monitor.py    # Main application
├── requirements.txt           # Python dependencies
├── yolo11n.pt                # YOLO model weights
├── parking_config.json       # Generated parking space config
├── parking_log.json          # Event logs
└── README.md                 # This file
```

## Support
For issues or questions, check the YOLO documentation:
https://docs.ultralytics.com/

## License
Based on Ultralytics YOLO (AGPL-3.0 license)
