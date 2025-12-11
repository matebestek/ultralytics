# Parking Lot Monitor - Version 3.0

## 🆕 What's New in v3

Version 3 continues the predictive analytics evolution and adds hot-plug camera support and improved UX for camera switching.

**Key Additions:**
- 🔌 **Hot-plug camera detection**: The app now rescans attached cameras periodically and updates the available camera list without restart.
- 🎛️ **Runtime camera switching**: Press `[` / `]` or `-` / `+` to switch previous/next camera while the monitor is running.
- 💬 **On-screen camera hint**: A small prompt appears in the video feed showing camera switching keys.

## Backward Compatibility
- Compatible with v2 configuration files (`parking_config.json`) and analytics data (`parking_analytics.json`).

## Deployment Notes
- Same requirements as v2 (Python 3.10+, OpenCV, ultralytics YOLO weights).
- The package includes the new hot-plug support and updated `README.md`.

## Release Date
December 11, 2025

---

This package is intended for users who want the most responsive multi-camera experience and easier camera management during monitoring sessions.
