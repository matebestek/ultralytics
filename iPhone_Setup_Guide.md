# 📱 iPhone Camera Setup for Redarstvo Detection

## Quick Setup Guide

### Method 1: USB Connection (Recommended)

**For iOS 16+ (Continuity Camera):**
1. Update iPhone to iOS 16 or later
2. Update Mac to macOS 13+ OR use third-party software on Windows
3. Connect iPhone via USB cable
4. Run the detection script - it should auto-detect your iPhone camera

**For Windows (Third-party apps):**
1. Install one of these apps on your iPhone:
   - **Camo** (paid, high quality): https://reincubate.com/camo/
   - **EpocCam** (free/paid): Search "EpocCam" in App Store
   - **iVCam** (free/paid): Search "iVCam" in App Store

2. Install the companion software on your Windows PC
3. Connect iPhone via USB or WiFi
4. Run the detection script

### Method 2: WiFi IP Camera (Free)

1. **Install IP Camera app on iPhone:**
   - Search "IP Webcam" or "IP Camera" in App Store
   - Popular options: "IP Webcam", "AtHome Camera", "manycam"

2. **Setup steps:**
   ```
   1. Install the app and open it
   2. Connect iPhone and PC to same WiFi network
   3. Start streaming in the app
   4. Note the IP address shown (e.g., 192.168.1.100:8080)
   5. Run our detection script and enter this IP when prompted
   ```

3. **Common IP formats:**
   - `http://192.168.1.100:8080/video`
   - `rtmp://192.168.1.100:1935/live`
   - `http://192.168.1.100:8080/cam.mjpeg`

## Running the Detection

Once your iPhone camera is connected, run:

```powershell
python iphone_redarstvo_detector.py
```

The script will:
1. Auto-detect available cameras (USB connections)
2. If no USB camera found, prompt for IP camera setup
3. Start real-time Redarstvo uniform detection
4. Show higher quality detection optimized for iPhone cameras

## iPhone Camera Advantages

✅ **Higher Resolution**: iPhones typically provide 1080p or higher video
✅ **Better Image Quality**: Superior camera sensors improve detection accuracy  
✅ **Better Lighting**: Advanced image processing handles various lighting conditions
✅ **Mobility**: Can position camera at optimal angles and distances
✅ **Stability**: Can use tripods or mounts for steady detection

## Controls During Detection

- `q` = Quit detection
- `s` = Save high-resolution screenshot  
- `d` = Toggle debug mode (shows detection confidence scores)
- `f` = Toggle fullscreen mode
- `r` = Reset detection history

## Troubleshooting

**No camera detected:**
- Try different USB ports
- Restart iPhone camera apps
- Check Windows camera permissions
- Test camera with native camera app first

**Low detection accuracy:**
- Ensure good lighting
- Position camera 3-8 feet from subjects
- Try debug mode (`d` key) to see confidence scores
- Adjust camera angle to see torso/chest area clearly

**IP Camera issues:**
- Verify iPhone and PC are on same WiFi
- Check firewall settings
- Try different IP camera apps
- Test IP address in web browser first

## Expected Performance

With iPhone cameras, you should see:
- Better detection accuracy due to higher resolution
- Improved performance in varied lighting conditions
- More detailed uniform analysis
- Real-time processing at 15-30 FPS depending on resolution

## Optimization Tips

1. **Positioning**: Aim camera at chest/torso level for best uniform detection
2. **Distance**: 4-6 feet optimal for person detection and uniform analysis
3. **Lighting**: Avoid backlighting; ensure subjects are well-lit
4. **Resolution**: Higher resolution improves accuracy but may reduce frame rate
5. **Angle**: Front or slight angle view works better than profile view