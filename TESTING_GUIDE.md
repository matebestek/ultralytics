# Pattern Analytics Testing Guide

## Quick Test (Simulated Data)

### 1. Generate Test Data
```bash
python test_analytics.py
```
This creates `parking_analytics.json` with 80 simulated events for Space #1:
- 25 occupied events
- 55 empty events  
- Peak empty hours: 9 AM, 1 PM, 5 PM

### 2. Verify Data Structure
```bash
python verify_analytics.py
```
Expected output:
```
🅿️  Space #1:
   Occupancy Events: 25
   Empty Events: 55
   Hourly Stats: 24 hours
   Avg Occupied Duration: 84.0 minutes
   Peak Empty Hours:
      17:00 - 11 events
      13:00 - 11 events
      09:00 - 11 events
   Current Hour (XX:00) Empty Probability: XX.X%
```

### 3. Run Monitor with Test Data
```bash
python parking_lot_monitor.py
```

**Test the 'a' key:**
- Press **'a'** to display analytics summary
- Should show Space #1 predictions
- Verify peak hours: 09:00, 13:00, 17:00

**Test on-screen overlay:**
- Look for parking Space #1 in the video
- Should display:
  - `P(empty): XX%` (probability)
  - `Avg: 84min` (average duration)
  - `Peak: 17:00,13:00` (top 2 hours)

**Test color-coding:**
- If Space #1 is occupied:
  - High probability (>50%): **Bright green** overlay
  - Medium probability (20-50%): **Yellow-green** overlay
  - Low probability (<20%): **Red** overlay

## Real-World Testing (Multi-Day)

### Day 1: Setup & Initial Data Collection

1. **Define parking spaces:**
   ```bash
   python parking_lot_monitor.py
   ```
   - Draw polygons around all parking spaces
   - Press 's' to save configuration

2. **Run continuously:**
   - Let the system run for at least 8 hours
   - Ensure camera has clear view of parking spaces
   - Monitor console for state changes

3. **Check analytics file:**
   ```bash
   # Should exist and grow over time
   ls -lh parking_analytics.json
   ```

### Day 2-3: Pattern Emergence

4. **Continue monitoring:**
   - Run for multiple days to capture weekly patterns
   - Different patterns emerge on weekdays vs weekends

5. **Analyze patterns daily:**
   ```bash
   python parking_lot_monitor.py
   # Press 'a' periodically to view evolving patterns
   ```

### Day 4-7: Validation

6. **Test predictions:**
   - Note what time system predicts space will be empty
   - Observe actual occupancy changes
   - Compare predicted vs actual times

7. **Validate probability accuracy:**
   - Check `P(empty): XX%` on-screen
   - If shows 75%, space should be empty ~3 out of 4 times at that hour
   - Track accuracy over multiple days

## Testing Checklist

### Functional Tests
- [ ] Test data generates correctly (80 events)
- [ ] Analytics file loads on startup
- [ ] 'a' key displays summary without errors
- [ ] On-screen overlay shows for each space
- [ ] Color-coding changes based on probability
- [ ] Predictions update as new data collected

### Data Integrity Tests
- [ ] Analytics file persists between sessions
- [ ] No data loss after crashes/restarts
- [ ] Events limited to 1000 per type (memory management)
- [ ] Auto-save every 10 events works

### Performance Tests
- [ ] No FPS drop with analytics enabled
- [ ] Console output for 'a' key is fast (<1 second)
- [ ] On-screen overlay doesn't slow rendering
- [ ] File I/O doesn't block video processing

### Visual Tests
- [ ] Green color visible for high probability (>50%)
- [ ] Yellow-green for medium probability (20-50%)
- [ ] Text overlay readable on video
- [ ] Peak hours display correctly (HH:00 format)
- [ ] Probability percentage shows 0-100%

## Expected Results

### After Test Data (Immediate)
```
Space #1:
✅ 80 total events
✅ Peak hours: 09:00, 13:00, 17:00
✅ Avg duration: ~84 minutes
✅ Current hour probability: varies by time
```

### After 24 Hours Real Data
```
Space #1:
✅ 10-30 events collected
✅ Basic hourly patterns emerging
✅ Some peak hours identified
⚠️  Probabilities may fluctuate (limited data)
```

### After 7 Days Real Data
```
Space #1:
✅ 100+ events collected
✅ Clear peak hours identified
✅ Accurate average durations
✅ Reliable hourly probabilities
✅ Day-of-week patterns visible
```

## Troubleshooting

### No Analytics Displayed
**Problem:** Press 'a' shows "No pattern data available yet"
**Solution:** 
- Run test_analytics.py to generate data
- Or wait for real occupancy changes to be recorded

### Wrong Predictions
**Problem:** Peak hours don't match reality
**Solution:**
- Need more data (run longer)
- Delete parking_analytics.json to reset
- Ensure camera view is stable

### On-Screen Overlay Not Visible
**Problem:** No `P(empty)` or `Avg:` text on screen
**Solution:**
- Ensure parking_analytics.json exists
- Verify space IDs match between config and analytics
- Check if space has >0 events in analytics file

### Color Not Changing
**Problem:** Spaces always red, never green
**Solution:**
- Check current hour probability (may be low)
- Verify empty events are being recorded
- Run during peak empty hours identified in analytics

## Data Analysis Tips

### View Raw Data
```bash
# Pretty print analytics
python -m json.tool parking_analytics.json | less
```

### Count Events per Hour
```python
import json
with open("parking_analytics.json") as f:
    data = json.load(f)
    
for space_id, space_data in data.items():
    print(f"\nSpace {space_id} hourly distribution:")
    for hour, stats in sorted(space_data['hourly_stats'].items()):
        total = stats['occupied'] + stats['empty']
        print(f"  {hour:>2}:00 - {total:>3} events")
```

### Reset Analytics
```bash
# Start fresh
rm parking_analytics.json

# Or backup and reset
mv parking_analytics.json parking_analytics.backup.json
```

## Success Criteria

Pattern analytics is **working correctly** when:

1. ✅ Test data generates and displays properly
2. ✅ Real occupancy changes are recorded to JSON
3. ✅ 'a' key shows meaningful statistics
4. ✅ On-screen overlay displays for occupied spaces
5. ✅ Color changes based on probability thresholds
6. ✅ Peak hours align with actual empty patterns (after 3+ days)
7. ✅ Predictions help identify best parking times

## Next Steps

Once testing is complete:

1. **Deploy to production:**
   ```bash
   # Use the deployment package
   cd parking-lot-monitor-deploy/
   ./start.sh  # or start.bat on Windows
   ```

2. **Monitor long-term:**
   - Run continuously for weeks/months
   - Patterns will become more accurate
   - System learns seasonal changes

3. **Extend functionality:**
   - Add database storage (replace JSON)
   - Implement REST API for remote access
   - Create web dashboard for pattern visualization
   - Add email/SMS notifications for predicted empty times
