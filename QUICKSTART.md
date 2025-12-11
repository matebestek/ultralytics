# 🚀 Quick Start - Pattern Analytics

## 60-Second Test

```bash
# 1. Generate test data (5 seconds)
python test_analytics.py

# 2. Verify it worked (2 seconds)
python verify_analytics.py

# 3. Run the monitor (starts immediately)
python parking_lot_monitor.py
```

**Once running:**
- Press **'a'** to see analytics summary in console
- Look at video feed - you'll see pattern predictions on-screen!

## What You'll See

### In Console (Press 'a'):
```
================================================================================
📊 PARKING PATTERN ANALYTICS
================================================================================
🅿️ Space #1:
   Total Events: 25 occupied, 55 empty events
   Avg Occupied Duration: 84.0 minutes
   Peak Empty Hours: 09:00 (11x), 13:00 (11x), 17:00 (11x), 
   Current Hour Empty Probability: 50.0%
================================================================================
```

### On Video Feed:
Each parking space shows:
```
#1                          ← Space number
OCCUPIED (LIKELY EMPTY SOON)  ← Status
P(empty): 75%               ← Probability  
Avg: 45min                  ← Average duration
Peak: 09:00,17:00           ← Best times
```

### Color Coding:
- 🟢 **Bright Green** = >50% likely to empty soon
- 🟡 **Yellow-Green** = 20-50% chance
- 🔴 **Red** = <20% chance

## Real Data Collection

Want real patterns? Just run normally:
```bash
python parking_lot_monitor.py
```

The system automatically records every parking space change:
- ✅ No setup needed
- ✅ Saves every 10 events
- ✅ Persists between sessions
- ✅ Learns patterns over time

**After 1 day:** Basic patterns emerge
**After 3 days:** Reliable predictions
**After 7 days:** Accurate peak hours

## Common Commands

```bash
# Start monitoring
python parking_lot_monitor.py

# View analytics (while running)
Press 'a'

# Reset pattern data (start fresh)
rm parking_analytics.json

# Backup analytics
cp parking_analytics.json backup_$(date +%Y%m%d).json
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| **a** | Show analytics summary |
| **q** | Quit |
| **p** | Toggle setup mode |
| **s** | Save configuration |
| **d** | Toggle debug info |
| **n** | Toggle notifications |

## Troubleshooting

**No analytics displayed?**
```bash
# Generate test data first
python test_analytics.py
```

**Want to see raw data?**
```bash
# View JSON (pretty printed)
python -m json.tool parking_analytics.json | less
```

**Reset everything?**
```bash
# Delete all data files
rm parking_*.json
```

## File Locations

- 📊 **parking_analytics.json** - Pattern learning data
- 📝 **parking_config.json** - Space definitions
- 📋 **parking_log.json** - Event logs

## Success Check ✅

Pattern analytics is working if:
1. ✅ Press 'a' shows statistics
2. ✅ On-screen text appears over spaces
3. ✅ Green/yellow/red colors change based on probability
4. ✅ `parking_analytics.json` file exists and grows

## Next Steps

1. **Test with simulated data** (what you just did!)
2. **Run with real camera** for actual learning
3. **Monitor for 3-7 days** for accurate patterns
4. **Deploy to production** using the deployment package

---

**That's it!** The system is now learning parking patterns and predicting availability. 🎉
