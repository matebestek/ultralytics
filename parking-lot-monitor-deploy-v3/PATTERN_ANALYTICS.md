# Parking Pattern Analytics

## Overview
The parking lot monitor now includes **pattern learning** to predict when parking spaces are most likely to become empty. This feature analyzes historical occupancy data to identify patterns and provide predictions.

## How It Works

### Data Collection
The system automatically tracks every parking space state change:
- **Timestamp**: When the change occurred
- **Hour of Day**: 0-23 (for time-of-day patterns)
- **Day of Week**: Monday-Sunday (for weekly patterns)
- **Duration**: How long the space was occupied or empty

All data is stored in `parking_analytics.json` and persists between sessions.

### Pattern Analysis
The system calculates:
1. **Average Occupied Duration**: How long cars typically stay in each space
2. **Peak Empty Hours**: The top 3 hours when spaces become empty most often
3. **Current Hour Probability**: Likelihood (0-100%) that the space will be empty during the current hour

### Usage

#### Viewing Analytics
Press **'a'** at any time to display pattern analytics:
```
📊 PARKING PATTERN ANALYTICS
================================================================================
🅿️ Space #1:
   Total Events: 25 occupied, 55 empty events
   Avg Occupied Duration: 78.4 minutes
   Peak Empty Hours: 09:00 (10x), 13:00 (10x), 17:00 (10x), 
   Current Hour Empty Probability: 18.2%

🅿️ Space #2:
   Total Events: 15 occupied, 20 empty events
   Avg Occupied Duration: 45.2 minutes
   Peak Empty Hours: 12:00 (5x), 18:00 (4x), 08:00 (3x), 
   Current Hour Empty Probability: 0.0%
================================================================================
```

#### Understanding Predictions
- **Avg Occupied Duration**: Plan your arrival based on typical parking durations
- **Peak Empty Hours**: Times when you're most likely to find an empty space
- **Current Hour Probability**: Real-time likelihood based on historical data

### Data Storage
- **File**: `parking_analytics.json`
- **Auto-save**: Every 10 events (to avoid excessive disk writes)
- **Retention**: Last 1000 events per event type per space
- **Format**: Human-readable JSON for easy analysis

### Example Use Cases

#### Finding the Best Time to Arrive
"Space #3 typically becomes empty at 9:00 AM, 1:00 PM, and 5:00 PM"
→ Plan your arrival for these peak hours

#### Estimating Wait Times
"Space #1 has an average occupied duration of 45 minutes"
→ If you see a car park at 2:00 PM, it will likely leave by 2:45 PM

#### Current Availability Likelihood
"Current hour empty probability: 75%"
→ High chance of finding this space empty right now

### Privacy & Data
- All data is stored **locally** on your machine
- No external servers or cloud storage
- Data is fully under your control
- Delete `parking_analytics.json` to reset all pattern data

### Tips for Best Results
1. **Run Continuously**: The more data collected, the more accurate predictions become
2. **Monitor Multiple Days**: Weekly patterns emerge after 7+ days of data
3. **Check Analytics Regularly**: Press 'a' to stay informed about patterns
4. **Clean Data**: If parking patterns change (e.g., new tenants), delete old analytics

## Technical Details

### Data Structure
```json
{
  "space_id": {
    "occupancy_events": [
      {
        "timestamp": "2025-01-20T14:30:00",
        "hour": 14,
        "day_of_week": "Monday",
        "duration_seconds": 3600
      }
    ],
    "empty_events": [...],
    "hourly_stats": {
      "9": {"occupied": 5, "empty": 10},
      "14": {"occupied": 8, "empty": 2}
    }
  }
}
```

### Algorithm
- **Duration Calculation**: Time between consecutive state changes
- **Peak Hours**: Top 3 hours sorted by empty event count
- **Probability**: (empty_events_this_hour / total_events_this_hour) × 100%

### Performance
- Minimal CPU overhead (event recording only on state changes)
- Low memory footprint (max 1000 events per type per space)
- Fast analytics computation (O(n) where n = events per space)
