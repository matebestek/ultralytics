"""Test script to verify parking pattern analytics."""
import json
from datetime import datetime, timedelta

# Create sample analytics data
analytics_data = {}

# Simulate 50 events for space #1 over the past week
base_time = datetime.now() - timedelta(days=7)

for i in range(50):
    # Simulate occupied events mostly during work hours (8-18)
    event_time = base_time + timedelta(hours=i*3, minutes=i*7)
    
    space_id = "1"
    if space_id not in analytics_data:
        analytics_data[space_id] = {
            "occupancy_events": [],
            "empty_events": [],
            "hourly_stats": {},
            "durations": []
        }
    
    # Alternate between occupied and empty events
    if i % 2 == 0:
        # Occupied event
        analytics_data[space_id]["occupancy_events"].append({
            "timestamp": event_time.isoformat(),
            "hour": event_time.hour,
            "day_of_week": event_time.strftime("%A"),
            "duration_seconds": 3600 + (i * 60)  # 1+ hours
        })
        
        # Update hourly stats
        hour_key = str(event_time.hour)
        if hour_key not in analytics_data[space_id]["hourly_stats"]:
            analytics_data[space_id]["hourly_stats"][hour_key] = {"occupied": 0, "empty": 0}
        analytics_data[space_id]["hourly_stats"][hour_key]["occupied"] += 1
    else:
        # Empty event
        analytics_data[space_id]["empty_events"].append({
            "timestamp": event_time.isoformat(),
            "hour": event_time.hour,
            "day_of_week": event_time.strftime("%A"),
            "duration_seconds": 1800 + (i * 30)  # 30+ minutes
        })
        
        # Update hourly stats
        hour_key = str(event_time.hour)
        if hour_key not in analytics_data[space_id]["hourly_stats"]:
            analytics_data[space_id]["hourly_stats"][hour_key] = {"occupied": 0, "empty": 0}
        analytics_data[space_id]["hourly_stats"][hour_key]["empty"] += 1

# Add more events concentrated at specific hours (9 AM, 1 PM, 5 PM)
for peak_hour in [9, 13, 17]:
    for _ in range(10):
        event_time = base_time + timedelta(hours=peak_hour, minutes=_*5)
        
        analytics_data[space_id]["empty_events"].append({
            "timestamp": event_time.isoformat(),
            "hour": peak_hour,
            "day_of_week": event_time.strftime("%A"),
            "duration_seconds": 600
        })
        
        hour_key = str(peak_hour)
        if hour_key not in analytics_data[space_id]["hourly_stats"]:
            analytics_data[space_id]["hourly_stats"][hour_key] = {"occupied": 0, "empty": 0}
        analytics_data[space_id]["hourly_stats"][hour_key]["empty"] += 1

# Save to file
with open("parking_analytics.json", "w") as f:
    json.dump(analytics_data, f, indent=2)

print("✅ Test analytics data created!")
print(f"   Space #1: {len(analytics_data['1']['occupancy_events'])} occupied, {len(analytics_data['1']['empty_events'])} empty events")
print(f"   Hourly stats: {len(analytics_data['1']['hourly_stats'])} hours tracked")
print("\n📊 Peak empty hours should be: 09:00, 13:00, 17:00")
print("\n💡 Now run parking_lot_monitor.py and press 'a' to see analytics!")
