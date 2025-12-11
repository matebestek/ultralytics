"""
Quick test to verify pattern analytics functionality
Run this after generating test_analytics.py data
"""
import json
from datetime import datetime

# Load the analytics data
with open("parking_analytics.json", "r") as f:
    data = json.load(f)

print("="*80)
print("📊 PARKING ANALYTICS TEST VERIFICATION")
print("="*80)

for space_id, space_data in data.items():
    print(f"\n🅿️  Space #{space_id}:")
    print(f"   Occupancy Events: {len(space_data['occupancy_events'])}")
    print(f"   Empty Events: {len(space_data['empty_events'])}")
    print(f"   Hourly Stats: {len(space_data['hourly_stats'])} hours")
    
    # Calculate average durations
    if space_data['occupancy_events']:
        durations = [e['duration_seconds'] for e in space_data['occupancy_events'] if 'duration_seconds' in e]
        if durations:
            avg_duration_min = sum(durations) / len(durations) / 60
            print(f"   Avg Occupied Duration: {avg_duration_min:.1f} minutes")
    
    # Find peak empty hours
    empty_hours = {}
    for event in space_data['empty_events']:
        hour = event['hour']
        empty_hours[hour] = empty_hours.get(hour, 0) + 1
    
    if empty_hours:
        sorted_hours = sorted(empty_hours.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_hours[:3]
        print(f"   Peak Empty Hours:")
        for hour, count in top_3:
            print(f"      {hour:02d}:00 - {count} events")
    
    # Current hour probability
    current_hour = datetime.now().hour
    if str(current_hour) in space_data['hourly_stats']:
        stats = space_data['hourly_stats'][str(current_hour)]
        total = stats['occupied'] + stats['empty']
        if total > 0:
            prob = (stats['empty'] / total) * 100
            print(f"   Current Hour ({current_hour:02d}:00) Empty Probability: {prob:.1f}%")

print("\n" + "="*80)
print("✅ Analytics data structure verified!")
print("\n💡 Now run: python parking_lot_monitor.py")
print("   Then press 'a' to see live analytics display")
print("="*80)
