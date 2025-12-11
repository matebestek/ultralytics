"""
Visual demonstration of pattern analytics color coding
Shows how probability affects visual display
"""
print("\n" + "="*80)
print("🎨 PATTERN ANALYTICS - COLOR CODING DEMONSTRATION")
print("="*80)

scenarios = [
    {
        "space": 1,
        "status": "OCCUPIED",
        "probability": 85,
        "avg_duration": 45,
        "peak_hours": ["09:00", "17:00"],
        "color": "🟢 BRIGHT GREEN",
        "meaning": "Very likely to empty soon!"
    },
    {
        "space": 2,
        "status": "OCCUPIED",
        "probability": 35,
        "avg_duration": 72,
        "peak_hours": ["12:00", "18:00"],
        "color": "🟡 YELLOW-GREEN",
        "meaning": "Moderate chance to empty"
    },
    {
        "space": 3,
        "status": "OCCUPIED",
        "probability": 12,
        "avg_duration": 120,
        "peak_hours": ["10:00", "15:00"],
        "color": "🔴 RED",
        "meaning": "Low chance to empty soon"
    },
    {
        "space": 4,
        "status": "EMPTY",
        "probability": 50,
        "avg_duration": 30,
        "peak_hours": ["08:00", "16:00"],
        "color": "⚪ WHITE (default empty color)",
        "meaning": "Available now!"
    }
]

for scenario in scenarios:
    print(f"\n{'─'*80}")
    print(f"🅿️  SPACE #{scenario['space']}")
    print(f"{'─'*80}")
    print(f"Status: {scenario['status']}")
    print(f"Color:  {scenario['color']}")
    print(f"Meaning: {scenario['meaning']}")
    print(f"\n📊 On-Screen Display:")
    print(f"   #{scenario['space']}")
    print(f"   {scenario['status']}")
    if scenario['status'] == "OCCUPIED":
        print(f"   P(empty): {scenario['probability']}%")
        print(f"   Avg: {scenario['avg_duration']}min")
        print(f"   Peak: {','.join(scenario['peak_hours'])}")

print(f"\n{'═'*80}")
print("💡 KEY INSIGHTS:")
print("═"*80)
print("✅ Space #1: GREEN = Park here if waiting, likely to empty soon!")
print("⚠️  Space #2: YELLOW = Maybe wait, moderate chance")
print("❌ Space #3: RED = Don't wait, low probability of emptying")
print("✨ Space #4: Already empty, grab it now!")

print("\n" + "="*80)
print("🎯 PROBABILITY THRESHOLDS")
print("="*80)
print("• >50%  → 🟢 Bright Green  → 'LIKELY EMPTY SOON'")
print("• 20-50% → 🟡 Yellow-Green → 'MAY EMPTY SOON'")
print("• <20%  → 🔴 Red          → 'OCCUPIED'")
print("• Empty → ⚪ White        → 'EMPTY'")

print("\n" + "="*80)
print("📈 HOW PROBABILITY IS CALCULATED")
print("="*80)
print("For current hour (e.g., 2:00 PM):")
print("1. Count all times space became empty at 2:00 PM = 15")
print("2. Count all events at 2:00 PM (occupied + empty) = 20")
print("3. Probability = (15 / 20) × 100% = 75%")
print("4. If >50% → Display GREEN color")

print("\n" + "="*80)
print("🚗 REAL-WORLD EXAMPLE")
print("="*80)
print("It's 2:00 PM. You're looking for parking...")
print()
print("Space #1: 🟢 GREEN  → P(empty): 75% → WAIT HERE!")
print("          History shows this space empties 75% of the time at 2 PM")
print()
print("Space #2: 🔴 RED    → P(empty): 10% → DON'T WAIT")
print("          History shows this space rarely empties at 2 PM")
print()
print("Space #3: 🟡 YELLOW → P(empty): 40% → MAYBE")
print("          50/50 chance, your call")

print("\n" + "="*80)
print("✅ Now run: python parking_lot_monitor.py")
print("   Press 'a' to see real analytics")
print("   Watch the colors change based on probability!")
print("="*80 + "\n")
