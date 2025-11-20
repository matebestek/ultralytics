#!/usr/bin/env python3
"""
Direct uniform analysis and detection script
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import json

def analyze_uniform_image(image_path, uniform_name):
    """Analyze a uniform image and extract characteristics."""
    print(f"Analyzing {uniform_name}: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    # Extract dominant colors using K-means
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_colors = np.uint8(centers).tolist()
    
    # Analyze HSV distribution
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    h_hist = (h_hist.flatten() / h_hist.sum()).tolist()
    s_hist = (s_hist.flatten() / s_hist.sum()).tolist()
    v_hist = (v_hist.flatten() / v_hist.sum()).tolist()
    
    profile = {
        "name": uniform_name,
        "dominant_colors": dominant_colors,
        "hsv_histogram": {
            "hue": h_hist,
            "saturation": s_hist,
            "value": v_hist
        }
    }
    
    print(f"✓ Analyzed {uniform_name} - Dominant colors: {len(dominant_colors)}")
    return profile

def analyze_all_uniforms():
    """Analyze all uniform images in the reference folder."""
    ref_dir = "uniform_references"
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
        image_files.extend(Path(ref_dir).glob(ext))
    
    print(f"Found {len(image_files)} uniform images:")
    for img in image_files:
        print(f"  - {img.name}")
    
    uniform_profiles = {}
    
    for i, img_file in enumerate(image_files):
        uniform_name = f"Redarstvo_Uniform_{i+1}"
        profile = analyze_uniform_image(str(img_file), uniform_name)
        if profile:
            uniform_profiles[uniform_name] = profile
    
    # Save profiles
    with open("uniform_config.json", 'w') as f:
        json.dump(uniform_profiles, f, indent=2)
    
    print(f"\n✓ Created {len(uniform_profiles)} uniform profiles!")
    print("✓ Configuration saved to uniform_config.json")
    
    return uniform_profiles

def classify_person_uniform(person_region, uniform_profiles):
    """Classify detected person against uniform profiles."""
    if person_region.size == 0 or not uniform_profiles:
        return "No Match", 0.0
    
    # Extract person's dominant colors
    pixels = person_region.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    if len(pixels) < 5:
        return "No Match", 0.0
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, _, centers = cv2.kmeans(pixels, min(3, len(pixels)), None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        person_colors = np.uint8(centers).tolist()
    except:
        return "No Match", 0.0
    
    best_match = "No Match"
    best_score = 0.0
    
    for uniform_name, profile in uniform_profiles.items():
        score = compare_colors(person_colors, profile["dominant_colors"])
        
        if score > best_score:
            best_match = uniform_name
            best_score = score
    
    return best_match, best_score

def compare_colors(colors1, colors2):
    """Compare two sets of colors and return similarity score."""
    if not colors1 or not colors2:
        return 0.0
    
    max_similarity = 0.0
    
    for c1 in colors1:
        for c2 in colors2:
            # Calculate color distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
            similarity = max(0, 1 - distance / (255 * np.sqrt(3)))
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def run_detection():
    """Run real-time uniform detection."""
    # Load uniform profiles
    try:
        with open("uniform_config.json", 'r') as f:
            uniform_profiles = json.load(f)
        print(f"Loaded {len(uniform_profiles)} uniform profiles")
    except:
        print("No uniform profiles found. Run analysis first.")
        return
    
    # Load YOLO model
    model = YOLO("yolo11n.pt")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🎯 Redarstvo Uniform Detection Active!")
    print("Press 'q' to quit | 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        redarstvo_count = 0
        person_count = 0
        
        if detections is not None:
            for box in detections.data:
                x1, y1, x2, y2, conf, class_id = box
                class_name = model.names[int(class_id)]
                
                if class_name == "person" and conf > 0.5:
                    person_count += 1
                    
                    # Extract person region
                    person_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Classify uniform
                    uniform_type, uniform_confidence = classify_person_uniform(person_region, uniform_profiles)
                    
                    if uniform_confidence > 0.4 and "Redarstvo" in uniform_type:
                        color = (0, 0, 255)  # Red for Redarstvo uniform
                        label = f"🚨 REDARSTVO UNIFORM ({uniform_confidence:.2f})"
                        redarstvo_count += 1
                        thickness = 3
                    else:
                        color = (0, 255, 0)  # Green for regular person
                        label = f"Person ({conf:.2f})"
                        thickness = 2
                    
                    # Draw detection
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - 35), 
                                (int(x1) + label_size[0] + 10, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1) + 5, int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status display
        status_color = (0, 0, 255) if redarstvo_count > 0 else (0, 255, 0)
        cv2.putText(frame, f"🚨 REDARSTVO DETECTED: {redarstvo_count}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Total People: {person_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to save", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Redarstvo Uniform Detection', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"redarstvo_detection_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

def main():
    print("=== Redarstvo Uniform Detection System ===\n")
    
    choice = input("Choose: (1) Analyze uniform images | (2) Run detection | (q) Quit: ").strip()
    
    if choice == '1':
        print("\n🔍 Analyzing Redarstvo uniform images...")
        uniform_profiles = analyze_all_uniforms()
        
        if uniform_profiles:
            print(f"\n✅ Successfully analyzed {len(uniform_profiles)} uniform types!")
            print("You can now run detection (option 2)")
            
            # Show color analysis
            print("\n📊 Uniform Analysis Results:")
            for name, profile in uniform_profiles.items():
                colors = profile["dominant_colors"]
                print(f"{name}: {len(colors)} dominant colors detected")
                for i, color in enumerate(colors[:3]):  # Show first 3 colors
                    print(f"  Color {i+1}: RGB{tuple(color)}")
        else:
            print("❌ No uniform images found or analysis failed")
            
    elif choice == '2':
        print("\n🎯 Starting Redarstvo uniform detection...")
        run_detection()
        
    elif choice == 'q':
        print("Goodbye!")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()