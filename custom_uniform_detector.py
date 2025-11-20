#!/usr/bin/env python3
"""
Custom Uniform Training and Detection System

This script allows you to:
1. Analyze reference uniform images to extract color and feature patterns
2. Train a custom uniform detector based on your specific uniform types
3. Run real-time detection optimized for your uniform specifications

Usage:
1. Place your reference uniform images in a 'uniform_references' folder
2. Run this script to analyze the images and create a custom detector
3. Use the trained detector for real-time webcam detection

Requirements:
    - ultralytics
    - opencv-python
    - numpy
    - A working webcam/camera
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import json
from typing import Dict, List, Tuple, Any


class CustomUniformDetector:
    def __init__(self):
        """Initialize the custom uniform detection system."""
        self.model = YOLO("yolo11n.pt")
        self.uniform_profiles = {}
        self.reference_dir = "uniform_references"
        self.config_file = "uniform_config.json"
        
    def create_reference_directory(self):
        """Create directory for reference images if it doesn't exist."""
        Path(self.reference_dir).mkdir(exist_ok=True)
        print(f"Created/verified directory: {self.reference_dir}")
        print(f"Please place your 3 JPEG uniform reference images in the '{self.reference_dir}' folder")
        return self.reference_dir
    
    def analyze_reference_image(self, image_path: str, uniform_name: str) -> Dict[str, Any]:
        """Analyze a reference uniform image to extract characteristics."""
        print(f"Analyzing {uniform_name}: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return {}
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract dominant colors
        dominant_colors = self.extract_dominant_colors(image, num_colors=5)
        
        # Analyze color distribution in HSV
        hsv_histogram = self.analyze_hsv_distribution(hsv)
        
        # Extract texture features (simplified)
        texture_features = self.extract_texture_features(image)
        
        # Analyze color uniformity
        color_uniformity = self.analyze_color_uniformity(image)
        
        profile = {
            "name": uniform_name,
            "image_path": image_path,
            "dominant_colors": dominant_colors,
            "hsv_histogram": hsv_histogram,
            "texture_features": texture_features,
            "color_uniformity": color_uniformity,
            "image_dimensions": image.shape[:2]
        }
        
        print(f"✓ Analysis complete for {uniform_name}")
        return profile
    
    def extract_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the image using K-means clustering."""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and return as list of tuples
        centers = np.uint8(centers)
        dominant_colors = [tuple(map(int, color)) for color in centers]
        
        return dominant_colors
    
    def analyze_hsv_distribution(self, hsv_image: np.ndarray) -> Dict[str, List[float]]:
        """Analyze HSV color distribution."""
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / h_hist.sum()
        s_hist = s_hist.flatten() / s_hist.sum()
        v_hist = v_hist.flatten() / v_hist.sum()
        
        return {
            "hue": h_hist.tolist(),
            "saturation": s_hist.tolist(),
            "value": v_hist.tolist()
        }
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic texture features from the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude (texture indicator)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture statistics
        texture_mean = np.mean(gradient_magnitude)
        texture_std = np.std(gradient_magnitude)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        return {
            "texture_mean": float(texture_mean),
            "texture_std": float(texture_std),
            "edge_density": float(edge_density)
        }
    
    def analyze_color_uniformity(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze how uniform the colors are in the image."""
        # Convert to LAB color space for perceptual color difference
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color variance
        color_variance = np.var(lab.reshape(-1, 3), axis=0)
        mean_variance = np.mean(color_variance)
        
        # Calculate color range
        color_range = np.max(lab.reshape(-1, 3), axis=0) - np.min(lab.reshape(-1, 3), axis=0)
        mean_range = np.mean(color_range)
        
        return {
            "color_variance": float(mean_variance),
            "color_range": float(mean_range),
            "uniformity_score": float(1.0 / (1.0 + mean_variance))  # Higher score = more uniform
        }
    
    def load_and_analyze_references(self) -> bool:
        """Load and analyze all reference images in the directory."""
        ref_path = Path(self.reference_dir)
        
        if not ref_path.exists():
            print(f"Reference directory '{self.reference_dir}' not found. Creating it...")
            self.create_reference_directory()
            return False
        
        # Find all JPEG images
        image_files = list(ref_path.glob("*.jpg")) + list(ref_path.glob("*.jpeg")) + list(ref_path.glob("*.JPG")) + list(ref_path.glob("*.JPEG"))
        
        if len(image_files) == 0:
            print(f"No JPEG images found in '{self.reference_dir}' directory.")
            print("Please add your uniform reference images and run again.")
            return False
        
        print(f"Found {len(image_files)} reference images:")
        for img_file in image_files:
            print(f"  - {img_file.name}")
        
        # Analyze each image
        self.uniform_profiles = {}
        for i, img_file in enumerate(image_files):
            uniform_name = f"Uniform_Type_{i+1}"
            profile = self.analyze_reference_image(str(img_file), uniform_name)
            if profile:
                self.uniform_profiles[uniform_name] = profile
        
        # Save configuration
        self.save_uniform_config()
        
        print(f"\n✓ Analysis complete! Created {len(self.uniform_profiles)} uniform profiles.")
        return True
    
    def save_uniform_config(self):
        """Save uniform profiles to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        config_data = {}
        for name, profile in self.uniform_profiles.items():
            config_data[name] = {
                "name": profile["name"],
                "dominant_colors": profile["dominant_colors"],
                "hsv_histogram": profile["hsv_histogram"],
                "texture_features": profile["texture_features"],
                "color_uniformity": profile["color_uniformity"],
                "image_dimensions": profile["image_dimensions"]
            }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✓ Uniform configuration saved to {self.config_file}")
    
    def load_uniform_config(self) -> bool:
        """Load uniform profiles from JSON file."""
        if not os.path.exists(self.config_file):
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.uniform_profiles = config_data
            print(f"✓ Loaded {len(self.uniform_profiles)} uniform profiles from {self.config_file}")
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def classify_uniform_type(self, person_region: np.ndarray) -> Tuple[str, float]:
        """Classify the detected person's clothing against reference uniforms."""
        if not self.uniform_profiles or person_region.size == 0:
            return "Unknown", 0.0
        
        # Extract features from the person region
        region_colors = self.extract_dominant_colors(person_region, num_colors=3)
        region_hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        region_hsv_hist = self.analyze_hsv_distribution(region_hsv)
        region_uniformity = self.analyze_color_uniformity(person_region)
        
        best_match = "No Uniform Detected"
        best_score = 0.0
        
        for uniform_name, profile in self.uniform_profiles.items():
            score = self.calculate_uniform_similarity(
                region_colors, region_hsv_hist, region_uniformity,
                profile["dominant_colors"], profile["hsv_histogram"], profile["color_uniformity"]
            )
            
            if score > best_score:
                best_match = uniform_name
                best_score = score
        
        # Only consider it a match if score is above threshold
        if best_score > 0.6:  # Adjust threshold as needed
            return best_match, best_score
        else:
            return "No Uniform Detected", best_score
    
    def calculate_uniform_similarity(self, region_colors, region_hsv_hist, region_uniformity,
                                   ref_colors, ref_hsv_hist, ref_uniformity) -> float:
        """Calculate similarity between detected region and reference uniform."""
        
        # Color similarity (compare dominant colors)
        color_score = self.compare_color_lists(region_colors, ref_colors)
        
        # HSV histogram similarity
        hsv_score = self.compare_histograms(region_hsv_hist, ref_hsv_hist)
        
        # Uniformity similarity
        uniformity_score = self.compare_uniformity(region_uniformity, ref_uniformity)
        
        # Weighted combination
        total_score = (color_score * 0.4 + hsv_score * 0.4 + uniformity_score * 0.2)
        
        return total_score
    
    def compare_color_lists(self, colors1: List[Tuple[int, int, int]], 
                           colors2: List[Tuple[int, int, int]]) -> float:
        """Compare two lists of colors and return similarity score."""
        if not colors1 or not colors2:
            return 0.0
        
        max_similarity = 0.0
        
        for c1 in colors1:
            for c2 in colors2:
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
                # Convert to similarity (0-1, where 1 is identical)
                similarity = max(0, 1 - distance / (255 * np.sqrt(3)))
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def compare_histograms(self, hist1: Dict[str, List[float]], 
                          hist2: Dict[str, List[float]]) -> float:
        """Compare HSV histograms and return similarity score."""
        try:
            h_corr = cv2.compareHist(np.array(hist1["hue"], dtype=np.float32), 
                                   np.array(hist2["hue"], dtype=np.float32), 
                                   cv2.HISTCMP_CORREL)
            s_corr = cv2.compareHist(np.array(hist1["saturation"], dtype=np.float32), 
                                   np.array(hist2["saturation"], dtype=np.float32), 
                                   cv2.HISTCMP_CORREL)
            v_corr = cv2.compareHist(np.array(hist1["value"], dtype=np.float32), 
                                   np.array(hist2["value"], dtype=np.float32), 
                                   cv2.HISTCMP_CORREL)
            
            # Average correlation (normalized to 0-1)
            avg_corr = (max(0, h_corr) + max(0, s_corr) + max(0, v_corr)) / 3
            return avg_corr
        except:
            return 0.0
    
    def compare_uniformity(self, uniformity1: Dict[str, float], 
                          uniformity2: Dict[str, float]) -> float:
        """Compare uniformity characteristics."""
        try:
            score1 = uniformity1["uniformity_score"]
            score2 = uniformity2["uniformity_score"]
            
            # Similarity based on how close the uniformity scores are
            similarity = 1 - abs(score1 - score2)
            return max(0, similarity)
        except:
            return 0.0
    
    def run_detection(self):
        """Run real-time uniform detection using the trained profiles."""
        if not self.uniform_profiles:
            print("No uniform profiles loaded. Please analyze reference images first.")
            return
        
        print(f"\nStarting custom uniform detection...")
        print(f"Loaded {len(self.uniform_profiles)} uniform types:")
        for name in self.uniform_profiles.keys():
            print(f"  - {name}")
        print("\nPress 'q' to quit | Press 's' to save screenshot")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            
            uniform_detections = 0
            person_count = 0
            
            if detections is not None:
                for box in detections.data:
                    x1, y1, x2, y2, conf, class_id = box
                    class_name = self.model.names[int(class_id)]
                    
                    if class_name == "person" and conf > 0.5:
                        person_count += 1
                        
                        # Extract person region
                        person_region = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Classify uniform type
                        uniform_type, uniform_confidence = self.classify_uniform_type(person_region)
                        
                        # Determine box color and label
                        if uniform_type != "No Uniform Detected" and uniform_confidence > 0.6:
                            color = (0, 0, 255)  # Red for uniform detection
                            label = f"UNIFORM: {uniform_type} ({uniform_confidence:.2f})"
                            uniform_detections += 1
                        else:
                            color = (0, 255, 0)  # Green for person without uniform
                            label = f"Person ({conf:.2f})"
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Label background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (int(x1), int(y1) - 30), 
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(frame, label, (int(x1), int(y1) - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add status information
            cv2.putText(frame, f"People: {person_count} | Uniforms: {uniform_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit | 's' to save", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Custom Uniform Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"uniform_detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")


def main():
    """Main function."""
    detector = CustomUniformDetector()
    
    print("=== Custom Uniform Detection System ===")
    print("\nOptions:")
    print("1. Analyze new reference images")
    print("2. Run detection with existing profiles")
    print("3. Show current uniform profiles")
    
    # Try to load existing configuration
    config_loaded = detector.load_uniform_config()
    
    if not config_loaded:
        print("\nNo existing uniform profiles found.")
        print("Creating reference directory for your uniform images...")
        detector.create_reference_directory()
        print("\nPlease:")
        print("1. Place your 3 JPEG uniform reference images in the 'uniform_references' folder")
        print("2. Run this script again to analyze them")
        return
    
    while True:
        choice = input("\nEnter your choice (1-3, or 'q' to quit): ").strip()
        
        if choice == 'q':
            break
        elif choice == '1':
            print("\nAnalyzing reference images...")
            if detector.load_and_analyze_references():
                print("✓ Reference analysis complete!")
            else:
                print("Please add reference images and try again.")
        elif choice == '2':
            detector.run_detection()
        elif choice == '3':
            print(f"\nCurrent uniform profiles ({len(detector.uniform_profiles)}):")
            for name, profile in detector.uniform_profiles.items():
                print(f"  - {name}: {len(profile['dominant_colors'])} colors analyzed")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 'q'.")


if __name__ == "__main__":
    main()