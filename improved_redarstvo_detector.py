#!/usr/bin/env python3
"""
Improved Redarstvo Uniform Detection System with Debug Features
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import json

class ImprovedRedarstvoDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.uniform_profiles = {}
        self.debug_mode = True
        
    def load_uniform_profiles(self):
        """Load the analyzed uniform profiles."""
        try:
            with open("uniform_config.json", 'r') as f:
                self.uniform_profiles = json.load(f)
            print(f"✅ Loaded {len(self.uniform_profiles)} uniform profiles")
            return True
        except:
            print("❌ No uniform profiles found. Please run analysis first.")
            return False
    
    def extract_person_colors(self, person_region):
        """Extract dominant colors from person region with improved method."""
        if person_region.size == 0:
            return []
        
        # Resize region for faster processing if too large
        if person_region.shape[0] > 200 or person_region.shape[1] > 200:
            person_region = cv2.resize(person_region, (200, 200))
        
        # Focus on torso area (middle portion) for better uniform detection
        h, w = person_region.shape[:2]
        torso_region = person_region[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        if torso_region.size == 0:
            torso_region = person_region
        
        # Extract colors using multiple methods
        colors = []
        
        # Method 1: K-means clustering
        pixels = torso_region.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        if len(pixels) >= 5:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            try:
                _, _, centers = cv2.kmeans(pixels, min(4, len(pixels)//10), None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
                kmeans_colors = np.uint8(centers).tolist()
                colors.extend(kmeans_colors)
            except:
                pass
        
        # Method 2: Sample colors from different regions
        sample_colors = []
        for y in range(0, torso_region.shape[0], 20):
            for x in range(0, torso_region.shape[1], 20):
                if y < torso_region.shape[0] and x < torso_region.shape[1]:
                    color = torso_region[y, x].tolist()
                    sample_colors.append(color)
        
        colors.extend(sample_colors[:10])  # Add up to 10 sample colors
        
        return colors
    
    def analyze_color_distribution(self, person_region):
        """Analyze color distribution in the person region."""
        if person_region.size == 0:
            return {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        
        # Analyze hue distribution
        h_channel = hsv[:, :, 0]
        
        # Count pixels in different color ranges
        color_analysis = {
            'dark_pixels': np.sum((hsv[:, :, 2] < 80)),  # Dark areas (low value)
            'light_pixels': np.sum((hsv[:, :, 2] > 180)),  # Light areas (high value)
            'blue_pixels': np.sum((h_channel >= 100) & (h_channel <= 130) & (hsv[:, :, 1] > 50)),  # Blue hues
            'gray_pixels': np.sum((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 200)),  # Gray areas
            'total_pixels': person_region.shape[0] * person_region.shape[1]
        }
        
        # Calculate percentages
        for key in color_analysis:
            if key != 'total_pixels' and color_analysis['total_pixels'] > 0:
                color_analysis[key + '_percent'] = color_analysis[key] / color_analysis['total_pixels'] * 100
        
        return color_analysis
    
    def improved_uniform_classification(self, person_region):
        """Improved uniform classification with multiple analysis methods."""
        if person_region.size == 0:
            return "No Match", 0.0, {}
        
        # Extract colors and analyze distribution
        person_colors = self.extract_person_colors(person_region)
        color_analysis = self.analyze_color_distribution(person_region)
        
        debug_info = {
            'person_colors_count': len(person_colors),
            'color_analysis': color_analysis,
            'scores': {}
        }
        
        if not person_colors:
            return "No Match", 0.0, debug_info
        
        best_match = "No Match"
        best_score = 0.0
        
        # Check against each uniform profile
        for uniform_name, profile in self.uniform_profiles.items():
            uniform_colors = profile["dominant_colors"]
            
            # Multiple scoring methods
            scores = []
            
            # Score 1: Color similarity
            color_score = self.calculate_color_similarity(person_colors, uniform_colors)
            scores.append(color_score * 0.4)
            
            # Score 2: Color distribution similarity (focus on dark/light balance)
            distribution_score = self.calculate_distribution_similarity(color_analysis)
            scores.append(distribution_score * 0.3)
            
            # Score 3: Redarstvo-specific pattern matching
            pattern_score = self.redarstvo_pattern_matching(color_analysis, person_colors)
            scores.append(pattern_score * 0.3)
            
            total_score = sum(scores)
            
            debug_info['scores'][uniform_name] = {
                'color_score': color_score,
                'distribution_score': distribution_score,
                'pattern_score': pattern_score,
                'total_score': total_score
            }
            
            if total_score > best_score:
                best_match = uniform_name
                best_score = total_score
        
        # Lower threshold for better detection
        if best_score > 0.25:  # Reduced from 0.4
            return best_match, best_score, debug_info
        else:
            return "No Match", best_score, debug_info
    
    def calculate_color_similarity(self, person_colors, uniform_colors):
        """Calculate similarity between person colors and uniform colors."""
        if not person_colors or not uniform_colors:
            return 0.0
        
        max_similarities = []
        
        for p_color in person_colors:
            best_similarity = 0.0
            for u_color in uniform_colors:
                # Calculate color distance in RGB space
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p_color, u_color)))
                similarity = max(0, 1 - distance / (255 * np.sqrt(3)))
                best_similarity = max(best_similarity, similarity)
            max_similarities.append(best_similarity)
        
        # Return average of top similarities
        max_similarities.sort(reverse=True)
        top_similarities = max_similarities[:min(5, len(max_similarities))]
        return sum(top_similarities) / len(top_similarities) if top_similarities else 0.0
    
    def calculate_distribution_similarity(self, color_analysis):
        """Calculate how well the color distribution matches uniform patterns."""
        # Redarstvo uniforms typically have:
        # - Significant dark areas (dark clothing)
        # - Some light areas (badges, stripes, reflective elements)
        # - Moderate gray areas
        
        score = 0.0
        
        # Check for dark areas (should be substantial for uniforms)
        if 'dark_pixels_percent' in color_analysis:
            dark_percent = color_analysis['dark_pixels_percent']
            if 20 <= dark_percent <= 70:  # Good range for uniform dark areas
                score += 0.3
        
        # Check for light accents (badges, reflective strips)
        if 'light_pixels_percent' in color_analysis:
            light_percent = color_analysis['light_pixels_percent']
            if 5 <= light_percent <= 30:  # Some but not too many light areas
                score += 0.2
        
        # Check for gray areas (uniform fabric often appears grayish)
        if 'gray_pixels_percent' in color_analysis:
            gray_percent = color_analysis['gray_pixels_percent']
            if gray_percent >= 10:  # Some gray areas expected
                score += 0.2
        
        # Bonus for balanced distribution (not all one color)
        total_colored = color_analysis.get('dark_pixels_percent', 0) + \
                       color_analysis.get('light_pixels_percent', 0) + \
                       color_analysis.get('gray_pixels_percent', 0)
        
        if 40 <= total_colored <= 80:  # Good color distribution
            score += 0.3
        
        return min(score, 1.0)
    
    def redarstvo_pattern_matching(self, color_analysis, person_colors):
        """Specific pattern matching for Redarstvo uniforms."""
        score = 0.0
        
        # Look for specific Redarstvo color combinations from the analysis
        redarstvo_indicators = {
            'has_dark_blue': False,
            'has_dark_clothing': False,
            'has_light_accents': False,
            'has_gray_elements': False
        }
        
        # Check person colors for Redarstvo-specific patterns
        for color in person_colors:
            b, g, r = color
            
            # Dark blue check (based on your uniform analysis: RGB(55,48,147))
            if b > 100 and g < 80 and r < 80:
                redarstvo_indicators['has_dark_blue'] = True
                score += 0.25
            
            # Dark clothing check
            if b < 50 and g < 50 and r < 50:
                redarstvo_indicators['has_dark_clothing'] = True
                score += 0.2
            
            # Light accents check (badges, reflective elements)
            if b > 200 and g > 200 and r > 200:
                redarstvo_indicators['has_light_accents'] = True
                score += 0.15
            
            # Gray elements check
            if abs(b - g) < 20 and abs(g - r) < 20 and 80 < b < 180:
                redarstvo_indicators['has_gray_elements'] = True
                score += 0.1
        
        # Bonus for having multiple indicators
        indicator_count = sum(redarstvo_indicators.values())
        if indicator_count >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def run_detection_with_debug(self):
        """Run detection with enhanced debugging."""
        if not self.load_uniform_profiles():
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎯 Enhanced Redarstvo Uniform Detection Active!")
        print("Press 'q' to quit | 's' to save screenshot | 'd' to toggle debug")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            
            redarstvo_count = 0
            person_count = 0
            debug_text_y = 100
            
            if detections is not None:
                for i, box in enumerate(detections.data):
                    x1, y1, x2, y2, conf, class_id = box
                    class_name = self.model.names[int(class_id)]
                    
                    if class_name == "person" and conf > 0.4:  # Lowered threshold
                        person_count += 1
                        
                        # Extract person region
                        person_region = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Classify with debug info
                        uniform_type, uniform_confidence, debug_info = self.improved_uniform_classification(person_region)
                        
                        # Determine detection result
                        is_redarstvo = uniform_confidence > 0.25 and "Redarstvo" in uniform_type
                        
                        if is_redarstvo:
                            color = (0, 0, 255)  # Red for Redarstvo
                            label = f"🚨 REDARSTVO ({uniform_confidence:.2f})"
                            redarstvo_count += 1
                            thickness = 3
                        else:
                            color = (0, 255, 0)  # Green for regular person
                            label = f"Person ({conf:.2f})"
                            thickness = 2
                        
                        # Draw detection
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                        
                        # Label with background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (int(x1), int(y1) - 35), 
                                    (int(x1) + label_size[0] + 10, int(y1)), color, -1)
                        cv2.putText(frame, label, (int(x1) + 5, int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Debug information
                        if self.debug_mode and uniform_confidence > 0.1:
                            debug_text = f"P{i+1}: Conf={uniform_confidence:.3f}"
                            cv2.putText(frame, debug_text, (400, debug_text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            debug_text_y += 20
                            
                            # Show color analysis
                            if 'color_analysis' in debug_info:
                                ca = debug_info['color_analysis']
                                dark_pct = ca.get('dark_pixels_percent', 0)
                                light_pct = ca.get('light_pixels_percent', 0)
                                debug_detail = f"  Dark:{dark_pct:.1f}% Light:{light_pct:.1f}%"
                                cv2.putText(frame, debug_detail, (400, debug_text_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                                debug_text_y += 15
            
            # Status display
            status_color = (0, 0, 255) if redarstvo_count > 0 else (0, 255, 0)
            cv2.putText(frame, f"🚨 REDARSTVO DETECTED: {redarstvo_count}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"People: {person_count} | Debug: {'ON' if self.debug_mode else 'OFF'}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press: q=quit | s=save | d=debug toggle", 
                       (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Enhanced Redarstvo Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"redarstvo_debug_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

def main():
    detector = ImprovedRedarstvoDetector()
    print("=== Enhanced Redarstvo Uniform Detection ===")
    print("Starting improved detection with debug features...")
    detector.run_detection_with_debug()

if __name__ == "__main__":
    main()