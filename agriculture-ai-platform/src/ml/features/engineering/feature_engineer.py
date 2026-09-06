import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_color_features(self, image: Image.Image) -> np.ndarray:
        """Extract color-based features"""
        img_array = np.array(image)
        
        features = []
        
        # Mean and std for each channel
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        # Color ratios
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        total = r + g + b + 1e-6
        features.extend([
            np.mean(r / total),
            np.mean(g / total),
            np.mean(b / total)
        ])
        
        # HSV features
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        for channel in range(3):
            channel_data = hsv[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        return np.array(features)
    
    def extract_texture_features(self, image: Image.Image) -> np.ndarray:
        """Extract texture features using GLCM and LBP"""
        gray = image.convert('L')
        img_array = np.array(gray)
        
        features = []
        
        # GLCM features
        glcm = self.calculate_glcm(img_array)
        glcm_features = self.calculate_glcm_features(glcm)
        features.extend(glcm_features)
        
        # LBP features
        lbp = self.calculate_lbp(img_array)
        lbp_hist, _ = np.histogram(lbp, bins=32, density=True)
        features.extend(lbp_hist)
        
        return np.array(features)
    
    def calculate_glcm(self, image: np.ndarray, distances: List[int] = [1], angles: List[int] = [0]) -> np.ndarray:
        """Calculate Gray Level Co-occurrence Matrix"""
        max_val = image.max() + 1
        glcm = np.zeros((max_val, max_val), dtype=np.uint32)
        
        for d in distances:
            for angle in angles:
                if angle == 0:
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1] - d):
                            glcm[image[i, j], image[i, j + d]] += 1
                elif angle == 90:
                    for i in range(image.shape[0] - d):
                        for j in range(image.shape[1]):
                            glcm[image[i, j], image[i + d, j]] += 1
        
        # Normalize
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
        
        return glcm
    
    def calculate_glcm_features(self, glcm: np.ndarray) -> List[float]:
        """Calculate features from GLCM"""
        features = []
        
        i, j = np.meshgrid(range(glcm.shape[0]), range(glcm.shape[1]))
        
        # Contrast
        contrast = np.sum((i - j) ** 2 * glcm)
        features.append(contrast)
        
        # Homogeneity
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
        features.append(homogeneity)
        
        # Energy
        energy = np.sum(glcm ** 2)
        features.append(energy)
        
        # Entropy
        entropy = -np.sum(glcm[glcm > 0] * np.log2(glcm[glcm > 0]))
        features.append(entropy)
        
        # Correlation
        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        correlation = np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j + 1e-6)
        features.append(correlation)
        
        return features
    
    def calculate_lbp(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        rows, cols = image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                code = 0
                
                # Check 8 neighbors
                neighbors = [
                    (i - radius, j - radius),
                    (i - radius, j),
                    (i - radius, j + radius),
                    (i, j + radius),
                    (i + radius, j + radius),
                    (i + radius, j),
                    (i + radius, j - radius),
                    (i, j - radius)
                ]
                
                for idx, (ni, nj) in enumerate(neighbors):
                    if image[ni, nj] >= center:
                        code |= (1 << idx)
                
                lbp[i, j] = code
        
        return lbp
    
    def extract_shape_features(self, image: Image.Image) -> np.ndarray:
        """Extract shape-based features"""
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        features = []
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area
            area = cv2.contourArea(largest_contour)
            features.append(area)
            
            # Perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            features.append(circularity)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / (h + 1e-6)
            features.append(aspect_ratio)
            
            # Extent
            rect_area = w * h
            extent = area / (rect_area + 1e-6)
            features.append(extent)
            
            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            features.append(solidity)
        else:
            features.extend([0] * 6)
        
        return np.array(features)
    
    def extract_all_features(self, image: Image.Image) -> np.ndarray:
        """Extract all features from image"""
        # Resize image
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Extract features
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        shape_features = self.extract_shape_features(image)
        
        # Combine features
        all_features = np.concatenate([color_features, texture_features, shape_features])
        
        return all_features
    
    def process_dataset(self, image_paths: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process entire dataset"""
        features_list = []
        valid_labels = []
        valid_paths = []
        
        for path, label in zip(image_paths, labels):
            try:
                image = Image.open(path).convert('RGB')
                features = self.extract_all_features(image)
                features_list.append(features)
                valid_labels.append(label)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        return X, y, valid_paths
    
    def reduce_dimensions(self, X: np.ndarray, n_components: int = 50) -> np.ndarray:
        """Reduce feature dimensions using PCA"""
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        return X_reduced
    
    def save_features(self, features: np.ndarray, labels: np.ndarray, output_path: str):
        """Save features to file"""
        np.savez(output_path, features=features, labels=labels)
        print(f"Features saved to {output_path}")
    
    def load_features(self, input_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load features from file"""
        data = np.load(input_path)
        return data['features'], data['labels']
