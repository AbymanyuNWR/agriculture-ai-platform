import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from typing import Tuple, List, Optional
import cv2

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(image_path).convert('RGB')
    
    def resize(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """Resize image"""
        if size is None:
            size = self.target_size
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def normalize(self, image: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
        """Normalize image"""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return image
    
    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """Convert image to tensor format (C, H, W)"""
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        return image
    
    def augment(self, image: Image.Image) -> List[Image.Image]:
        """Apply augmentation to image"""
        augmented = []
        
        # Original
        augmented.append(image.copy())
        
        # Horizontal flip
        augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        
        # Vertical flip
        augmented.append(image.transpose(Image.FLIP_TOP_BOTTOM))
        
        # Rotation
        augmented.append(image.rotate(45))
        augmented.append(image.rotate(-45))
        
        # Brightness
        enhancer = ImageEnhance.Brightness(image)
        augmented.append(enhancer.enhance(1.2))
        augmented.append(enhancer.enhance(0.8))
        
        # Contrast
        enhancer = ImageEnhance.Contrast(image)
        augmented.append(enhancer.enhance(1.2))
        augmented.append(enhancer.enhance(0.8))
        
        # Saturation
        enhancer = ImageEnhance.Color(image)
        augmented.append(enhancer.enhance(1.2))
        augmented.append(enhancer.enhance(0.8))
        
        # Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        augmented.append(enhancer.enhance(2.0))
        
        # Gaussian blur
        augmented.append(image.filter(ImageFilter.GaussianBlur(radius=1)))
        
        return augmented
    
    def random_crop(self, image: Image.Image, crop_size: Tuple[int, int] = None) -> Image.Image:
        """Random crop image"""
        if crop_size is None:
            crop_size = self.target_size
        
        width, height = image.size
        left = np.random.randint(0, width - crop_size[0])
        top = np.random.randint(0, height - crop_size[1])
        right = left + crop_size[0]
        bottom = top + crop_size[1]
        
        return image.crop((left, top, right, bottom))
    
    def center_crop(self, image: Image.Image, crop_size: Tuple[int, int] = None) -> Image.Image:
        """Center crop image"""
        if crop_size is None:
            crop_size = self.target_size
        
        width, height = image.size
        left = (width - crop_size[0]) // 2
        top = (height - crop_size[1]) // 2
        right = left + crop_size[0]
        bottom = top + crop_size[1]
        
        return image.crop((left, top, right, bottom))
    
    def extract_color_histogram(self, image: Image.Image, bins: int = 64) -> np.ndarray:
        """Extract color histogram features"""
        img_array = np.array(image)
        
        # Calculate histogram for each channel
        hist_r = cv2.calcHist([img_array], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([img_array], [2], None, [bins], [0, 256])
        
        # Concatenate histograms
        hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
        
        # Normalize
        hist = hist / hist.sum()
        
        return hist
    
    def extract_texture_features(self, image: Image.Image) -> np.ndarray:
        """Extract texture features using GLCM"""
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Calculate GLCM
        glcm = np.zeros((256, 256), dtype=np.uint8)
        
        for i in range(img_array.shape[0] - 1):
            for j in range(img_array.shape[1] - 1):
                glcm[img_array[i, j], img_array[i, j + 1]] += 1
        
        # Normalize GLCM
        glcm = glcm / glcm.sum()
        
        # Calculate features
        i, j = np.meshgrid(range(256), range(256))
        
        contrast = np.sum((i - j) ** 2 * glcm)
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
        energy = np.sum(glcm ** 2)
        correlation = np.sum((i - np.mean(glcm)) * (j - np.mean(glcm)) * glcm) / (np.std(glcm) * np.std(glcm) + 1e-6)
        
        return np.array([contrast, homogeneity, energy, correlation])
    
    def process_image(self, image_path: str, augment: bool = False) -> List[np.ndarray]:
        """Complete image processing pipeline"""
        # Load image
        image = self.load_image(image_path)
        
        # Resize
        image = self.resize(image)
        
        if augment:
            # Apply augmentation
            images = self.augment(image)
        else:
            images = [image]
        
        processed_images = []
        for img in images:
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize
            img_array = self.normalize(img_array)
            
            # Convert to tensor
            img_tensor = self.to_tensor(img_array)
            
            processed_images.append(img_tensor)
        
        return processed_images
    
    def process_batch(self, image_paths: List[str], augment: bool = False) -> np.ndarray:
        """Process a batch of images"""
        processed_batch = []
        
        for path in image_paths:
            processed = self.process_image(path, augment)
            processed_batch.extend(processed)
        
        return np.stack(processed_batch)
