import numpy as np
from PIL import Image
from typing import Tuple
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process(self, image: Image.Image) -> np.ndarray:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms
        processed = self.transform(image)
        
        return processed.numpy()
    
    def process_batch(self, images: list) -> np.ndarray:
        processed_images = []
        for image in images:
            processed_images.append(self.process(image))
        
        return np.stack(processed_images)
    
    def augment(self, image: Image.Image) -> list:
        augmented_images = []
        
        # Original
        augmented_images.append(self.process(image))
        
        # Horizontal flip
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(self.process(flipped))
        
        # Rotation
        rotated = image.rotate(45)
        augmented_images.append(self.process(rotated))
        
        # Brightness adjustment
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        bright = enhancer.enhance(1.2)
        augmented_images.append(self.process(bright))
        
        return augmented_images
