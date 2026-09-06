import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Optional
import random
import cv2

class AdvancedAugmentation:
    """Advanced data augmentation techniques"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError


class CutMix(AdvancedAugmentation):
    """CutMix augmentation"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(
        self,
        image1: Image.Image,
        image2: Image.Image,
        label1: int,
        label2: int
    ) -> Tuple[Image.Image, float, int]:
        lam = np.random.beta(self.alpha, self.alpha)
        
        W, H = image1.size
        
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        image1_array = np.array(image1)
        image2_array = np.array(image2)
        
        image1_array[bby1:bby2, bbx1:bbx2] = image2_array[bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        mixed_image = Image.fromarray(image1_array)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label


class MixUp(AdvancedAugmentation):
    """MixUp augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(
        self,
        image1: Image.Image,
        image2: Image.Image,
        label1: int,
        label2: int
    ) -> Tuple[Image.Image, float, int]:
        lam = np.random.beta(self.alpha, self.alpha)
        
        image1_array = np.array(image1).astype(np.float32)
        image2_array = np.array(image2).astype(np.float32)
        
        mixed_array = lam * image1_array + (1 - lam) * image2_array
        mixed_image = Image.fromarray(mixed_array.astype(np.uint8))
        
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label


class Mosaic(AdvancedAugmentation):
    """Mosaic augmentation"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
    def __call__(self, images: List[Image.Image]) -> Image.Image:
        if len(images) < 4:
            images = images * (4 // len(images) + 1)
        
        images = images[:4]
        
        w, h = self.image_size
        
        # Create mosaic canvas
        canvas = Image.new('RGB', (2 * w, 2 * h))
        
        # Place images in quadrants
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        
        for img, pos in zip(images, positions):
            img_resized = img.resize((w, h))
            canvas.paste(img_resized, pos)
        
        # Random crop to original size
        left = random.randint(0, w)
        top = random.randint(0, h)
        right = left + w
        bottom = top + h
        
        canvas = canvas.crop((left, top, right, bottom))
        
        return canvas


class RandomErasing(AdvancedAugmentation):
    """Random Erasing augmentation"""
    
    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        
        image_array = np.array(image)
        
        h, w, c = image_array.shape
        area = h * w
        
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            rh = int(round(np.sqrt(target_area * aspect_ratio)))
            rw = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if rw < w and rh < h:
                x1 = random.randint(0, h - rh)
                y1 = random.randint(0, w - rw)
                
                image_array[x1:x1 + rh, y1:y1 + rw] = 0
                
                return Image.fromarray(image_array)
        
        return image


class GridMask(AdvancedAugmentation):
    """Grid Mask augmentation"""
    
    def __init__(
        self,
        d: int = 96,
        ratio: float = 0.6,
        probability: float = 0.5
    ):
        self.d = d
        self.ratio = ratio
        self.probability = probability
        
    def _get_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h, self.d):
            for j in range(0, w, self.d):
                mask[i:i + int(self.d * self.ratio), j:j + int(self.d * self.ratio)] = 1
        
        return mask
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        mask = self._get_mask(h, w)
        
        if len(image_array.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)
        
        image_array = image_array * mask
        
        return Image.fromarray(image_array.astype(np.uint8))


class AutoAugment(AdvancedAugmentation):
    """Auto Augment policy"""
    
    def __init__(self):
        self.policies = [
            [
                (ImageEnhance.Brightness, 0.4, (0.5, 1.5)),
                (ImageEnhance.Contrast, 0.4, (0.5, 1.5)),
            ],
            [
                (ImageEnhance.Color, 0.4, (0.5, 1.5)),
                (ImageEnhance.Sharpness, 0.4, (0.5, 1.5)),
            ],
            [
                (ImageFilter.GaussianBlur, 0.4, (0.5, 2.0)),
                (ImageEnhance.Brightness, 0.4, (0.5, 1.5)),
            ],
            [
                (ImageEnhance.Color, 0.4, (0.5, 1.5)),
                (ImageEnhance.Brightness, 0.4, (0.5, 1.5)),
            ],
        ]
        
    def _apply_policy(self, image: Image.Image, policy: list) -> Image.Image:
        for augmenter, probability, magnitude_range in policy:
            if random.random() > probability:
                continue
                
                if augmenter == ImageFilter.GaussianBlur:
                    magnitude = random.uniform(*magnitude_range)
                    image = image.filter(ImageFilter.GaussianBlur(radius=magnitude))
                else:
                    augmenter_obj = augmenter(image)
                    magnitude = random.uniform(*magnitude_range)
                    image = augmenter_obj.enhance(magnitude)
        
        return image
    
    def __call__(self, image: Image.Image) -> Image.Image:
        policy = random.choice(self.policies)
        return self._apply_policy(image, policy)


class TrivialAugment(AdvancedAugmentation):
    """Trivial Augment"""
    
    def __init__(self):
        self.augmenters = [
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
            self._posterize,
        ]
        
    def _auto_contrast(self, image, magnitude):
        return ImageOps.autocontrast(image)
    
    def _equalize(self, image, magnitude):
        return ImageOps.equalize(image)
    
    def _rotate(self, image, magnitude):
        return image.rotate(magnitude * 30)
    
    def _solarize(self, image, magnitude):
        return ImageOps.solarize(image, magnitude * 256)
    
    def _color(self, image, magnitude):
        return ImageEnhance.Color(image).enhance(1 + magnitude * 0.9)
    
    def _contrast(self, image, magnitude):
        return ImageEnhance.Contrast(image).enhance(1 + magnitude * 0.9)
    
    def _brightness(self, image, magnitude):
        return ImageEnhance.Brightness(image).enhance(1 + magnitude * 0.9)
    
    def _sharpness(self, image, magnitude):
        return ImageEnhance.Sharpness(image).enhance(1 + magnitude * 0.9)
    
    def _shear_x(self, image, magnitude):
        return image.transform(image.size, Image.AFFINE, (1, magnitude * 0.3, 0, 0, 1, 0))
    
    def _shear_y(self, image, magnitude):
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, magnitude * 0.3, 1, 0))
    
    def _translate_x(self, image, magnitude):
        return image.transform(image.size, Image.AFFINE, (1, 0, magnitude * image.size[0] * 0.3, 0, 1, 0))
    
    def _translate_y(self, image, magnitude):
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * image.size[1] * 0.3))
    
    def _posterize(self, image, magnitude):
        return ImageOps.posterize(image, int(magnitude * 4))
    
    def __call__(self, image: Image.Image) -> Image.Image:
        augmenter = random.choice(self.augmenters)
        magnitude = random.random()
        return augmenter(image, magnitude)


class RandAugment(AdvancedAugmentation):
    """Rand Augment"""
    
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m
        
        self.augmenters = [
            (self._auto_contrast, 0.5, 1.0),
            (self._equalize, 0.5, 1.0),
            (self._rotate, 0.5, 0.3),
            (self._solarize, 0.5, 0.4),
            (self._color, 0.5, 0.9),
            (self._contrast, 0.5, 0.9),
            (self._brightness, 0.5, 0.9),
            (self._sharpness, 0.5, 0.9),
            (self._posterize, 0.5, 0.4),
        ]
        
    def _auto_contrast(self, image, magnitude):
        return ImageOps.autocontrast(image)
    
    def _equalize(self, image, magnitude):
        return ImageOps.equalize(image)
    
    def _rotate(self, image, magnitude):
        return image.rotate(magnitude * 30)
    
    def _solarize(self, image, magnitude):
        return ImageOps.solarize(image, magnitude * 256)
    
    def _color(self, image, magnitude):
        return ImageEnhance.Color(image).enhance(1 + magnitude * 0.9)
    
    def _contrast(self, image, magnitude):
        return ImageEnhance.Contrast(image).enhance(1 + magnitude * 0.9)
    
    def _brightness(self, image, magnitude):
        return ImageEnhance.Brightness(image).enhance(1 + magnitude * 0.9)
    
    def _sharpness(self, image, magnitude):
        return ImageEnhance.Sharpness(image).enhance(1 + magnitude * 0.9)
    
    def _posterize(self, image, magnitude):
        return ImageOps.posterize(image, int(magnitude * 4))
    
    def __call__(self, image: Image.Image) -> Image.Image:
        augmenters = random.choices(self.augmenters, k=self.n)
        
        for augmenter, probability, magnitude in augmenters:
            if random.random() < probability:
                image = augmenter(image, magnitude)
        
        return image
