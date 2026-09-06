import os
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import hashlib
from datetime import datetime

class ImageCollector:
    def __init__(self, output_dir: str, metadata_file: str = "metadata.csv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / metadata_file
        self.metadata = []
        
    def collect_from_directory(self, source_dir: str, class_name: str) -> int:
        """Collect images from a source directory"""
        source_path = Path(source_dir)
        count = 0
        
        if not source_path.exists():
            print(f"Source directory not found: {source_dir}")
            return 0
        
        # Create class directory
        class_dir = self.output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for img_path in source_path.iterdir():
            if img_path.suffix.lower() in extensions:
                try:
                    # Validate image
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    # Generate unique filename
                    file_hash = self.get_file_hash(img_path)
                    new_filename = f"{class_name}_{file_hash}{img_path.suffix.lower()}"
                    new_path = class_dir / new_filename
                    
                    # Copy image
                    shutil.copy2(img_path, new_path)
                    
                    # Add to metadata
                    self.metadata.append({
                        'image_path': f"{class_name}/{new_filename}",
                        'label': class_name,
                        'original_path': str(img_path),
                        'file_hash': file_hash,
                        'file_size': img_path.stat().st_size,
                        'source': 'directory',
                        'collected_at': datetime.now().isoformat()
                    })
                    
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return count
    
    def collect_from_list(self, image_paths: List[str], class_name: str) -> int:
        """Collect images from a list of paths"""
        count = 0
        class_dir = self.output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for img_path in image_paths:
            img_path = Path(img_path)
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    file_hash = self.get_file_hash(img_path)
                    new_filename = f"{class_name}_{file_hash}{img_path.suffix.lower()}"
                    new_path = class_dir / new_filename
                    
                    shutil.copy2(img_path, new_path)
                    
                    self.metadata.append({
                        'image_path': f"{class_name}/{new_filename}",
                        'label': class_name,
                        'original_path': str(img_path),
                        'file_hash': file_hash,
                        'file_size': img_path.stat().st_size,
                        'source': 'list',
                        'collected_at': datetime.now().isoformat()
                    })
                    
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return count
    
    def validate_image(self, image_path: Path) -> Dict:
        """Validate a single image"""
        result = {
            'valid': False,
            'error': None,
            'width': 0,
            'height': 0,
            'format': None,
            'mode': None
        }
        
        try:
            with Image.open(image_path) as img:
                img.verify()
                
            with Image.open(image_path) as img:
                result['valid'] = True
                result['width'] = img.width
                result['height'] = img.height
                result['format'] = img.format
                result['mode'] = img.mode
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:8]
    
    def save_metadata(self):
        """Save metadata to CSV"""
        df = pd.DataFrame(self.metadata)
        df.to_csv(self.metadata_file, index=False)
        print(f"Metadata saved to {self.metadata_file}")
        
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        df = pd.DataFrame(self.metadata)
        
        if df.empty:
            return {'total_images': 0}
        
        stats = {
            'total_images': len(df),
            'classes': df['label'].value_counts().to_dict(),
            'total_size_mb': df['file_size'].sum() / (1024 * 1024),
            'avg_size_kb': df['file_size'].mean() / 1024
        }
        
        return stats
    
    def remove_duplicates(self) -> int:
        """Remove duplicate images based on hash"""
        df = pd.DataFrame(self.metadata)
        
        if df.empty:
            return 0
        
        # Find duplicates
        duplicates = df[df.duplicated(subset=['file_hash'], keep='first')]
        
        # Remove duplicate files
        for _, row in duplicates.iterrows():
            file_path = self.output_dir / row['image_path']
            if file_path.exists():
                file_path.unlink()
        
        # Update metadata
        self.metadata = df.drop_duplicates(subset=['file_hash']).to_dict('records')
        
        return len(duplicates)
