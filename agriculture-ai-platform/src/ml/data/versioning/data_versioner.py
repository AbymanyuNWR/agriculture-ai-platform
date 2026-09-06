import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil

class DataVersioner:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.versions_file = self.data_dir / "versions.json"
        self.versions = self.load_versions()
        
    def load_versions(self) -> Dict:
        """Load version history"""
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                return json.load(f)
        return {'versions': [], 'current_version': None}
    
    def save_versions(self):
        """Save version history"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def calculate_dataset_hash(self, version_dir: Path) -> str:
        """Calculate hash of dataset"""
        hash_md5 = hashlib.md5()
        
        for file_path in sorted(version_dir.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def create_version(
        self,
        description: str = "",
        metadata: Dict = None
    ) -> str:
        """Create a new data version"""
        # Determine version number
        if self.versions['versions']:
            latest_version = self.versions['versions'][-1]['version']
            version_num = int(latest_version.split('v')[1]) + 1
        else:
            version_num = 1
        
        version_name = f"v{version_num}"
        
        # Create version directory
        version_dir = self.data_dir / "versions" / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy current data
        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            shutil.copytree(raw_dir, version_dir / "raw", dirs_exist_ok=True)
        
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            shutil.copytree(processed_dir, version_dir / "processed", dirs_exist_ok=True)
        
        # Calculate hash
        dataset_hash = self.calculate_dataset_hash(version_dir)
        
        # Create version info
        version_info = {
            'version': version_name,
            'hash': dataset_hash,
            'description': description,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'path': str(version_dir)
        }
        
        # Add to versions
        self.versions['versions'].append(version_info)
        self.versions['current_version'] = version_name
        
        # Save versions
        self.save_versions()
        
        print(f"Created version: {version_name}")
        print(f"Dataset hash: {dataset_hash}")
        
        return version_name
    
    def get_version(self, version_name: str) -> Optional[Dict]:
        """Get version info"""
        for version in self.versions['versions']:
            if version['version'] == version_name:
                return version
        return None
    
    def list_versions(self) -> List[Dict]:
        """List all versions"""
        return self.versions['versions']
    
    def get_current_version(self) -> Optional[str]:
        """Get current version"""
        return self.versions.get('current_version')
    
    def set_current_version(self, version_name: str):
        """Set current version"""
        version = self.get_version(version_name)
        if version:
            self.versions['current_version'] = version_name
            self.save_versions()
            print(f"Current version set to: {version_name}")
        else:
            print(f"Version {version_name} not found")
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two versions"""
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if not v1 or not v2:
            return {'error': 'Version not found'}
        
        comparison = {
            'version1': v1,
            'version2': v2,
            'same_hash': v1['hash'] == v2['hash'],
            'version1_path': Path(v1['path']),
            'version2_path': Path(v2['path'])
        }
        
        # Count files
        v1_files = list(Path(v1['path']).rglob("*"))
        v2_files = list(Path(v2['path']).rglob("*"))
        
        comparison['version1_files'] = len([f for f in v1_files if f.is_file()])
        comparison['version2_files'] = len([f for f in v2_files if f.is_file()])
        
        return comparison
    
    def delete_version(self, version_name: str):
        """Delete a version"""
        version = self.get_version(version_name)
        if not version:
            print(f"Version {version_name} not found")
            return
        
        # Delete directory
        version_dir = Path(version['path'])
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from versions
        self.versions['versions'] = [
            v for v in self.versions['versions'] if v['version'] != version_name
        ]
        
        # Update current version if needed
        if self.versions['current_version'] == version_name:
            if self.versions['versions']:
                self.versions['current_version'] = self.versions['versions'][-1]['version']
            else:
                self.versions['current_version'] = None
        
        self.save_versions()
        print(f"Deleted version: {version_name}")
    
    def restore_version(self, version_name: str):
        """Restore a version to current"""
        version = self.get_version(version_name)
        if not version:
            print(f"Version {version_name} not found")
            return
        
        version_dir = Path(version['path'])
        
        # Clear current data
        raw_dir = self.data_dir / "raw"
        processed_dir = self.data_dir / "processed"
        
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        
        # Copy version data
        if (version_dir / "raw").exists():
            shutil.copytree(version_dir / "raw", raw_dir)
        if (version_dir / "processed").exists():
            shutil.copytree(version_dir / "processed", processed_dir)
        
        self.set_current_version(version_name)
        print(f"Restored version: {version_name}")
    
    def get_version_history(self) -> List[Dict]:
        """Get version history with details"""
        history = []
        for version in self.versions['versions']:
            history.append({
                'version': version['version'],
                'created_at': version['created_at'],
                'description': version['description'],
                'hash': version['hash'][:8]  # Short hash
            })
        return history
