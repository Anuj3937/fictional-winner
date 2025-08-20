import os
import shutil
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
from datetime import datetime
import hashlib
import mimetypes
from loguru import logger
import asyncio
import aiofiles

class FileManager:
    """High-performance file management with async operations"""
    
    def __init__(self, base_directory: str = "generated"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True, parents=True)
        
    async def create_project_structure(self, project_name: str) -> Path:
        """Create project directory structure"""
        project_path = self.base_directory / project_name
        
        # Create main directories
        directories = [
            project_path,
            project_path / "models",
            project_path / "data",
            project_path / "logs", 
            project_path / "tests",
            project_path / "static",
            project_path / "temp",
            project_path / "config"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
            
        logger.info(f"Created project structure: {project_path}")
        return project_path
    
    async def save_file_async(self, file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
        """Asynchronously save file with content"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(exist_ok=True, parents=True)
            
            async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
                await f.write(content)
            
            logger.debug(f"Saved file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {str(e)}")
            return False
    
    async def read_file_async(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        """Asynchronously read file content"""
        try:
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            return None
    
    def save_json(self, file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """Save data as JSON file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.debug(f"Saved JSON file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON {file_path}: {str(e)}")
            return False
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON {file_path}: {str(e)}")
            return None
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Copy file from source to destination"""
        try:
            source = Path(source)
            destination = Path(destination)
            destination.parent.mkdir(exist_ok=True, parents=True)
            
            shutil.copy2(source, destination)
            logger.debug(f"Copied file: {source} -> {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file {source} -> {destination}: {str(e)}")
            return False
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Move file from source to destination"""
        try:
            source = Path(source)
            destination = Path(destination)
            destination.parent.mkdir(exist_ok=True, parents=True)
            
            shutil.move(str(source), str(destination))
            logger.debug(f"Moved file: {source} -> {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file {source} -> {destination}: {str(e)}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete file"""
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    def delete_directory(self, directory_path: Union[str, Path]) -> bool:
        """Delete directory and all contents"""
        try:
            directory_path = Path(directory_path)
            if directory_path.exists():
                shutil.rmtree(directory_path)
                logger.debug(f"Deleted directory: {directory_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete directory {directory_path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get comprehensive file information"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            
            info = {
                "name": file_path.name,
                "path": str(file_path.absolute()),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "extension": file_path.suffix,
                "mime_type": mimetypes.guess_type(str(file_path))[0]
            }
            
            # Add file hash for integrity checking
            if file_path.is_file() and stat.st_size < 100 * 1024 * 1024:  # Only for files < 100MB
                info["md5_hash"] = self.calculate_file_hash(file_path)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info {file_path}: {str(e)}")
            return None
    
    def calculate_file_hash(self, file_path: Union[str, Path], algorithm: str = "md5") -> Optional[str]:
        """Calculate file hash for integrity checking"""
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {str(e)}")
            return None
    
    def list_directory(self, directory_path: Union[str, Path], recursive: bool = False) -> List[Dict[str, Any]]:
        """List directory contents with file information"""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                return []
            
            files = []
            
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for item in directory_path.glob(pattern):
                file_info = self.get_file_info(item)
                if file_info:
                    files.append(file_info)
            
            return sorted(files, key=lambda x: (not x['is_directory'], x['name'].lower()))
            
        except Exception as e:
            logger.error(f"Failed to list directory {directory_path}: {str(e)}")
            return []
    
    def create_archive(self, source_directory: Union[str, Path], archive_path: Union[str, Path]) -> bool:
        """Create ZIP archive from directory"""
        try:
            source_directory = Path(source_directory)
            archive_path = Path(archive_path)
            archive_path.parent.mkdir(exist_ok=True, parents=True)
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_directory.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_directory)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Created archive: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create archive {archive_path}: {str(e)}")
            return False
    
    def extract_archive(self, archive_path: Union[str, Path], extract_to: Union[str, Path]) -> bool:
        """Extract ZIP archive to directory"""
        try:
            archive_path = Path(archive_path)
            extract_to = Path(extract_to)
            extract_to.mkdir(exist_ok=True, parents=True)
            
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_to)
            
            logger.info(f"Extracted archive: {archive_path} -> {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract archive {archive_path}: {str(e)}")
            return False
    
    def cleanup_old_files(self, directory_path: Union[str, Path], max_age_days: int = 30) -> int:
        """Clean up old files from directory"""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                return 0
            
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            deleted_count = 0
            
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files from {directory_path}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup files in {directory_path}: {str(e)}")
            return 0
    
    def get_directory_size(self, directory_path: Union[str, Path]) -> Dict[str, Any]:
        """Get directory size information"""
        try:
            directory_path = Path(directory_path)
            
            total_size = 0
            file_count = 0
            directory_count = 0
            
            for item in directory_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
                elif item.is_dir():
                    directory_count += 1
            
            return {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "file_count": file_count,
                "directory_count": directory_count,
                "path": str(directory_path.absolute())
            }
            
        except Exception as e:
            logger.error(f"Failed to get directory size {directory_path}: {str(e)}")
            return {
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "total_size_gb": 0,
                "file_count": 0,
                "directory_count": 0,
                "path": str(directory_path),
                "error": str(e)
            }

# Global file manager instance
file_manager = FileManager()

# Utility functions for backwards compatibility
def save_file(file_path: Union[str, Path], content: str) -> bool:
    """Save file synchronously"""
    return file_manager.save_file_async(file_path, content)

def read_file(file_path: Union[str, Path]) -> Optional[str]:
    """Read file synchronously"""
    return file_manager.read_file_async(file_path)

def create_project_directory(project_name: str) -> Path:
    """Create project directory structure"""
    return asyncio.run(file_manager.create_project_structure(project_name))
