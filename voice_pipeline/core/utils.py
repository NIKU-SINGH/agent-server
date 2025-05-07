import os
import tempfile
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_temp_file(data: bytes, prefix: str = "audio_", suffix: str = ".wav") -> str:
    """Create a temporary file with the given data and return its path"""
    temp_dir = tempfile.gettempdir()
    filename = f"{prefix}{uuid.uuid4()}{suffix}"
    filepath = os.path.join(temp_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(data)
    
    return filepath

def cleanup_temp_file(filepath: str) -> bool:
    """Remove a temporary file and return success status"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    except Exception as e:
        logger.error(f"Failed to remove temporary file {filepath}: {e}")
        return False
    return False