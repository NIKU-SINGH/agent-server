import logging
import os
from dotenv import load_dotenv
from typing import List

from voice_pipeline.core.interfaces import TTSInterface
from voice_pipeline.core.models import TTSResult
from cartesia import AsyncCartesia

# Load environment variables
load_dotenv()

# Get voice ID from environment variables
voice_id: str = os.getenv('CARTESIA_VOICE_ID', '')  # Default to empty string if not found

logger = logging.getLogger(__name__)

class CartesiaTTS(TTSInterface):
    """Implementation of TTS using Cartesia API"""
    
    def __init__(self, api_key: str):
        """Initialize with Cartesia API key"""
        self.api_key = api_key
        self.current_language = "en"  # default language
        
    def set_language(self, language: str):
        """Set the TTS language
        
        Args:
            language: ISO language code (e.g., 'en', 'es', 'fr')
        """
        self.current_language = language
    
    async def synthesize(self, 
                        text: str, 
                        voice_id: str = voice_id,
                        language: str = None,
                        speed: float = 0.6,
                        pitch: float = 1.0) -> TTSResult:
        """Convert text to speech using Cartesia TTS API"""
        try:
            # Use provided language or fall back to current_language
            use_language = language or self.current_language
            
            # List of supported languages (add more as needed)
            SUPPORTED_LANGUAGES = {'en', 'es', 'fr', 'de', 'it'}
            
            # Check if language is supported, if not default to 'en'
            base_language = use_language.split('-')[0] if '-' in use_language else use_language
            if base_language.lower() not in SUPPORTED_LANGUAGES:
                logger.warning(f"Language '{use_language}' not supported, defaulting to English")
                base_language = 'en'
            
            # Initialize Cartesia client
            client = AsyncCartesia(api_key=self.api_key)
            
            # Map speed to string value if needed
            speed_str = "normal"
            if speed < 0.8:
                speed_str = "slow"
            elif speed > 1.2:
                speed_str = "fast"
            
            # Make TTS request and collect all chunks
            audio_chunks = []
            async for chunk in client.tts.bytes(
                model_id="sonic-2",
                transcript=text,
                voice={
                    "id": voice_id,
                    "experimental_controls": {
                        "speed": speed_str,
                        "emotion": [],
                    },
                },
                language=base_language,
                output_format={
                    "container": "mp3",
                    "sample_rate": 44100,
                },
            ):
                audio_chunks.append(chunk)
            
            # Combine all chunks into a single audio buffer
            audio_bytes = b''.join(audio_chunks)
            
            return TTSResult(
                audio=audio_bytes,
                format="mp3",
                sample_rate=44100
            )
                    
        except Exception as e:
            logger.error(f"Error in TTS conversion: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise