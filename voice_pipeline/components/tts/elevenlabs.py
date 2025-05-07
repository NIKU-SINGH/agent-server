import logging
import os
from dotenv import load_dotenv
from typing import List

from voice_pipeline.core.interfaces import TTSInterface
from voice_pipeline.core.models import TTSResult
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

# Get voice ID from environment variables
voice_id: str = os.getenv('ELEVENLABS_VOICE_ID', '')  # Default to empty string if not found

logger = logging.getLogger(__name__)

class ElevenLabsTTS(TTSInterface):
    """Implementation of TTS using ElevenLabs API"""
    
    def __init__(self, api_key: str):
        """Initialize with ElevenLabs API key"""
        self.api_key = api_key
        self.current_language = "en"  # default language
        self.client = ElevenLabs(api_key=api_key)
        
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
        """Convert text to speech using ElevenLabs TTS API"""
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
            
            # Generate audio using the new client API
            audio_bytes = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
            
            # Convert generator to bytes
            if hasattr(audio_bytes, '__iter__'):
                audio_bytes = b''.join(audio_bytes)
            
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