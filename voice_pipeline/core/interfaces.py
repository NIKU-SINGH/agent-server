# Core interfaces for pipeline components
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import AudioData, ConversationContext, LLMResponse, TranscriptionResult, TTSResult

class VADInterface(ABC):
    """Voice Activity Detection interface"""
    
    @abstractmethod
    async def detect_speech(self, audio_data: AudioData) -> bool:
        """Detect if audio contains speech"""
        pass
    
    @abstractmethod
    async def detect_end_of_utterance(self, audio_stream) -> bool:
        """Detect if user has finished speaking"""
        pass

class STTInterface(ABC):
    """Speech-to-Text interface"""
    
    @abstractmethod
    async def transcribe(self, audio_data: AudioData) -> TranscriptionResult:
        """Transcribe audio to text"""
        pass

class LLMInterface(ABC):
    """Language Model interface"""
    
    @abstractmethod
    async def generate_response(self, 
                               context: ConversationContext,
                               temperature: float = 0.7) -> LLMResponse:
        """Generate a response based on conversation context"""
        pass

class TTSInterface(ABC):
    """Text-to-Speech interface"""
    
    @abstractmethod
    async def synthesize(self, text: str, language: str = None) -> TTSResult:
        """Synthesize speech from text
        
        Args:
            text: Text to synthesize
            language: ISO language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            TTSResult: Audio synthesis result
        """
        pass

class TurnDetectorInterface(ABC):
    """Turn detection interface"""
    
    @abstractmethod
    async def is_turn_complete(self, audio_data: AudioData) -> bool:
        """Determine if the user's turn is complete"""
        pass