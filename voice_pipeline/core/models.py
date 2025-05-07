from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

class AudioData(BaseModel):
    """Data class for audio data"""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    
class TranscriptionResult(BaseModel):
    """Data class for transcription results"""
    text: str
    segments: List[Dict[str, Any]] = []
    language: Optional[str] = None
    language_probability: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    
class LLMResponse(BaseModel):
    """Data class for LLM responses"""
    text: str
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = {}
    
class TTSResult(BaseModel):
    """Data class for TTS results"""
    audio: bytes
    format: str = "mp3"
    sample_rate: int = 44100
    
class Message(BaseModel):
    """Data class for conversation messages"""
    role: str  # "user" or "assistant"
    content: str
    
class ConversationContext(BaseModel):
    """Context for a conversation"""
    messages: List[Message] = []
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append(Message(role=role, content=content))
        
    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get all messages in the format expected by most LLM APIs"""
        result = []
        if include_system and self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        
        for msg in self.messages:
            result.append({"role": msg.role, "content": msg.content})
            
        return result
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []