from pydantic import BaseModel, Field

class VoiceConfig(BaseModel):
    voice_id: str = "1525f5a4-9897-4e2d-b1c2-f296246d243b"
    language: str = "en-US"
    speed: float = 1.0
    pitch: float = 1.0

class AssistantConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.7
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    voice: VoiceConfig = Field(default_factory=VoiceConfig)