import logging
from typing import Dict, Any

from voice_pipeline.core.interfaces import (
    VADInterface, STTInterface, LLMInterface, TTSInterface, TurnDetectorInterface
)
from voice_pipeline.core.models import AudioData, ConversationContext
from voice_pipeline.components.turn_detector.eou import EOUTurnDetector

logger = logging.getLogger(__name__)

class VoicePipelineAgent:
    """Voice Pipeline Agent that orchestrates the conversation flow"""
    
    def __init__(self,
                vad: VADInterface,
                stt: STTInterface,
                llm: LLMInterface,
                tts: TTSInterface,
                turn_detector: TurnDetectorInterface = None,
                min_endpointing_delay: float = 0.5,
                max_endpointing_delay: float = 5.0,
                chat_ctx: ConversationContext = None):
        """Initialize the voice pipeline agent with components"""
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.turn_detector = turn_detector or EOUTurnDetector()
        self.min_endpointing_delay = min_endpointing_delay
        self.max_endpointing_delay = max_endpointing_delay
        self.chat_ctx = chat_ctx or ConversationContext()
        
    async def process_audio(self, audio_data: AudioData) -> Dict[str, Any]:
        """Process audio end-to-end: from speech to response audio"""
        result = {
            "success": False,
            "transcription": None,
            "llm_response": None,
            "audio_response": None,
        }
        
        # Step 1: Transcribe audio
        transcription = await self.stt.transcribe(audio_data)
        result["transcription"] = transcription
        
        if not transcription.text:
            logger.info("No speech detected or transcription failed")
            return result
        
        # Step 2: Process with LLM
        logger.info(f"Adding user message to context: {transcription.text}")
        self.chat_ctx.add_message("user", transcription.text)
        
        llm_response = await self.llm.generate_response(self.chat_ctx)
        result["llm_response"] = llm_response
        
        # Add assistant response to context
        self.chat_ctx.add_message("assistant", llm_response.text)
        
        # Step 3: Convert to speech
        try:
            tts_result = await self.tts.synthesize(llm_response.text)
            result["audio_response"] = tts_result
            result["success"] = True
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            result["success"] = False
        
        return result
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """Process text input (without audio)"""
        result = {
            "success": False,
            "llm_response": None,
            "audio_response": None,
        }
        
        # Add user message to context
        self.chat_ctx.add_message("user", text)
        
        # Get LLM response
        llm_response = await self.llm.generate_response(self.chat_ctx)
        result["llm_response"] = llm_response
        
        # Add assistant response to context
        self.chat_ctx.add_message("assistant", llm_response.text)
        
        # Convert to speech
        try:
            tts_result = await self.tts.synthesize(llm_response.text)
            result["audio_response"] = tts_result
            result["success"] = True
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            result["success"] = False
        
        return result
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.chat_ctx.clear()
        
    def update_system_prompt(self, prompt: str):
        """Update the system prompt"""
        self.chat_ctx.system_prompt = prompt