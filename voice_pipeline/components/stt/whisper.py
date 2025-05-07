import os
import tempfile
import uuid
import logging
from typing import List

from voice_pipeline.core.interfaces import STTInterface
from voice_pipeline.core.models import AudioData, TranscriptionResult
from voice_pipeline.core.utils import create_temp_file, cleanup_temp_file

logger = logging.getLogger(__name__)

class FasterWhisperSTT(STTInterface):
    """Implementation of STT using Faster Whisper"""
    
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """Initialize with Whisper model settings"""
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper model: {model_size}")
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
    
    async def transcribe(self, audio_data: AudioData) -> TranscriptionResult:
        """Transcribe audio using Faster Whisper"""
        try:
            # Save audio to temporary file
            temp_file = create_temp_file(audio_data.data)
            
            logger.info(f"Processing audio with Whisper model")
            segments, info = self.whisper_model.transcribe(temp_file, beam_size=5)
            
            # Convert generator to list
            segments_list = list(segments)
            
            # Process results
            text = ""
            segment_data = []
            
            for segment in segments_list:
                text += segment.text + " "
                segment_data.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [{"word": w.word, "start": w.start, "end": w.end, "prob": w.probability} 
                             for w in (segment.words or [])]
                })
            
            # Clean up
            cleanup_temp_file(temp_file)
            
            return TranscriptionResult(
                text=text.strip(),
                segments=segment_data,
                language=info.language,
                language_probability=info.language_probability
            )
        
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return TranscriptionResult(text="", error=str(e))