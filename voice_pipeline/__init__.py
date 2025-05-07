"""Voice pipeline package for modular voice assistant components"""

# Import main interfaces
from voice_pipeline.components.llm.groq_llama import GroqLlamaLLM
from voice_pipeline.components.tts.elevenlabs import ElevenLabsTTS
from voice_pipeline.core.interfaces import (
    VADInterface, STTInterface, LLMInterface, TTSInterface, TurnDetectorInterface
)

# Import data models
from voice_pipeline.core.models import (
    AudioData, TranscriptionResult, LLMResponse, TTSResult, 
    Message, ConversationContext
)

# Import main agent class
from voice_pipeline.pipeline.agent import VoicePipelineAgent

# Import component implementations
from voice_pipeline.components.stt.whisper import FasterWhisperSTT
from voice_pipeline.components.llm.openai import OpenAILLM
# from voice_pipeline.components.llm.anthropic import AnthropicLLM  # Add more LLM imports
from voice_pipeline.components.tts.cartesia import CartesiaTTS
from voice_pipeline.components.vad.simple import SimpleEndpointingVAD
from voice_pipeline.components.turn_detector.eou import EOUTurnDetector

# Factory functions for easier component creation
def create_llm(model_name: str, api_key: str, **kwargs) -> LLMInterface:
    """Create an LLM component based on the specified model name
    
    Args:
        model_name: Name of the LLM service ('openai', 'anthropic', etc.)
        api_key: API key for the selected service
        **kwargs: Additional model-specific parameters
        
    Returns:
        LLMInterface: Configured LLM component
    """
    llm_models = {
        'openai': lambda: OpenAILLM(api_key=api_key, model=kwargs.get('model', 'gpt-4')),
        'llama': lambda: GroqLlamaLLM(api_key=api_key),
        # Add more LLM providers here
    }
    
    if model_name.lower() not in llm_models:
        raise ValueError(f"Unsupported LLM model: {model_name}. Available models: {list(llm_models.keys())}")
    
    return llm_models[model_name.lower()]()

def create_stt(model_name: str, api_key: str = None, **kwargs) -> STTInterface:
    """Create an STT component based on the specified model name
    
    Args:
        model_name: Name of the STT service ('whisper', 'google', etc.)
        api_key: API key for the selected service (if required)
        **kwargs: Additional model-specific parameters
        
    Returns:
        STTInterface: Configured STT component
    """
    stt_models = {
        'whisper': lambda: FasterWhisperSTT(model_size=kwargs.get('model_size', 'base')),
    }
    
    if model_name.lower() not in stt_models:
        raise ValueError(f"Unsupported STT model: {model_name}. Available models: {list(stt_models.keys())}")
    
    return stt_models[model_name.lower()]()

def create_tts(model_name: str, api_key: str, **kwargs) -> TTSInterface:
    """Create a TTS component based on the specified model name
    
    Args:
        model_name: Name of the TTS service ('cartesia', 'elevenlabs', etc.)
        api_key: API key for the selected service
        **kwargs: Additional parameters like default_language
        
    Returns:
        TTSInterface: Configured TTS component
    """
    tts_models = {
        'cartesia': lambda: CartesiaTTS(api_key=api_key),
        'elevenlabs': lambda: ElevenLabsTTS(api_key=api_key),
    }
    
    if model_name.lower() not in tts_models:
        raise ValueError(f"Unsupported TTS model: {model_name}. Available models: {list(tts_models.keys())}")
    
    tts = tts_models[model_name.lower()]()
    
    # Set initial language if provided
    if kwargs.get('default_language'):
        if hasattr(tts, 'set_language'):
            tts.set_language(kwargs['default_language'])
            
    return tts