from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging
import json
from typing import Dict
from dotenv import load_dotenv
from starlette.websockets import WebSocketState

from voice_pipeline import (
    VoicePipelineAgent,
    AudioData,
    ConversationContext,
    create_openai_llm,
    create_whisper_stt,
    create_cartesia_tts,
    SimpleEndpointingVAD,
    EOUTurnDetector
)
from voice_pipeline.api.models import VoiceConfig, AssistantConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
if not CARTESIA_API_KEY:
    logger.warning("CARTESIA_API_KEY not found in environment variables")

# Create component instances
stt = create_whisper_stt(model_size="base")
llm = create_openai_llm(api_key=OPENAI_API_KEY, model="gpt-4o")
tts = create_cartesia_tts(api_key=CARTESIA_API_KEY)
vad = SimpleEndpointingVAD()
turn_detector = EOUTurnDetector()

# Store active connections and their agents
active_agents: Dict[str, VoicePipelineAgent] = {}

@app.websocket("/ws/assistant")
async def websocket_assistant(websocket: WebSocket):
    """WebSocket endpoint for the voice assistant"""
    await websocket.accept()
    
    # Generate a unique ID for this connection
    connection_id = str(uuid.uuid4())
    
    # Create initial conversation context
    initial_ctx = ConversationContext(
        system_prompt="You are a helpful voice assistant. Keep responses concise and conversational."
    )
    
    # Create the voice pipeline agent
    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        turn_detector=turn_detector,
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx
    )
    
    # Store agent
    active_agents[connection_id] = agent
    
    logger.info(f"New voice assistant connection: {connection_id}")
    
    try:
        while True:
            # Receive message
            message = await websocket.receive()
            
            # Handle different message types
            if "bytes" in message:
                # Audio data case
                audio_data = message["bytes"]
                logger.info(f"Received audio data: {len(audio_data)} bytes")
                
                # Process audio through pipeline
                audio = AudioData(data=audio_data, format="wav")
                result = await agent.process_audio(audio)
                
                # Send transcription result
                if result["transcription"]:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result["transcription"].text
                    })
                
                # Send text response
                if result["llm_response"]:
                    await websocket.send_json({
                        "type": "text_response",
                        "text": result["llm_response"].text
                    })
                
                # Send audio response
                if result["audio_response"] and websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send({
                        "type": "websocket.send", 
                        "bytes": result["audio_response"].audio, 
                        "subprotocol": f"audio/{result['audio_response'].format}"
                    })
                    
            elif "text" in message:
                # Text message for configuration or commands
                text_data = message["text"]
                
                try:
                    data = json.loads(text_data)
                    
                    # Handle configuration updates
                    if data.get("type") == "config":
                        config_data = data.get("config", {})
                        
                        # Update system prompt if provided
                        if "system_prompt" in config_data:
                            agent.update_system_prompt(config_data["system_prompt"])
                            
                        # Update voice settings if provided
                        if "voice" in config_data and hasattr(agent.tts, "voice_id"):
                            voice_data = config_data["voice"]
                            if "voice_id" in voice_data:
                                agent.tts.voice_id = voice_data["voice_id"]
                        
                        await websocket.send_json({
                            "type": "config_updated",
                            "success": True
                        })
                        
                    # Handle conversation history commands
                    elif data.get("type") == "history":
                        action = data.get("action")
                        if action == "clear":
                            agent.clear_conversation()
                            await websocket.send_json({
                                "type": "history_cleared",
                                "success": True
                            })
                        elif action == "get":
                            history = agent.chat_ctx.messages
                            await websocket.send_json({
                                "type": "history",
                                "data": [{"role": msg.role, "content": msg.content} for msg in history]
                            })
                            
                    # Handle text-only input (no audio)
                    elif data.get("type") == "text_input":
                        user_input = data.get("text", "")
                        if user_input:
                            # Process text through pipeline
                            result = await agent.process_text(user_input)
                            
                            # Send text response
                            if result["llm_response"]:
                                await websocket.send_json({
                                    "type": "text_response",
                                    "text": result["llm_response"].text
                                })
                            
                            # Send audio response if requested
                            if data.get("tts", True) and result["audio_response"]:
                                await websocket.send({
                                    "type": "websocket.send", 
                                    "bytes": result["audio_response"].audio, 
                                    "subprotocol": f"audio/{result['audio_response'].format}"
                                })
                
                except json.JSONDecodeError:
                    # Handle plain text input
                    user_input = text_data
                    result = await agent.process_text(user_input)
                    
                    # Send text response
                    if result["llm_response"]:
                        await websocket.send_json({
                            "type": "text_response",
                            "text": result["llm_response"].text
                        })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Clean up
        if connection_id in active_agents:
            del active_agents[connection_id]

@app.get("/")
async def root():
    return {
        "message": "Voice Assistant API is running",
        "endpoints": {
            "ws_assistant": "/ws/assistant"
        }
    }