import logging
from typing import Dict, Any

from voice_pipeline.core.interfaces import LLMInterface
from voice_pipeline.core.models import ConversationContext, LLMResponse

logger = logging.getLogger(__name__)

class OpenAILLM(LLMInterface):
    """Implementation of LLM using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize with OpenAI API key and model"""
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    async def generate_response(self, 
                               context: ConversationContext,
                               temperature: float = 0.7) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            messages = context.get_messages()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=300  # Adjust as needed
            )
            
            if response.choices and len(response.choices) > 0:
                assistant_message = response.choices[0].message.content
                return LLMResponse(text=assistant_message)
            else:
                logger.error("Empty response from OpenAI")
                return LLMResponse(text="I'm sorry, I couldn't generate a response.")
                
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return LLMResponse(text=f"I'm sorry, there was an error processing your request.")