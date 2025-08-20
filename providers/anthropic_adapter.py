"""
Anthropic adapter that returns structured JSON matching the unified schema.
"""

import os
from typing import Dict, Any, Optional
from utils.json_utils import coerce_json
from core.schema import validate_result, get_fallback_result
from core.prompt import make_prompt

def get_anthropic_key() -> Optional[str]:
    """Get Anthropic API key from environment"""
    return os.getenv('ANTHROPIC_API_KEY')

def call_anthropic_api(prompt: str) -> Optional[str]:
    """Make API call to Anthropic with timeout and error handling"""
    try:
        import anthropic
        import httpx
        
        api_key = get_anthropic_key()
        if not api_key:
            return None
        
        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(timeout=20.0)
        )
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            temperature=0.2,
            system="You are a visibility analyst. Return ONLY valid JSON matching the exact schema requested. No markdown, no explanations.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract text from response
        if hasattr(message, 'content') and message.content:
            if isinstance(message.content, list) and len(message.content) > 0:
                return message.content[0].text
            elif hasattr(message.content, 'text'):
                return message.content.text
            else:
                return str(message.content)
        
        return None
        
    except Exception as e:
        # Log error but don't crash
        print(f"Anthropic API error: {str(e)}")
        return None

def analyze_with_anthropic(entity: str, category: str) -> Dict[str, Any]:
    """
    Analyze entity visibility using Anthropic API.
    
    Args:
        entity: Entity name (normalized)
        category: Category hint
        
    Returns:
        Dict matching unified schema or structured fallback
    """
    prompt = make_prompt(entity, category)
    
    # Make API call
    response_text = call_anthropic_api(prompt)
    if not response_text:
        return get_fallback_result(entity, "Anthropic API call failed or timed out.")
    
    # Try to parse JSON
    result = coerce_json(response_text)
    if not result:
        return get_fallback_result(entity, "Could not parse Anthropic response as valid JSON.")
    
    # Validate against schema
    if not validate_result(result):
        return get_fallback_result(entity, "Anthropic response did not match required schema.")
    
    return result 