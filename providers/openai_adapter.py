"""
OpenAI adapter that returns structured JSON matching the unified schema.
"""

import os
from typing import Dict, Any, Optional
from utils.json_utils import coerce_json
from core.schema import validate_result, get_fallback_result
from core.prompt import make_prompt

def get_openai_key() -> Optional[str]:
    """Get OpenAI API key from environment"""
    return os.getenv('OPENAI_API_KEY')

def call_openai_api(prompt: str) -> Optional[str]:
    """Make API call to OpenAI with timeout and error handling"""
    try:
        from openai import OpenAI
        import httpx
        
        api_key = get_openai_key()
        if not api_key:
            return None
        
        client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(timeout=20.0)
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a visibility analyst. Return ONLY valid JSON matching the exact schema requested. No markdown, no explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Log error but don't crash
        print(f"OpenAI API error: {str(e)}")
        return None

def analyze_with_openai(entity: str, category: str) -> Dict[str, Any]:
    """
    Analyze entity visibility using OpenAI API.
    
    Args:
        entity: Entity name (normalized)
        category: Category hint
        
    Returns:
        Dict matching unified schema or structured fallback
    """
    prompt = make_prompt(entity, category)
    
    # Make API call
    response_text = call_openai_api(prompt)
    if not response_text:
        return get_fallback_result(entity, "OpenAI API call failed or timed out.")
    
    # Try to parse JSON
    result = coerce_json(response_text)
    if not result:
        return get_fallback_result(entity, "Could not parse OpenAI response as valid JSON.")
    
    # Validate against schema
    if not validate_result(result):
        return get_fallback_result(entity, "OpenAI response did not match required schema.")
    
    return result 