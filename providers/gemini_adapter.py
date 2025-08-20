"""
Gemini adapter that returns structured JSON matching the unified schema.
"""

import os
from typing import Dict, Any, Optional
from utils.json_utils import coerce_json
from core.schema import validate_result, get_fallback_result
from core.prompt import make_prompt

def get_gemini_key() -> Optional[str]:
    """Get Gemini API key from environment"""
    return os.getenv('GEMINI_API_KEY')

def call_gemini_api(prompt: str) -> Optional[str]:
    """Make API call to Gemini with timeout and error handling"""
    try:
        import google.generativeai as genai
        
        api_key = get_gemini_key()
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 800,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction="You are a visibility analyst. Return ONLY valid JSON matching the exact schema requested. No markdown, no explanations."
        )
        
        # Add JSON format instruction to prompt
        enhanced_prompt = prompt + "\n\nIMPORTANT: Return ONLY the JSON object, no other text."
        
        response = model.generate_content(enhanced_prompt)
        
        if response and response.text:
            return response.text
        
        return None
        
    except Exception as e:
        # Log error but don't crash
        print(f"Gemini API error: {str(e)}")
        return None

def analyze_with_gemini(entity: str, category: str) -> Dict[str, Any]:
    """
    Analyze entity visibility using Gemini API.
    
    Args:
        entity: Entity name (normalized)
        category: Category hint
        
    Returns:
        Dict matching unified schema or structured fallback
    """
    prompt = make_prompt(entity, category)
    
    # Make API call
    response_text = call_gemini_api(prompt)
    if not response_text:
        return get_fallback_result(entity, "Gemini API call failed or timed out.")
    
    # Try to parse JSON
    result = coerce_json(response_text)
    if not result:
        return get_fallback_result(entity, "Could not parse Gemini response as valid JSON.")
    
    # Validate against schema
    if not validate_result(result):
        return get_fallback_result(entity, "Gemini response did not match required schema.")
    
    return result 