"""
FIXED OpenAI adapter using modern SDK pattern
This fixes the simulation fallback issue by using proper error handling
"""

import os
import json
import traceback
from typing import Dict, Any, Optional

def get_openai_key() -> Optional[str]:
    """Get OpenAI API key from Streamlit secrets or environment"""
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key and isinstance(key, str) and key.strip():
            return key.strip()
    except:
        pass
    
    # Fallback to environment
    key = os.getenv("OPENAI_API_KEY")
    if key and isinstance(key, str) and key.strip():
        return key.strip()
    
    return None

def get_openai_model() -> str:
    """Get OpenAI model name with safe default"""
    try:
        import streamlit as st
        model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
        if model and isinstance(model, str):
            return model.strip()
    except:
        pass
    
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def test_openai_connection(entity: str = "Tesla") -> tuple[bool, str]:
    """Test OpenAI connection with detailed error reporting"""
    api_key = get_openai_key()
    if not api_key:
        return False, "No OpenAI API key found"
    
    try:
        # Use modern OpenAI SDK pattern
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = get_openai_model()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"In one sentence, what is {entity}?"}
            ],
            temperature=0.2,
            max_tokens=50,
            timeout=30
        )
        
        text = response.choices[0].message.content.strip()
        return True, f"Model: {model} · OK · '{text[:100]}'"
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Provide specific guidance for common errors
        if "401" in error_msg or "Unauthorized" in error_msg:
            guidance = "Check your API key - it may be invalid or expired"
        elif "404" in error_msg or "NotFound" in error_msg:
            guidance = f"Model '{get_openai_model()}' not available to your account"
        elif "429" in error_msg or "rate_limit" in error_msg.lower():
            guidance = "Rate limit exceeded - wait or upgrade your plan"
        elif "403" in error_msg or "Permission" in error_msg:
            guidance = "Permission denied - check your organization/project settings"
        else:
            guidance = "Check network connection and API status"
        
        return False, f"{error_type}: {error_msg} | {guidance}"

def analyze_with_openai(entity: str, category: str) -> Dict[str, Any]:
    """
    Analyze entity visibility using OpenAI API with modern SDK pattern.
    Returns proper error information instead of silent fallback.
    """
    api_key = get_openai_key()
    if not api_key:
        return {
            "error": "No OpenAI API key available",
            "entity": entity,
            "category": category,
            "simulated": True,
            "reason": "Missing API key"
        }
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = get_openai_model()
        
        # Construct the analysis prompt
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with this EXACT structure:
{{
    "entity": "{entity}",
    "category": "{category}",
    "overall_score": 75,
    "breakdown": {{
        "recognition": 80,
        "media": 70,
        "context": 75,
        "competitors": 85,
        "consistency": 80
    }},
    "notes": "Brief analysis of {entity}'s visibility and recognition in LLM knowledge bases",
    "sources": ["source1.com", "source2.com", "source3.com"]
}}

Base the scores (0-100) on:
- Recognition: How well LLMs know {entity}
- Media: Coverage in news/articles  
- Context: Industry position understanding
- Competitors: Awareness of alternatives
- Consistency: Stability across queries

Use realistic scores based on {entity}'s actual prominence and visibility."""

        # Make the API call with modern SDK
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an LLM visibility analyst. Return ONLY valid JSON matching the exact schema requested. No markdown, no explanations, just JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"},  # Force JSON mode
            timeout=45
        )
        
        # Extract and parse the response
        result_text = response.choices[0].message.content
        if not result_text or not result_text.strip():
            raise ValueError("Empty response from OpenAI")
        
        # Parse JSON
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as je:
            raise ValueError(f"Invalid JSON response: {je}") from je
        
        # Validate required fields
        required_fields = ["entity", "category", "overall_score", "breakdown", "notes", "sources"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate breakdown structure
        breakdown = result.get("breakdown", {})
        required_breakdown = ["recognition", "media", "context", "competitors", "consistency"]
        for field in required_breakdown:
            if field not in breakdown:
                raise ValueError(f"Missing breakdown field: {field}")
            if not isinstance(breakdown[field], (int, float)):
                raise ValueError(f"Invalid breakdown score for {field}: {breakdown[field]}")
        
        # Validate overall score
        overall_score = result.get("overall_score")
        if not isinstance(overall_score, (int, float)) or overall_score < 0 or overall_score > 100:
            raise ValueError(f"Invalid overall_score: {overall_score}")
        
        # Ensure entity and category match
        result["entity"] = entity
        result["category"] = category
        result["provider"] = "OpenAI"
        result["model"] = model
        result["simulated"] = False
        
        return result
        
    except Exception as e:
        # Detailed error reporting
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Get traceback for debugging
        tb = traceback.format_exc(limit=3)
        
        # Return structured error instead of silent fallback
        return {
            "error": f"OpenAI API call failed: {error_type}: {error_msg}",
            "entity": entity,
            "category": category,
            "provider": "OpenAI", 
            "model": get_openai_model(),
            "simulated": True,
            "reason": f"API failure: {error_type}",
            "traceback": tb,
            "api_key_present": bool(api_key),
            "api_key_preview": api_key[:10] + "..." if api_key else None
        }

def get_fallback_result(entity: str, category: str, reason: str) -> Dict[str, Any]:
    """Generate realistic fallback result when API fails"""
    import hashlib
    import random
    
    # Generate consistent scores based on entity
    seed = int(hashlib.md5(entity.lower().encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    # Entity-based scoring
    entity_lower = entity.lower()
    if any(term in entity_lower for term in ['tesla', 'apple', 'google', 'microsoft', 'amazon']):
        base_score = random.randint(80, 95)
    elif any(term in entity_lower for term in ['vuori', 'anthropic', 'openai', 'claude']):
        base_score = random.randint(65, 85)
    elif any(term in entity_lower for term in ['ai', 'startup', 'tech']):
        base_score = random.randint(50, 75)
    else:
        base_score = random.randint(35, 65)
    
    # Generate breakdown with variation
    variation = 20
    breakdown = {
        "recognition": max(0, min(100, base_score + random.randint(-variation, variation))),
        "media": max(0, min(100, base_score + random.randint(-variation, variation))),
        "context": max(0, min(100, base_score + random.randint(-variation, variation))),
        "competitors": max(0, min(100, base_score + random.randint(-variation, variation))),
        "consistency": max(0, min(100, base_score + random.randint(-10, 10)))
    }
    
    return {
        "entity": entity,
        "category": category,
        "overall_score": base_score,
        "breakdown": breakdown,
        "notes": f"FALLBACK SIMULATION: {reason}. Add valid OpenAI API key for real analysis.",
        "sources": ["simulation.demo", "fallback.data", "mock.source"],
        "provider": "OpenAI (Simulated)",
        "simulated": True,
        "reason": reason
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing OpenAI adapter...")
    
    # Test connection
    ok, msg = test_openai_connection("Tesla")
    print(f"Connection test: {ok} - {msg}")
    
    # Test analysis (will fail without API key)
    result = analyze_with_openai("Tesla", "automotive")
    print(f"Analysis result: {result.get('simulated', False)}")
    
    if result.get('simulated'):
        print(f"Fallback reason: {result.get('reason')}")
        if result.get('error'):
            print(f"Error: {result['error']}") 