"""
Robust JSON coercion utilities for handling LLM text output.
"""

import json
import re
from typing import Dict, Any, Optional

def coerce_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from LLM text output.
    Attempts multiple strategies to extract valid JSON from potentially messy LLM responses
    including markdown code fences, extra text, etc.
    
    Args:
        text: Raw text from LLM provider
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Strategy 1: Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from code fences
    code_fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first balanced JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: More aggressive extraction - find any { ... } block
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    
    # Strategy 5: Try to fix common JSON issues
    try:
        # Replace single quotes with double quotes
        fixed_text = re.sub(r"'([^']*)':", r'"\1":', text)
        fixed_text = re.sub(r": '([^']*)'", r': "\1"', fixed_text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    return None

def sanitize_json_string(text: str) -> str:
    """
    Clean up a string to be JSON-safe.
    
    Args:
        text: Input string that may contain problematic characters
        
    Returns:
        JSON-safe string
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove or escape problematic characters
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = text.strip()
    
    # Limit length
    if len(text) > 500:
        text = text[:497] + "..."
    
    return text 