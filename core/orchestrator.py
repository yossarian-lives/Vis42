"""
FIXED Orchestrator - Main orchestrator that coordinates providers and merges results.
This version fixes the consistent score issue by ensuring proper API calls.
"""

import statistics
import json
import os
from typing import Dict, Any, List, Optional
from utils.entity import normalize_entity
from core.schema import get_fallback_result, validate_result

def get_available_providers() -> Dict[str, bool]:
    """Check which providers have API keys available"""
    try:
        import streamlit as st
        return {
            "openai": bool(st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
            "gemini": bool(st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"))
        }
    except:
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY"))
        }

def call_openai_fixed(entity: str, category: str) -> Optional[Dict[str, Any]]:
    """Fixed OpenAI API call"""
    try:
        import openai
        import streamlit as st
        
        # Get API key
        api_key = None
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""Analyze the visibility of "{entity}" in the {category or 'general'} space.

Return ONLY a JSON object with this exact structure:
{{
    "entity": "{entity}",
    "category": "{category or 'auto-detected'}",
    "overall_score": 75,
    "breakdown": {{
        "recognition": 80,
        "media": 70,
        "context": 75,
        "competitors": 85,
        "consistency": 80
    }},
    "notes": "Brief analysis of {entity}'s visibility and recognition",
    "sources": ["source1.com", "source2.com"]
}}

Base the scores on how well-known and documented {entity} is. Use realistic values between 0-100."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a brand visibility analyst. Return ONLY valid JSON matching the exact schema."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Validate the result has required fields
        if not isinstance(result.get("overall_score"), (int, float)):
            return None
        if not isinstance(result.get("breakdown"), dict):
            return None
        
        # Ensure entity and category are set correctly
        result["entity"] = entity
        result["category"] = category or "auto-detected"
        
        return result
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def call_anthropic_fixed(entity: str, category: str) -> Optional[Dict[str, Any]]:
    """Fixed Anthropic API call"""
    try:
        import anthropic
        import streamlit as st
        
        # Get API key
        api_key = None
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            return None
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""Analyze the visibility of "{entity}" in the {category or 'general'} space.

Return ONLY a JSON object with this exact structure:
{{
    "entity": "{entity}",
    "category": "{category or 'auto-detected'}",
    "overall_score": 75,
    "breakdown": {{
        "recognition": 80,
        "media": 70,
        "context": 75,
        "competitors": 85,
        "consistency": 80
    }},
    "notes": "Brief analysis of {entity}'s visibility and recognition",
    "sources": ["source1.com", "source2.com"]
}}

Base the scores on how well-known and documented {entity} is. Use realistic values between 0-100."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            system="You are a brand visibility analyst. Return ONLY valid JSON matching the exact schema.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = json.loads(result_text)
        
        # Ensure entity and category are set correctly
        result["entity"] = entity
        result["category"] = category or "auto-detected"
        
        return result
        
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return None

def call_gemini_fixed(entity: str, category: str) -> Optional[Dict[str, Any]]:
    """Fixed Gemini API call"""
    try:
        import google.generativeai as genai
        import streamlit as st
        
        # Get API key
        api_key = None
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""Analyze the visibility of "{entity}" in the {category or 'general'} space.

Return ONLY a JSON object with this exact structure:
{{
    "entity": "{entity}",
    "category": "{category or 'auto-detected'}",
    "overall_score": 75,
    "breakdown": {{
        "recognition": 80,
        "media": 70,
        "context": 75,
        "competitors": 85,
        "consistency": 80
    }},
    "notes": "Brief analysis of {entity}'s visibility and recognition",
    "sources": ["source1.com", "source2.com"]
}}

Base the scores on how well-known and documented {entity} is. Use realistic values between 0-100. Return ONLY JSON, no other text."""

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3}
        )
        
        result_text = response.text
        result = json.loads(result_text)
        
        # Ensure entity and category are set correctly
        result["entity"] = entity
        result["category"] = category or "auto-detected"
        
        return result
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_simulation_result(entity: str, category: str) -> Dict[str, Any]:
    """Generate realistic simulation when APIs fail"""
    import hashlib
    import random
    
    # Generate consistent but varied scores based on entity
    seed = int(hashlib.md5(entity.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    entity_lower = entity.lower()
    
    # Base scores on entity recognition patterns
    if any(term in entity_lower for term in ['tesla', 'apple', 'google', 'microsoft', 'amazon', 'meta']):
        base_range = (75, 95)
    elif any(term in entity_lower for term in ['vuori', 'anthropic', 'openai', 'startup']):
        base_range = (55, 80)
    elif any(term in entity_lower for term in ['ai', 'machine learning', 'blockchain']):
        base_range = (60, 85)
    else:
        base_range = (35, 70)
    
    overall_score = random.randint(*base_range)
    
    # Generate breakdown scores around the overall score
    variation = 15
    breakdown = {
        "recognition": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "media": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "context": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "competitors": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "consistency": max(0, min(100, overall_score + random.randint(-10, 10)))  # Less variation for consistency
    }
    
    return {
        "entity": entity,
        "category": category or "auto-detected",
        "overall_score": overall_score,
        "breakdown": breakdown,
        "notes": f"Simulation analysis for {entity}. These scores are generated based on typical recognition patterns. Add API keys for real LLM analysis.",
        "sources": ["simulation.demo", "example.data", "mock.source"]
    }

def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple provider results using median scoring"""
    if not results:
        return get_fallback_result("unknown", "No provider responded.")
    
    if len(results) == 1:
        return results[0]
    
    # Use first result as base
    base_result = results[0]
    entity = base_result["entity"]
    category = base_result["category"]
    
    # Collect all breakdown scores for median calculation
    breakdown_scores = {
        "recognition": [],
        "media": [],
        "context": [],
        "competitors": [],
        "consistency": []
    }
    
    overall_scores = []
    
    for result in results:
        overall_scores.append(result["overall_score"])
        for metric, score in result["breakdown"].items():
            if metric in breakdown_scores:
                breakdown_scores[metric].append(score)
    
    # Calculate median scores
    merged_breakdown = {}
    for metric, scores in breakdown_scores.items():
        merged_breakdown[metric] = int(statistics.median(scores)) if scores else 40
    
    # Calculate median overall score
    overall_score = int(statistics.median(overall_scores)) if overall_scores else 40
    
    # Merge notes and sources
    all_notes = [result.get("notes", "") for result in results if result.get("notes")]
    merged_notes = " | ".join(all_notes)[:600]
    
    all_sources = []
    for result in results:
        all_sources.extend(result.get("sources", []))
    # Deduplicate sources
    unique_sources = list(dict.fromkeys(all_sources))[:8]
    
    return {
        "entity": entity,
        "category": category,
        "overall_score": overall_score,
        "breakdown": merged_breakdown,
        "notes": merged_notes or f"Merged analysis from {len(results)} providers.",
        "sources": unique_sources
    }

def analyze_entity(raw_entity: str, selected_providers: List[str] = None) -> Dict[str, Any]:
    """
    FIXED Main orchestration function that properly calls APIs.
    
    Args:
        raw_entity: Raw entity string from UI input
        selected_providers: List of provider names to use (optional)
        
    Returns:
        Final merged analysis result
    """
    if not raw_entity or not raw_entity.strip():
        return get_fallback_result("", "Empty entity provided.")
    
    # Step 1: Normalize entity
    normalized_entity = normalize_entity(raw_entity)
    
    # Step 2: Guess category (simple version)
    category = "general"  # For now, use simple category
    
    # Step 3: Determine available providers
    available_providers = get_available_providers()
    
    if selected_providers:
        # Filter to only selected and available providers
        providers_to_use = [p for p in selected_providers if available_providers.get(p, False)]
    else:
        # Use all available providers
        providers_to_use = [p for p, available in available_providers.items() if available]
    
    # Step 4: Call each enabled provider with fixed functions
    results = []
    
    if "openai" in providers_to_use:
        result = call_openai_fixed(normalized_entity, category)
        if result:
            results.append(result)
    
    if "anthropic" in providers_to_use:
        result = call_anthropic_fixed(normalized_entity, category)
        if result:
            results.append(result)
    
    if "gemini" in providers_to_use:
        result = call_gemini_fixed(normalized_entity, category)
        if result:
            results.append(result)
    
    # Step 5: If no real results, use simulation
    if not results:
        simulation_result = generate_simulation_result(normalized_entity, category)
        return simulation_result
    
    # Step 6: Merge results if multiple providers succeeded
    return merge_results(results) 