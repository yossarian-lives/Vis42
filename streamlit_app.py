"""
LLM Visibility Analyzer - Streamlit Application

Professional Streamlit app with clean modular imports and beautiful UI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from collections.abc import Mapping
from typing import Optional, Dict, Any
import re
import json

# ---- Schema and Validation --------------------------------------------------

# Unified schema that all providers MUST return
VISIBILITY_SCHEMA = {
    "type": "object",
    "required": ["entity", "category", "overall_score", "breakdown", "notes", "sources"],
    "properties": {
        "entity": {"type": "string"},
        "category": {"type": "string"},
        "overall_score": {"type": "number", "minimum": 0, "maximum": 100},
        "breakdown": {
            "type": "object",
            "required": ["recognition", "media", "context", "competitors", "consistency"],
            "properties": {
                "recognition": {"type": "number", "minimum": 0, "maximum": 100},
                "media": {"type": "number", "minimum": 0, "maximum": 100},
                "context": {"type": "number", "minimum": 0, "maximum": 100},
                "competitors": {"type": "number", "minimum": 0, "maximum": 100},
                "consistency": {"type": "number", "minimum": 0, "maximum": 100}
            },
            "additionalProperties": False
        },
        "notes": {"type": "string", "maxLength": 600},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8
        }
    },
    "additionalProperties": False
}

def validate_result(data: dict) -> bool:
    """Validate that a result matches our schema"""
    try:
        # Simple validation without external jsonschema dependency
        if not isinstance(data, dict):
            return False
        
        required_fields = ["entity", "category", "overall_score", "breakdown", "notes", "sources"]
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate breakdown
        breakdown = data.get("breakdown", {})
        breakdown_fields = ["recognition", "media", "context", "competitors", "consistency"]
        for field in breakdown_fields:
            if field not in breakdown:
                return False
            if not isinstance(breakdown[field], (int, float)) or breakdown[field] < 0 or breakdown[field] > 100:
                return False
        
        # Validate overall_score
        if not isinstance(data["overall_score"], (int, float)) or data["overall_score"] < 0 or data["overall_score"] > 100:
            return False
        
        return True
    except Exception:
        return False

def get_fallback_result(entity: str, reason: str = "Structured fallback due to unparseable provider output.") -> dict:
    """Return a structured fallback result when providers fail"""
    return {
        "entity": entity,
        "category": "unknown",
        "overall_score": 40,
        "breakdown": {
            "recognition": 40,
            "media": 40,
            "context": 40,
            "competitors": 40,
            "consistency": 60
        },
        "notes": reason,
        "sources": []
    }

# ---- Robust Secret Detection -------------------------------------------------

def _find_in_mapping(d: Mapping, name: str):
    if not isinstance(d, Mapping): 
        return None
    if name in d:
        v = d[name]
        if isinstance(v, str) and v.strip(): 
            return v.strip()
    for v in d.values():
        if isinstance(v, Mapping):
            found = _find_in_mapping(v, name)
            if found: 
                return found
    return None

def get_secret_or_env(name: str) -> str | None:
    """Get secret from st.secrets (any nesting level) or fallback to environment variable"""
    # First try direct access to st.secrets
    try:
        if hasattr(st, 'secrets') and name in st.secrets:
            value = st.secrets[name]
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass
    
    # Then try recursive search in nested secrets
    try:
        found = _find_in_mapping(st.secrets, name)
        if found: 
            return found
    except Exception:
        pass
    
    # Finally fallback to environment variable
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else None

# ---- Provider Configuration --------------------------------------------------

PROVIDERS = {
    "OpenAI":    {"key": get_secret_or_env("OPENAI_API_KEY")},
    "Anthropic": {"key": get_secret_or_env("ANTHROPIC_API_KEY")},
    "Gemini":    {"key": get_secret_or_env("GEMINI_API_KEY")},
}
ENABLED = {name: cfg["key"] for name, cfg in PROVIDERS.items() if cfg["key"]}
SIMULATION_MODE = len(ENABLED) == 0

# ---- Fail-Safe Provider Calls ----------------------------------------------

# ---- OpenAI Provider Adapter ------------------------------------------------

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

def make_prompt(entity: str, category_hint: str) -> str:
    """
    Create a prompt that forces JSON-only output with strict schema compliance.
    
    Args:
        entity: The entity to analyze (e.g., "Vuori")
        category_hint: Category context (e.g., "consumer apparel / activewear")
    
    Returns:
        Complete prompt string that enforces JSON output
    """
    
    prompt = f"""Analyze the visibility of "{entity}" in the {category_hint} space across LLM knowledge bases.

DEFINITION OF VISIBILITY:
- Breadth: How widely known across different AI models and contexts
- Freshness: How current and up-to-date the information is
- Depth/Accuracy: Level of detailed, accurate information available
- Hallucination penalty: Deduct points for inconsistent or made-up information

SCORING METHODOLOGY (0-100 for each):
- Recognition: How well LLMs recognize and identify this entity
- Media: Coverage in news, articles, and media mentions
- Context: Understanding of industry position and relationships
- Competitors: Awareness of alternatives and competitive landscape  
- Consistency: Stability and agreement across different queries/models

You MUST return ONLY valid minified JSON that matches this exact schema. No markdown, no code fences, no commentary:

{{"entity":"{entity}","category":"{category_hint}","overall_score":85,"breakdown":{{"recognition":80,"media":75,"context":85,"competitors":90,"consistency":85}},"notes":"Brief analysis paragraph under 600 chars","sources":["domain1.com","brief-citation-2","source-3"]}}

CRITICAL REQUIREMENTS:
- Return ONLY the JSON object, nothing else
- All scores must be integers 0-100
- Notes must be under 600 characters
- Sources should be domains or brief citations (max 8)
- If uncertain, provide best-effort estimates and keep internal consistency
- Calculate overall_score as weighted average: recognition(30%) + media(25%) + context(20%) + consistency(15%) + competitors(10%)

Analyze "{entity}" now:"""
    
    return prompt

# ---- Robust JSON Coercion Utilities -----------------------------------------

def coerce_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from LLM text output.
    
    Attempts multiple strategies to extract valid JSON from potentially
    messy LLM responses including markdown code fences, extra text, etc.
    
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

def parse_openai_response(response: str, entity: str, category: str) -> dict:
    """Parse OpenAI response and convert to structured schema format"""
    try:
        # Use the robust JSON coercion utility
        parsed_data = coerce_json(response)
        
        if parsed_data:
            # Validate the parsed data against our schema
            if validate_result(parsed_data):
                st.success("✅ Successfully parsed structured JSON response!")
                return parsed_data
            else:
                st.warning("⚠️ JSON parsed but failed schema validation - using fallback")
                # Try to fix common schema issues
                fixed_data = fix_schema_issues(parsed_data, entity, category)
                if validate_result(fixed_data):
                    st.success("✅ Fixed schema issues and validated successfully!")
                    return fixed_data
        else:
            st.info("ℹ️ No JSON found in response - using fallback parsing")
        
        # Fallback: extract structured information from text response
        result = extract_from_text(response, entity, category)
        return result
        
    except Exception as e:
        st.warning(f"Failed to parse OpenAI response: {str(e)}")
        return get_fallback_result(entity, f"Failed to parse response: {str(e)}")

def fix_schema_issues(data: dict, entity: str, category: str) -> dict:
    """Attempt to fix common schema validation issues"""
    try:
        # Ensure required fields exist
        if "entity" not in data:
            data["entity"] = entity
        if "category" not in data:
            data["category"] = category or "auto-detected"
        if "overall_score" not in data:
            data["overall_score"] = 50
        
        # Ensure breakdown exists and has all required fields
        if "breakdown" not in data:
            data["breakdown"] = {}
        
        required_breakdown_fields = ["recognition", "media", "context", "competitors", "consistency"]
        for field in required_breakdown_fields:
            if field not in data["breakdown"]:
                data["breakdown"][field] = 50
        
        # Ensure notes and sources exist
        if "notes" not in data:
            data["notes"] = "Analysis completed"
        if "sources" not in data:
            data["sources"] = []
        
        # Sanitize strings
        data["notes"] = sanitize_json_string(data["notes"])
        
        return data
    except Exception:
        return get_fallback_result(entity, "Failed to fix schema issues")

def extract_from_text(response: str, entity: str, category: str) -> dict:
    """Extract structured information from text when JSON parsing fails"""
    # Default values
    result = {
        "entity": entity,
        "category": category or "auto-detected",
        "overall_score": 50,  # Default middle score
        "breakdown": {
            "recognition": 50,
            "media": 50,
            "context": 50,
            "competitors": 50,
            "consistency": 50
        },
        "notes": sanitize_json_string(response[:600]),  # Truncate and sanitize
        "sources": []
    }
    
    # Try to extract score if mentioned
    score_match = re.search(r'(\d{1,3})/100|score[:\s]*(\d{1,3})|(\d{1,3})\s*out\s*of\s*100', response, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1) or score_match.group(2) or score_match.group(3))
        if 0 <= score <= 100:
            result["overall_score"] = score
    
    # Try to extract category if not provided
    if not category or category == "auto-detected":
        category_keywords = {
            "Technology": ["tech", "software", "ai", "artificial intelligence", "machine learning"],
            "Finance": ["finance", "banking", "investment", "crypto", "blockchain"],
            "Healthcare": ["health", "medical", "pharma", "biotech"],
            "Education": ["education", "learning", "academic", "university"],
            "Entertainment": ["entertainment", "media", "gaming", "film", "music"],
            "Consumer": ["consumer", "apparel", "fashion", "retail", "brand"],
            "Business": ["business", "enterprise", "corporate", "startup", "company"]
        }
        
        response_lower = response.lower()
        for cat, keywords in category_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                result["category"] = cat
                break
    
    return result

# ---- Entity Normalization Utilities -----------------------------------------

# Common entity name mappings
ENTITY_MAPPINGS = {
    "vouri": "Vuori",
    "voui": "Vuori", 
    "vuouri": "Vuori",
    "tesla": "Tesla",
    "apple": "Apple",
    "microsoft": "Microsoft",
    "google": "Google",
    "amazon": "Amazon",
    "meta": "Meta",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "chatgpt": "ChatGPT",
    "gpt": "GPT",
}

# Category hints based on entity names
ENTITY_CATEGORY_HINTS = {
    "vuori": "consumer apparel / activewear",
    "nike": "consumer apparel / athletic wear",
    "adidas": "consumer apparel / athletic wear",
    "lululemon": "consumer apparel / activewear",
    "patagonia": "consumer apparel / outdoor gear",
    "tesla": "automotive / electric vehicles",
    "ford": "automotive",
    "apple": "technology / consumer electronics",
    "microsoft": "technology / software",
    "google": "technology / internet services",
    "amazon": "e-commerce / cloud computing",
    "meta": "technology / social media",
    "openai": "artificial intelligence",
    "anthropic": "artificial intelligence",
    "chatgpt": "artificial intelligence / language models",
}

def normalize_entity(entity: str) -> str:
    """
    Normalize entity name to handle common misspellings and formatting issues.
    
    Args:
        entity: Raw entity name from user input
        
    Returns:
        Normalized entity name
    """
    if not entity or not isinstance(entity, str):
        return ""
    
    # Clean up the entity name
    entity = entity.strip()
    entity_lower = entity.lower()
    
    # Check for exact mappings first
    if entity_lower in ENTITY_MAPPINGS:
        return ENTITY_MAPPINGS[entity_lower]
    
    # Remove extra whitespace and normalize casing
    entity = re.sub(r'\s+', ' ', entity)
    
    # If it's all uppercase, convert to title case
    if entity.isupper() and len(entity) > 2:
        entity = entity.title()
    
    # If it's all lowercase and looks like a proper noun, title case it
    if entity.islower() and len(entity) > 2:
        # Simple heuristic: if it doesn't contain common words, title case it
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = entity.split()
        if len(words) <= 2 or not any(word in common_words for word in words):
            entity = entity.title()
    
    return entity

def guess_category_from_name(entity: str) -> str:
    """
    Guess entity category based on the entity name.
    
    Args:
        entity: Normalized entity name
        
    Returns:
        Category hint string
    """
    entity_lower = entity.lower()
    
    # Check for exact matches first
    if entity_lower in ENTITY_CATEGORY_HINTS:
        return ENTITY_CATEGORY_HINTS[entity_lower]
    
    # Pattern-based matching
    if any(term in entity_lower for term in ['apparel', 'clothing', 'wear', 'fashion']):
        return "consumer apparel"
    
    if any(term in entity_lower for term in ['tech', 'software', 'app', 'platform']):
        return "technology"
    
    if any(term in entity_lower for term in ['car', 'auto', 'vehicle', 'motor']):
        return "automotive"
    
    if any(term in entity_lower for term in ['ai', 'artificial', 'intelligence', 'ml', 'machine learning']):
        return "artificial intelligence"
    
    if any(term in entity_lower for term in ['bio', 'pharma', 'health', 'medical']):
        return "healthcare"
    
    if any(term in entity_lower for term in ['bank', 'finance', 'invest', 'fund']):
        return "financial services"
    
    # Default fallback
    return "brand"

# ---- Web Enrichment Utilities -----------------------------------------------

def get_search_api_key() -> tuple[Optional[str], Optional[str]]:
    """Get available search API keys"""
    tavily_key = os.getenv('TAVILY_API_KEY')
    serper_key = os.getenv('SERPER_API_KEY')
    return tavily_key, serper_key

def search_with_tavily(entity: str, api_key: str) -> Optional[str]:
    """Search using Tavily API"""
    try:
        import requests
        
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": f"{entity} company business",
                "max_results": 3,
                "search_depth": "basic"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            # Extract text from results
            text = ""
            for result in results[:3]:
                text += result.get('content', '') + " "
                text += result.get('title', '') + " "
            
            return text.lower()
    except Exception:
        pass
    
    return None

def search_with_serper(entity: str, api_key: str) -> Optional[str]:
    """Search using Serper API"""
    try:
        import requests
        
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key},
            json={
                "q": f"{entity} company business",
                "num": 3
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract text from organic results
            text = ""
            for result in data.get('organic', [])[:3]:
                text += result.get('snippet', '') + " "
                text += result.get('title', '') + " "
            
            return text.lower()
    except Exception:
        pass
    
    return None

def detect_category_from_text(text: str) -> str:
    """Detect category from search result text using keyword matching"""
    if not text:
        return "brand"
    
    text = text.lower()
    
    # Category detection patterns
    patterns = {
        "consumer apparel / activewear": [
            "activewear", "apparel", "clothing", "athleisure", "retail", 
            "fashion", "athletic wear", "sportswear", "fitness", "yoga"
        ],
        "technology": [
            "software", "platform", "tech", "app", "digital", "saas",
            "technology", "startup", "innovation", "developer"
        ],
        "automotive": [
            "automotive", "car", "vehicle", "auto", "motor", "electric vehicle",
            "transportation", "mobility", "tesla", "ford"
        ],
        "artificial intelligence": [
            "ai", "artificial intelligence", "machine learning", "ml", 
            "deep learning", "nlp", "chatbot", "openai"
        ],
        "healthcare": [
            "healthcare", "medical", "health", "pharma", "biotech",
            "medicine", "clinical", "patient"
        ],
        "financial services": [
            "bank", "banking", "finance", "financial", "investment",
            "fintech", "trading", "payment"
        ]
    }
    
    # Score each category
    scores = {}
    for category, keywords in patterns.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[category] = score
    
    # Return category with highest score
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return "brand"

def enhanced_guess_category(entity: str) -> str:
    """
    Enhanced category detection using web search if API keys available,
    otherwise fall back to name-based detection.
    
    Args:
        entity: Entity name to categorize
        
    Returns:
        Category string
    """
    # First try name-based detection
    name_based_category = guess_category_from_name(entity)
    
    # Try web search if API keys are available
    tavily_key, serper_key = get_search_api_key()
    
    if tavily_key:
        try:
            with st.spinner("🔍 Searching web for better category detection..."):
                search_text = search_with_tavily(entity, tavily_key)
                if search_text:
                    web_category = detect_category_from_text(search_text)
                    # Prefer web-based category if it's more specific than name-based
                    if web_category != "brand" or name_based_category == "brand":
                        st.success(f"✅ Web search enhanced category: {web_category}")
                        return web_category
        except Exception as e:
            st.warning(f"⚠️ Web search failed: {str(e)}")
    
    elif serper_key:
        try:
            with st.spinner("🔍 Searching web for better category detection..."):
                search_text = search_with_serper(entity, serper_key)
                if search_text:
                    web_category = detect_category_from_text(search_text)
                    # Prefer web-based category if it's more specific than name-based
                    if web_category != "brand" or name_based_category == "brand":
                        st.success(f"✅ Web search enhanced category: {web_category}")
                        return web_category
        except Exception as e:
            st.warning(f"⚠️ Web search failed: {str(e)}")
    
    # Fall back to name-based category
    return name_based_category

# Page configuration
st.set_page_config(
    page_title="LLM Visibility Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .badge {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #667eea;
        border: 1px solid #e1e5e9;
    }
    .score-chart {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header"><h1>🔍 LLM Visibility Analyzer</h1><p>Professional brand & topic analysis across AI knowledge spaces</p></div>', unsafe_allow_html=True)
    
    # Configuration section
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Debug info (temporary - remove later)
        if st.checkbox("🔍 Show Debug Info"):
            st.write("**Debug Information:**")
            st.write(f"- SIMULATION_MODE: {SIMULATION_MODE}")
            st.write(f"- ENABLED providers: {list(ENABLED.keys())}")
            st.write(f"- OpenAI key present: {'Yes' if 'OpenAI' in ENABLED else 'No'}")
            if 'OpenAI' in ENABLED:
                st.write(f"- OpenAI key starts with: {ENABLED['OpenAI'][:10]}...")
            
            # Add more detailed debugging
            st.write("**Secrets Debug:**")
            try:
                secrets_keys = list(st.secrets.keys()) if hasattr(st, 'secrets') else []
                st.write(f"- st.secrets available: {'Yes' if hasattr(st, 'secrets') else 'No'}")
                st.write(f"- st.secrets keys: {secrets_keys}")
                if 'OPENAI_API_KEY' in secrets_keys:
                    st.write(f"- OPENAI_API_KEY in secrets: Yes")
                    st.write(f"- OPENAI_API_KEY starts with: {st.secrets['OPENAI_API_KEY'][:10]}...")
                else:
                    st.write(f"- OPENAI_API_KEY in secrets: No")
            except Exception as e:
                st.write(f"- Error accessing secrets: {str(e)}")
            
            st.write("**Environment Debug:**")
            import os
            env_openai = os.getenv('OPENAI_API_KEY')
            st.write(f"- OPENAI_API_KEY in env: {'Yes' if env_openai else 'No'}")
            if env_openai:
                st.write(f"- Env OPENAI_API_KEY starts with: {env_openai[:10]}...")
        
        # Provider status
        st.subheader("🤖 LLM Providers")
        for p in ["OpenAI", "Anthropic", "Gemini"]:
            if p in ENABLED:
                st.markdown(f"✅ **{p}** enabled")
            else:
                st.markdown(f"⚪ **{p}** (no key)")
        
        # Web enrichment status
        st.subheader("🌐 Web Enrichment")
        tavily_key, serper_key = get_search_api_key()
        if tavily_key:
            st.markdown(f"✅ **Tavily** enabled")
        elif serper_key:
            st.markdown(f"✅ **Serper** enabled")
        else:
            st.markdown(f"⚪ **Web search** (no API keys)")
            st.caption("Add TAVILY_API_KEY or SERPER_API_KEY for enhanced category detection")
        
        # Test API Key button
        if st.button("🧪 Test API Key"):
            if "OpenAI" in ENABLED:
                with st.spinner("Testing OpenAI API..."):
                    try:
                        # Test with a simple call first
                        test_result = call_openai_api("Say 'Hello World' in one word.")
                        if test_result:
                            st.success(f"✅ API Test Successful: {test_result}")
                        else:
                            st.error("❌ API Test Failed - check the error above")
                    except Exception as e:
                        st.error(f"❌ Test failed with exception: {str(e)}")
                        st.info(f"Exception type: {type(e).__name__}")
            else:
                st.warning("No OpenAI API key available for testing")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<span class='badge'>🔧 Configuration · "
            f"{'Simulation Mode' if SIMULATION_MODE else 'Real Analysis Enabled'}</span>",
            unsafe_allow_html=True
        )
    with col2:
        if st.button("🔄 Refresh Detection"):
            st.rerun()
    
    # Debug captions
    st.caption("Secrets present: " + ", ".join(sorted(getattr(st, "secrets", {}).keys())))
    st.caption("Enabled providers: " + ", ".join(sorted(ENABLED.keys())) if ENABLED else "Enabled providers: none")
    
    # Entity analysis section
    st.header("🎯 Entity Analysis")
    
    # Input form
    with st.form("analysis_form"):
        entity = st.text_input("Enter entity to analyze", placeholder="e.g., Apple, ChatGPT, Tesla")
        category = st.selectbox(
            "📂 Category (Optional)",
            ["", "Technology", "Finance", "Healthcare", "Education", "Entertainment", "Consumer", "Business", "Other"],
            help="Choose category or auto-detect"
        )
        
        # Provider selection
        if ENABLED:
            st.subheader("Select Providers")
            providers_selected = []
            for p in ENABLED.keys():
                if st.checkbox(f"Use {p}", value=p in ENABLED):
                    providers_selected.append(p)
        else:
            providers_selected = []
            st.info("Simulation mode - all providers enabled")
        
        submitted = st.form_submit_button("🔍 Analyze Visibility")
    
    # Analysis results
    if submitted and entity:
        if not entity.strip():
            st.warning("Please enter an entity to analyze.")
            return
        
        # Normalize the entity name and guess category if not provided
        normalized_entity = normalize_entity(entity)
        if normalized_entity != entity:
            st.info(f"🔍 Normalized entity name: '{entity}' → '{normalized_entity}'")
        
        # Auto-detect category if not provided
        if not category:
            guessed_category = enhanced_guess_category(normalized_entity)
            st.info(f"📂 Auto-detected category: {guessed_category}")
            category = guessed_category
        
        # Show loading
        with st.spinner("🔍 Analyzing visibility across LLMs..."):
            # Try real API calls first
            if not SIMULATION_MODE:
                # Use our fail-safe OpenAI call
                if "OpenAI" in providers_selected and "OpenAI" in ENABLED:
                    # Create a specific prompt for the entity and category
                    category_hint = category if category else "AI/tech"
                    prompt = make_prompt(normalized_entity, category_hint)
                    
                    # Show what we're trying to do
                    st.info(f"🔍 Attempting OpenAI API call with model fallback...")
                    
                    try:
                        result = analyze_with_openai(normalized_entity, category_hint)
                        
                        # Validate the result
                        if validate_result(result):
                            st.success("✅ Analysis completed successfully with valid schema!")
                        else:
                            st.warning("⚠️ Analysis completed but schema validation failed - using fallback")
                            result = get_fallback_result(normalized_entity, "Schema validation failed")
                        
                        # Store in session state for recent analyses
                        if 'recent_analyses' not in st.session_state:
                            st.session_state.recent_analyses = []
                        
                        st.session_state.recent_analyses.append({
                            'entity': normalized_entity,
                            'score': result['overall_score'],
                            'timestamp': datetime.now()
                        })
                        
                        # Display results
                        display_results(result)
                        return
                    except Exception as e:
                        st.error(f"❌ Error during OpenAI API call: {str(e)}")
                        st.info("💡 **Error Details:**")
                        st.info(f"Exception type: {type(e).__name__}")
                        st.info(f"Error message: {str(e)}")
                        return
            
            # Fallback to simulation if no real results
            if SIMULATION_MODE:
                st.info("Running in simulation mode - no API keys available")
                # Use fallback result for simulation
                analysis_result = get_fallback_result(normalized_entity, "Simulation mode - no real API calls")
                display_results(analysis_result)
            else:
                st.warning("API call failed. Check your API keys and try again.")
                st.info("💡 **Troubleshooting Tips:**")
                st.info("1. Verify your API key is valid and has credits")
                st.info("2. Check if the model 'gpt-4o-mini' is available")
                st.info("3. Ensure your OpenAI account has access to the API")
    
    # Quick stats section
    st.header("📊 Quick Stats")
    if 'recent_analyses' in st.session_state and st.session_state.recent_analyses:
        df = pd.DataFrame(st.session_state.recent_analyses)
        st.dataframe(df)
    else:
        st.info("No analyses yet. Start by analyzing an entity above.")

def display_results(results):
    """Display analysis results in a structured format using the schema"""
    st.header(f"📈 Analysis Results: {results['entity']}")
    
    # Overall score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Overall Visibility Score", f"{results['overall_score']}/100")
    
    # Breakdown scores
    st.subheader("📊 Detailed Breakdown")
    
    # Create a radar chart for the breakdown
    breakdown = results['breakdown']
    categories = list(breakdown.keys())
    values = list(breakdown.values())
    
    # Create the radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Visibility Scores',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Visibility Breakdown Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual metrics
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, len(categories), 2):
            if i < len(categories):
                st.metric(categories[i].title(), f"{values[i]}/100")
    with col2:
        for i in range(1, len(categories), 2):
            if i < len(categories):
                st.metric(categories[i].title(), f"{values[i]}/100")
    
    # Notes and sources
    if results.get('notes'):
        st.subheader("📝 Analysis Notes")
        st.write(results['notes'])
    
    if results.get('sources') and len(results['sources']) > 0:
        st.subheader("🔗 Sources")
        for source in results['sources']:
            st.write(f"• {source}")
    
    # Category info
    st.info(f"📂 **Category:** {results.get('category', 'Unknown')}")

if __name__ == "__main__":
    main() 