"""
Optional web enrichment to infer entity categories using search APIs.
"""

import os
import re
from typing import Optional
from utils.entity import guess_category_from_name

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
            "activewear", "apparel", "clothing", "athleisure", "retail", "fashion",
            "athletic wear", "sportswear", "fitness", "yoga"
        ],
        "technology": [
            "software", "platform", "tech", "app", "digital", "saas", "technology",
            "startup", "innovation", "developer"
        ],
        "automotive": [
            "automotive", "car", "vehicle", "auto", "motor", "electric vehicle",
            "transportation", "mobility", "tesla", "ford"
        ],
        "artificial intelligence": [
            "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
            "nlp", "chatbot", "openai"
        ],
        "healthcare": [
            "healthcare", "medical", "health", "pharma", "biotech", "medicine",
            "clinical", "patient"
        ],
        "financial services": [
            "bank", "banking", "finance", "financial", "investment", "fintech",
            "trading", "payment"
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

def guess_category(entity: str) -> str:
    """
    Guess entity category using web search if API keys available, otherwise fall back to name-based detection.
    
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
        search_text = search_with_tavily(entity, tavily_key)
        if search_text:
            web_category = detect_category_from_text(search_text)
            # Prefer web-based category if it's more specific than name-based
            if web_category != "brand" or name_based_category == "brand":
                return web_category
    
    elif serper_key:
        search_text = search_with_serper(entity, serper_key)
        if search_text:
            web_category = detect_category_from_text(search_text)
            # Prefer web-based category if it's more specific than name-based
            if web_category != "brand" or name_based_category == "brand":
                return web_category
    
    # Fall back to name-based category
    return name_based_category 