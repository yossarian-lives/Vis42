"""
Entity normalization utilities for handling common misspellings and formatting.
"""

import re
from typing import Dict

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