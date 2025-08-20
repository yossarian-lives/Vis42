"""
JSON Schema for LLM Visibility Analysis
Defines the unified schema that all providers must return.
"""

from typing import Dict, Any

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

def validate_result(data: Dict[str, Any]) -> bool:
    """Validate that a result matches our schema"""
    try:
        # Basic validation without external dependencies
        if not isinstance(data, dict):
            return False
        
        # Check required fields
        required_fields = ["entity", "category", "overall_score", "breakdown", "notes", "sources"]
        for field in required_fields:
            if field not in data:
                return False
        
        # Check breakdown structure
        breakdown = data.get("breakdown", {})
        if not isinstance(breakdown, dict):
            return False
        
        breakdown_fields = ["recognition", "media", "context", "competitors", "consistency"]
        for field in breakdown_fields:
            if field not in breakdown:
                return False
            if not isinstance(breakdown[field], (int, float)):
                return False
            if breakdown[field] < 0 or breakdown[field] > 100:
                return False
        
        # Check overall score
        if not isinstance(data["overall_score"], (int, float)):
            return False
        if data["overall_score"] < 0 or data["overall_score"] > 100:
            return False
        
        # Check notes length
        if not isinstance(data["notes"], str):
            return False
        if len(data["notes"]) > 600:
            return False
        
        # Check sources
        if not isinstance(data["sources"], list):
            return False
        if len(data["sources"]) > 8:
            return False
        
        return True
    except Exception:
        return False

def get_fallback_result(entity: str, reason: str = "Structured fallback due to unparseable provider output.") -> Dict[str, Any]:
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