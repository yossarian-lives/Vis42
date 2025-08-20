"""
Analysis Engine

Core analysis functions for LLM visibility assessment with simulation and real API support.
"""

import time
import random
from typing import Dict, Any, List, Optional

from .providers import SIMULATION_MODE, analyze_with_real_apis
from .json_utils import try_parse_json
from .scoring import visibility_from_subscores, seed_from_string, clamp01

def simulate_llm_analysis(entity: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Simulate LLM analysis with realistic scoring (fallback mode)"""
    # Generate deterministic but realistic scores based on entity
    seed = seed_from_string(entity + "|" + (category or ""))
    
    # Simulate different entity types and their typical scores
    entity_lower = entity.lower()
    
    # High visibility entities (well-known brands/people)
    if any(name in entity_lower for name in ['tesla', 'apple', 'google', 'microsoft', 'amazon', 'netflix']):
        base_recognition = 0.9
        base_detail = 0.85
        base_context = 0.8
        base_competitors = 0.9
        base_consistency = 0.85
    # Medium visibility entities
    elif any(name in entity_lower for name in ['startup', 'ai', 'machine learning', 'blockchain', 'crypto']):
        base_recognition = 0.7
        base_detail = 0.75
        base_context = 0.7
        base_competitors = 0.8
        base_consistency = 0.75
    # Lower visibility entities
    else:
        base_recognition = 0.5
        base_detail = 0.6
        base_context = 0.5
        base_competitors = 0.6
        base_consistency = 0.7
    
    # Add some variation based on seed
    random.seed(seed)
    
    subscores = {
        'recognition': clamp01(base_recognition + random.uniform(-0.1, 0.1)),
        'detail': clamp01(base_detail + random.uniform(-0.1, 0.1)),
        'context': clamp01(base_context + random.uniform(-0.1, 0.1)),
        'competitors': clamp01(base_competitors + random.uniform(-0.1, 0.1)),
        'consistency': clamp01(base_consistency + random.uniform(-0.05, 0.05))
    }
    
    overall = visibility_from_subscores(subscores)
    
    # Generate realistic analysis data
    if 'tesla' in entity_lower:
        summary = "Tesla is an American electric vehicle and clean energy company founded by Elon Musk. Known for innovative electric cars, battery energy storage, and solar products."
        facts = [
            "Founded in 2003 by Martin Eberhard and Marc Tarpenning",
            "Elon Musk joined as chairman in 2004 and became CEO in 2008",
            "First electric car was the Tesla Roadster (2008)",
            "Model S launched in 2012, Model 3 in 2017",
            "Pioneered over-the-air software updates for vehicles",
            "Built Gigafactories for battery production",
            "Market cap often exceeds traditional automakers",
            "Developed Autopilot advanced driver-assistance system"
        ]
        competitors = ["Ford", "General Motors", "Nissan", "Rivian", "Lucid Motors", "Volkswagen", "BMW", "Hyundai"]
        industry_rank = 10
    elif 'apple' in entity_lower:
        summary = "Apple Inc. is an American multinational technology company that designs, develops, and sells consumer electronics, computer software, and online services."
        facts = [
            "Founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne",
            "Headquartered in Cupertino, California",
            "Revolutionized personal computing with Macintosh (1984)",
            "Introduced iPhone in 2007, changing mobile industry",
            "iPad launched in 2010, creating tablet market",
            "Apple Watch debuted in 2015",
            "One of world's most valuable companies",
            "Known for premium design and user experience"
        ]
        competitors = ["Samsung", "Microsoft", "Google", "Amazon", "Sony", "Dell", "HP", "Lenovo"]
        industry_rank = 1
    else:
        # Generic analysis for other entities
        summary = f"{entity} is a notable entity in the {category or 'general'} space with varying levels of recognition across different knowledge bases."
        facts = [
            f"{entity} has established presence in the market",
            "Multiple sources provide information about this entity",
            "Industry recognition varies by region and sector",
            "Competitive landscape includes several players",
            "Technology and innovation play key roles"
        ]
        competitors = ["Competitor A", "Competitor B", "Competitor C", "Alternative 1", "Alternative 2"]
        industry_rank = random.randint(5, 15)
    
    return {
        "entity": entity,
        "category": category or "auto-detected",
        "overall": overall,
        "subscores": subscores,
        "summary": summary,
        "facts": facts,
        "competitors": competitors,
        "industry_rank": industry_rank,
        "providers": [
            {
                "provider": "simulation",
                "model": "simulated-analysis",
                "overall": overall,
                "subscores": subscores
            }
        ],
        "notes": [
            f"Analysis based on simulated LLM responses for {entity}",
            "Scores reflect typical recognition patterns across major AI models",
            "Add API keys to .streamlit/secrets.toml for real analysis"
        ]
    }

def analyze_visibility(entity: str, category: Optional[str], selected_providers: List[str]) -> Dict[str, Any]:
    """Main analysis function - automatically chooses real APIs or simulation"""
    
    if not SIMULATION_MODE and selected_providers:
        # Use real APIs
        prompt = f"""
        Analyze the visibility of "{entity}" in the {category or 'general'} space.
        
        Provide a JSON response with this structure:
        {{
            "summary": "Brief description of the entity",
            "facts": ["fact1", "fact2", "fact3"],
            "competitors": ["competitor1", "competitor2"],
            "industry_rank": 5,
            "subscores": {{
                "recognition": 0.85,
                "detail": 0.75,
                "context": 0.8,
                "competitors": 0.9,
                "consistency": 0.85
            }}
        }}
        
        Base scores on how well-known and detailed the information is about this entity.
        """
        
        # Get real API results
        api_results = analyze_with_real_apis(prompt)
        
        if api_results:
            # Process first successful result
            provider_name = list(api_results.keys())[0]
            result_text = api_results[provider_name]
            result = try_parse_json(result_text)
            
            if result:
                return {
                    "entity": entity,
                    "category": category or "auto-detected",
                    "overall": visibility_from_subscores(result.get("subscores", {})),
                    "subscores": result.get("subscores", {}),
                    "summary": result.get("summary", ""),
                    "facts": result.get("facts", []),
                    "competitors": result.get("competitors", []),
                    "industry_rank": result.get("industry_rank", 10),
                    "providers": [
                        {
                            "provider": provider_name,
                            "model": provider_name,
                            "overall": visibility_from_subscores(result.get("subscores", {})),
                            "subscores": result.get("subscores", {})
                        }
                    ],
                    "notes": [
                        f"Analysis using real LLM API: {provider_name}",
                        "Scores based on actual AI model response"
                    ]
                }
    
    # Fallback to simulation
    time.sleep(1)
    return simulate_llm_analysis(entity, category) 