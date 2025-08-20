"""
Main orchestrator that coordinates providers and merges results.
"""

import statistics
from typing import Dict, Any, List
from utils.entity import normalize_entity
from core.enrich import guess_category
from providers.openai_adapter import analyze_with_openai, get_openai_key
from providers.anthropic_adapter import analyze_with_anthropic, get_anthropic_key
from providers.gemini_adapter import analyze_with_gemini, get_gemini_key
from core.schema import get_fallback_result

def get_available_providers() -> Dict[str, bool]:
    """Check which providers have API keys available"""
    return {
        "openai": bool(get_openai_key()),
        "anthropic": bool(get_anthropic_key()),
        "gemini": bool(get_gemini_key())
    }

def calculate_overall_score(breakdown: Dict[str, int]) -> int:
    """
    Calculate overall score using weighted formula.
    
    Weights:
    - Recognition: 30%
    - Media: 25% 
    - Context: 20%
    - Consistency: 15%
    - Competitors: 10%
    """
    weights = {
        "recognition": 0.30,
        "media": 0.25,
        "context": 0.20,
        "consistency": 0.15,
        "competitors": 0.10
    }
    
    weighted_sum = 0
    for metric, weight in weights.items():
        score = breakdown.get(metric, 40)
        weighted_sum += score * weight
    
    return int(round(weighted_sum))

def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple provider results using median scoring.
    
    Args:
        results: List of provider results matching unified schema
        
    Returns:
        Merged result with median scores
    """
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
    
    for result in results:
        for metric, score in result["breakdown"].items():
            if metric in breakdown_scores:
                breakdown_scores[metric].append(score)
    
    # Calculate median scores
    merged_breakdown = {}
    for metric, scores in breakdown_scores.items():
        merged_breakdown[metric] = int(statistics.median(scores)) if scores else 40
    
    # Calculate weighted overall score
    overall_score = calculate_overall_score(merged_breakdown)
    
    # Merge notes (trim to 600 chars)
    all_notes = []
    for result in results:
        if result.get("notes"):
            all_notes.append(result["notes"])
    
    merged_notes = " | ".join(all_notes)[:600]
    if len(" | ".join(all_notes)) > 600:
        merged_notes = merged_notes.rsplit(" ", 1)[0] + "..."
    
    # Merge and deduplicate sources (cap at 8)
    all_sources = []
    for result in results:
        all_sources.extend(result.get("sources", []))
    
    # Deduplicate while preserving order
    unique_sources = []
    seen = set()
    for source in all_sources:
        if source not in seen and len(unique_sources) < 8:
            unique_sources.append(source)
            seen.add(source)
    
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
    Main orchestration function.
    
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
    
    # Step 2: Guess category  
    category = guess_category(normalized_entity)
    
    # Step 3: Determine available providers
    available_providers = get_available_providers()
    
    if selected_providers:
        # Filter to only selected and available providers
        providers_to_use = [p.lower() for p in selected_providers if available_providers.get(p.lower(), False)]
    else:
        # Use all available providers
        providers_to_use = [p for p, available in available_providers.items() if available]
    
    if not providers_to_use:
        return get_fallback_result(normalized_entity, "No providers available - add API keys to enable real analysis.")
    
    # Step 4: Call each enabled adapter
    results = []
    
    if "openai" in providers_to_use:
        try:
            result = analyze_with_openai(normalized_entity, category)
            if result:
                results.append(result)
        except Exception as e:
            print(f"OpenAI adapter failed: {e}")
    
    if "anthropic" in providers_to_use:
        try:
            result = analyze_with_anthropic(normalized_entity, category)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Anthropic adapter failed: {e}")
    
    if "gemini" in providers_to_use:
        try:
            result = analyze_with_gemini(normalized_entity, category)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Gemini adapter failed: {e}")
    
    # Step 5: Merge results or return fallback
    if not results:
        return get_fallback_result(normalized_entity, "All provider calls failed - check API keys and network connection.")
    
    # Step 6: Return merged result
    return merge_results(results) 