"""
Scoring logic for LLM visibility analysis
"""
from __future__ import annotations

from typing import List, Tuple
from schemas import ProviderResponse, Subscores, ProviderBreakdown


# Scoring weights (configurable via environment if needed)
W_RECOGNITION = 0.45
W_DETAIL = 0.25
W_CONTEXT = 0.20
W_COMPETITORS = 0.10

# Helper functions
def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


async def compute_subscores_from_provider(resp: ProviderResponse, entity: str) -> Subscores:
    """
    Compute subscores from a single provider's response.
    All subscores are 0.0 to 1.0.
    """
    profile = resp.profile or {}
    context = resp.context or {}
    alt = resp.alt or {}
    consistency = resp.consistency or {}

    # Recognition subscore (0.0 to 1.0)
    # Based on: recognized flag + summary richness + facts count
    recognized = 1.0 if profile.get("recognized") is True else 0.0
    summary = profile.get("summary", "") or ""
    summary_words = len(summary.split()) if summary else 0
    facts = profile.get("facts", []) or []
    facts_count = len(facts)
    
    recognition = (
        0.40 * recognized +
        0.35 * clamp(summary_words / 120.0) +  # Normalize by expected ~120 words for good summary
        0.25 * clamp(facts_count / 10.0)       # Normalize by expected ~10 facts
    )

    # Detail subscore (0.0 to 1.0)
    # Based on: factual richness + competitor awareness
    competitors = profile.get("competitors", []) or []
    competitors_count = len(competitors)
    
    detail = (
        0.60 * clamp(facts_count / 10.0) +
        0.40 * clamp(competitors_count / 10.0)
    )

    # Context subscore (0.0 to 1.0)
    # Based on: ranking position in top-10 list (1=best, 10=worst)
    rank = context.get("rank_of_entity")
    if isinstance(rank, int) and 1 <= rank <= 10:
        context_score = (11 - rank) / 10.0  # 1.0 for rank 1, 0.1 for rank 10
    else:
        context_score = 0.0

    # Competitors/Alternatives subscore (0.0 to 1.0)
    # Based on: competitors list + alternatives list (proxy for contextual awareness)
    alternatives = alt.get("alternatives", []) or []
    alternatives_count = len(alternatives)
    
    competitors_score = (
        0.50 * clamp(competitors_count / 10.0) +
        0.50 * clamp(alternatives_count / 10.0)
    )

    # Consistency subscore (0.0 to 1.0)
    # Based on: variance between context and consistency probe rankings
    consistency_rank = consistency.get("rank_of_entity")
    
    if isinstance(rank, int) and isinstance(consistency_rank, int):
        # Lower variance = higher consistency score
        rank_diff = abs(rank - consistency_rank)
        consistency_score = clamp(1.0 - (rank_diff / 10.0))  # Perfect consistency = 1.0, max diff = 0.0
    elif context_score > 0:
        # If we have context but not consistency, give neutral score
        consistency_score = 0.7
    else:
        # No ranking data available
        consistency_score = 0.5

    return Subscores(
        recognition=round(clamp(recognition), 4),
        detail=round(clamp(detail), 4),
        context=round(clamp(context_score), 4),
        competitors=round(clamp(competitors_score), 4),
        consistency=round(clamp(consistency_score), 4)
    )


def visibility_from_subscores(subscores: Subscores) -> float:
    """
    Convert subscores to overall visibility score (0-100).
    Applies consistency as a mild multiplier.
    """
    # Base score from weighted subscores
    base_score = (
        W_RECOGNITION * subscores.recognition +
        W_DETAIL * subscores.detail +
        W_CONTEXT * subscores.context +
        W_COMPETITORS * subscores.competitors
    )
    
    # Consistency acts as mild multiplier (0.85 to 1.0)
    consistency_multiplier = 0.85 + 0.15 * subscores.consistency
    
    # Final score: 0-100
    final_score = base_score * consistency_multiplier * 100.0
    
    return round(clamp(final_score, 0.0, 100.0), 1)


def aggregate_results(provider_results: List[ProviderBreakdown]) -> Tuple[float, Subscores]:
    """
    Aggregate results across multiple providers.
    Returns (overall_score, aggregated_subscores).
    """
    if not provider_results:
        return 0.0, Subscores(
            recognition=0.0, detail=0.0, context=0.0, 
            competitors=0.0, consistency=0.0
        )
    
    # Weight providers by presence of context ranking (more reliable providers get higher weight)
    weights = []
    for result in provider_results:
        # Base weight of 0.5, +0.5 if provider has valid context ranking
        has_context_rank = bool(
            result.probes.get("context", {}).get("rank_of_entity") is not None
        )
        weight = 0.5 + (0.5 if has_context_rank else 0.0)
        weights.append(weight)
    
    total_weight = sum(weights) or 1.0
    
    # Weighted average of overall scores
    overall_score = sum(
        weight * result.overall 
        for weight, result in zip(weights, provider_results)
    ) / total_weight
    
    # Weighted average of subscores
    def weighted_average_subscore(field: str) -> float:
        values = [getattr(result.subscores, field) for result in provider_results]
        return sum(weight * value for weight, value in zip(weights, values)) / total_weight
    
    aggregated_subscores = Subscores(
        recognition=round(weighted_average_subscore("recognition"), 4),
        detail=round(weighted_average_subscore("detail"), 4),
        context=round(weighted_average_subscore("context"), 4),
        competitors=round(weighted_average_subscore("competitors"), 4),
        consistency=round(weighted_average_subscore("consistency"), 4)
    )
    
    return round(overall_score, 1), aggregated_subscores