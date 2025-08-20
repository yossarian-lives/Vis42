"""
Scoring Utilities

Functions for calculating visibility scores and handling score-related operations.
"""

from typing import Dict

def seed_from_string(s: str) -> int:
    """Generate deterministic seed from string"""
    h = 2166136261
    for ch in s:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h

def clamp01(x: float) -> float:
    """Clamp value between 0 and 1"""
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def visibility_from_subscores(subscores: Dict[str, float]) -> float:
    """Calculate overall visibility score from subscores"""
    # Weights (tune as needed)
    W_RECOG = 0.45
    W_DETAIL = 0.25
    W_CONTEXT = 0.20
    W_COMP = 0.10
    
    base = (
        W_RECOG * subscores.get('recognition', 0.0)
        + W_DETAIL * subscores.get('detail', 0.0)
        + W_CONTEXT * subscores.get('context', 0.0)
        + W_COMP * subscores.get('competitors', 0.0)
    )  # 0..1
    
    # mild consistency multiplier: 0.85..1.0
    mult = 0.85 + 0.15 * clamp01(subscores.get('consistency', 0.0))
    return round(100.0 * clamp01(base * mult), 1) 