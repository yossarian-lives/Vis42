"""
Pydantic schemas for the LLM Visibility API
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class Health(BaseModel):
    """Health check response"""
    ok: bool
    time: float


class Subscores(BaseModel):
    """Visibility subscores (all 0.0 to 1.0)"""
    recognition: float = Field(..., ge=0.0, le=1.0, description="Entity recognition level")
    detail: float = Field(..., ge=0.0, le=1.0, description="Factual detail richness")
    context: float = Field(..., ge=0.0, le=1.0, description="Contextual ranking position")
    competitors: float = Field(..., ge=0.0, le=1.0, description="Competitive awareness")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Cross-probe consistency")


class ProviderBreakdown(BaseModel):
    """Results from a single LLM provider"""
    provider: str = Field(..., description="Provider name (openai, anthropic, gemini)")
    model: Optional[str] = Field(None, description="Model identifier used")
    subscores: Subscores = Field(..., description="Provider-specific subscores")
    overall: float = Field(..., ge=0.0, le=100.0, description="Provider overall score (0-100)")
    probes: Dict[str, Any] = Field(..., description="Parsed JSON responses from probes")
    raw: Dict[str, Any] = Field(..., description="Raw text responses for auditability")


class ProviderResponse(BaseModel):
    """Internal provider response structure"""
    profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    alt: Dict[str, Any] = Field(default_factory=dict)
    consistency: Dict[str, Any] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Request to analyze entity visibility"""
    entity: str = Field(..., min_length=1, max_length=200, description="Entity to analyze")
    category: Optional[str] = Field(None, max_length=100, description="Optional category hint")
    competitors: List[str] = Field(
        default_factory=list, 
        max_items=10,
        description="Optional competitor list for comparison"
    )
    providers: List[str] = Field(
        default_factory=lambda: ["openai", "anthropic", "gemini"],
        description="LLM providers to query"
    )

    @validator("entity")
    def validate_entity(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Entity name is required")
        if len(v) > 200:
            raise ValueError("Entity name too long (max 200 characters)")
        return v

    @validator("category")
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if len(v) > 100:
                raise ValueError("Category too long (max 100 characters)")
        return v

    @validator("competitors")
    def validate_competitors(cls, v: List[str]) -> List[str]:
        if not v:
            return []
        
        cleaned = []
        for comp in v:
            comp = comp.strip()
            if comp and len(comp) <= 100:
                cleaned.append(comp)
        
        return cleaned[:10]  # Limit to 10 competitors

    @validator("providers")
    def validate_providers(cls, v: List[str]) -> List[str]:
        if not v:
            return ["openai", "anthropic", "gemini"]
        
        valid_providers = {"openai", "anthropic", "gemini"}
        return [p for p in v if p in valid_providers]


class AnalyzeResponse(BaseModel):
    """Response from visibility analysis"""
    entity: str = Field(..., description="Analyzed entity")
    category: Optional[str] = Field(None, description="Entity category")
    competitors: List[str] = Field(default_factory=list, description="Competitor list")
    overall: float = Field(..., ge=0.0, le=100.0, description="Overall visibility score (0-100)")
    subscores: Subscores = Field(..., description="Aggregated subscores")
    providers: List[ProviderBreakdown] = Field(..., description="Per-provider results")
    notes: List[str] = Field(default_factory=list, description="Analysis notes and warnings")


# Probe response schemas for validation
class ProfileProbeResponse(BaseModel):
    """Expected structure for profile probe response"""
    recognized: bool = False
    summary: str = ""
    facts: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    competitors: List[str] = Field(default_factory=list)


class ContextProbeResponse(BaseModel):
    """Expected structure for context ranking probe response"""
    top_list: List[str] = Field(default_factory=list)
    rank_of_entity: Optional[int] = None

    @validator("rank_of_entity")
    def validate_rank(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 10):
            return None
        return v


class AlternativesProbeResponse(BaseModel):
    """Expected structure for alternatives probe response"""
    alternatives: List[str] = Field(default_factory=list)


class ConsistencyProbeResponse(BaseModel):
    """Expected structure for consistency probe response (same as context)"""
    top_list: List[str] = Field(default_factory=list)
    rank_of_entity: Optional[int] = None

    @validator("rank_of_entity")
    def validate_rank(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 10):
            return None
        return v