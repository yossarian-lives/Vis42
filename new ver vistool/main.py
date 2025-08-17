"""
LLM Visibility API — FastAPI Backend

Production-ready backend that queries multiple LLM providers (OpenAI, Anthropic, Gemini)
with standardized probes and converts responses into interpretable visibility scores.
"""
from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict, List, Optional

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from providers import OpenAIProvider, AnthropicProvider, GeminiProvider, BaseProvider
from schemas import AnalyzeRequest, AnalyzeResponse, Health, ProviderBreakdown
from scoring import compute_subscores_from_provider, visibility_from_subscores, aggregate_results
from cache import get_cache, set_cache


# Environment configuration
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", str(6 * 3600)))  # 6 hours

app = FastAPI(
    title="LLM Visibility API",
    version="1.0.0",
    description="Analyze entity visibility across major LLMs"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Rate limiting state (simple in-memory)
rate_limit_state: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # seconds


def check_rate_limit(ip: str) -> bool:
    """Simple rate limiting: 30 requests per minute per IP"""
    now = time.time()
    if ip not in rate_limit_state:
        rate_limit_state[ip] = []
    
    # Clean old entries
    rate_limit_state[ip] = [t for t in rate_limit_state[ip] if now - t < RATE_LIMIT_WINDOW]
    
    if len(rate_limit_state[ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    rate_limit_state[ip].append(now)
    return True


def seed_from_string(s: str) -> int:
    """Generate deterministic seed from string"""
    h = 2166136261
    for ch in s:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h


async def get_providers(provider_names: List[str]) -> tuple[List[BaseProvider], List[str]]:
    """Initialize requested providers, return (providers, error_notes)"""
    providers: List[BaseProvider] = []
    notes: List[str] = []
    
    for name in provider_names:
        try:
            if name == "openai":
                providers.append(OpenAIProvider())
            elif name == "anthropic":
                providers.append(AnthropicProvider())
            elif name == "gemini":
                providers.append(GeminiProvider())
            else:
                notes.append(f"Unknown provider: {name}")
        except Exception as e:
            notes.append(f"Provider {name} unavailable: {str(e)}")
    
    return providers, notes


async def analyze_entity(req: AnalyzeRequest) -> AnalyzeResponse:
    """Main analysis orchestration"""
    # Check cache first
    cache_key = f"visibility:{req.entity}:{req.category or ''}"
    cached = await get_cache(cache_key)
    if cached:
        return AnalyzeResponse(**cached)
    
    # Generate seed for deterministic results
    seed = seed_from_string(req.entity + "|" + (req.category or ""))
    
    # Initialize providers
    providers, notes = await get_providers(req.providers)
    
    if not providers:
        raise HTTPException(
            status_code=400, 
            detail="No providers available. Check API keys and configuration."
        )
    
    # Run probes in parallel across all providers
    async def run_provider_analysis(provider: BaseProvider) -> ProviderBreakdown:
        try:
            # Run the standardized probes
            provider_resp = await asyncio.wait_for(
                provider.run_probes(req.entity, req.category, seed=seed),
                timeout=REQUEST_TIMEOUT_SEC - 5
            )
            
            # Compute subscores
            subscores = await compute_subscores_from_provider(provider_resp, req.entity)
            
            # Calculate overall score
            overall = visibility_from_subscores(subscores)
            
            return ProviderBreakdown(
                provider=provider.name,
                model=provider_resp.model_name,
                subscores=subscores,
                overall=overall,
                probes={
                    "profile": provider_resp.profile,
                    "context": provider_resp.context,
                    "alt": provider_resp.alt,
                    "consistency": provider_resp.consistency,
                },
                raw=provider_resp.raw
            )
            
        except asyncio.TimeoutError:
            notes.append(f"Provider {provider.name} timed out")
            raise
        except Exception as e:
            notes.append(f"Provider {provider.name} failed: {str(e)}")
            raise
    
    # Execute all provider analyses
    try:
        results = await asyncio.gather(
            *[run_provider_analysis(p) for p in providers],
            return_exceptions=True
        )
        
        # Filter successful results
        successful_results = [r for r in results if isinstance(r, ProviderBreakdown)]
        
        if not successful_results:
            raise HTTPException(
                status_code=503,
                detail="All providers failed. " + "; ".join(notes)
            )
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Analysis failed: {str(e)}")
    
    # Aggregate results across providers
    overall_score, aggregated_subscores = aggregate_results(successful_results)
    
    # Check for high variance across providers
    if len(successful_results) >= 2:
        scores = [r.overall for r in successful_results]
        score_range = max(scores) - min(scores)
        if score_range >= 25:
            notes.append("High cross-provider variance — investigate prompt framing or category.")
    
    # Build response
    response = AnalyzeResponse(
        entity=req.entity,
        category=req.category,
        competitors=req.competitors,
        overall=overall_score,
        subscores=aggregated_subscores,
        providers=successful_results,
        notes=notes
    )
    
    # Cache the result
    await set_cache(cache_key, response.dict(), ttl=CACHE_TTL_SEC)
    
    return response


@app.get("/health", response_model=Health)
async def health_check():
    """Health check endpoint"""
    return Health(ok=True, time=time.time())


@app.post("/api/visibility", response_model=AnalyzeResponse)
async def visibility_analysis(request: Request, req: AnalyzeRequest):
    """Main visibility analysis endpoint"""
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Try again later."
        )
    
    try:
        return await analyze_entity(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# Custom JSON response to use orjson for better performance
@app.middleware("http")
async def json_middleware(request: Request, call_next):
    response = await call_next(request)
    return response


# Startup event to validate configuration
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup"""
    required_vars = []
    
    if not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        required_vars.append("ANTHROPIC_API_KEY")  
    if not os.getenv("GEMINI_API_KEY"):
        required_vars.append("GEMINI_API_KEY")
    
    if required_vars:
        print(f"Warning: Missing environment variables: {', '.join(required_vars)}")
        print("Some providers will be unavailable.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5051"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )