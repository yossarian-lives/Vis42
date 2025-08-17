"""
LLM Visibility API — FastAPI (v1)

Goal
----
Provide a backend that queries multiple LLM providers (OpenAI, Anthropic, Gemini)
with standardized, JSON-structured probes and converts their answers into a
stable, interpretable 0–100 "visibility" score + subscores.

Highlights
---------
- Deterministic-ish: temperature=0, JSON-mode/structured outputs, optional seeds.
- Provider-agnostic adapters (OpenAI / Anthropic / Gemini) with graceful fallbacks.
- Parallel execution (asyncio) for speed.
- Robust JSON parsing with auto-repair.
- Transparent scoring: recognition, detail, context-rank, competitor-recall, and
  a mild consistency factor.
- CORS enabled for local/SPA usage.

Setup
-----
1) Python 3.10+
2) `pip install fastapi uvicorn pydantic openai anthropic google-genai httpx orjson python-dotenv`
3) Environment variables:
   - OPENAI_API_KEY, OPENAI_MODEL (e.g., "gpt-4o-mini" or a current snapshot)
   - ANTHROPIC_API_KEY, ANTHROPIC_MODEL (e.g., "claude-sonnet-4-20250514")
   - GEMINI_API_KEY, GEMINI_MODEL (e.g., "gemini-2.5-flash" or latest)
4) Run: `uvicorn main:app --reload --port 5051`

Docs
----
- OpenAI Responses API + Structured Outputs: https://platform.openai.com/docs/api-reference/responses
  and https://platform.openai.com/docs/guides/structured-outputs
- Anthropic Messages API (Claude 4 family): https://docs.anthropic.com/en/api/messages
- Gemini API (Google GenAI SDK) + JSON Mode: https://ai.google.dev/api/generate-content (JSON Mode)

Copyright
---------
This file is provided as a starting point. Review provider ToS and safety policies
before production use.
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- Utilities ----------

def oj_dumps(v: Any, *, default):  # FastAPI jsonable encoder hook
    return orjson.dumps(v, default=default)

class ORJSONResponseMixin:
    @staticmethod
    def orjson(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")

JSON_REPAIR_RE = re.compile(r"\{[\s\S]*\}")

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse strict JSON. If it fails, attempt to repair by extracting the
    first balanced-looking {...} block.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_REPAIR_RE.search(text)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

# deterministic-ish seeds from string
def seed_from_string(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h

# ---------- Scoring ----------

class Subscores(BaseModel):
    recognition: float
    detail: float
    context: float
    competitors: float
    consistency: float

class ProviderBreakdown(BaseModel):
    provider: str
    model: Optional[str] = None
    subscores: Subscores
    overall: float
    probes: Dict[str, Any]
    raw: Dict[str, Any]

class VisibilityResult(BaseModel):
    entity: str
    category: Optional[str] = None
    competitors: List[str] = Field(default_factory=list)
    overall: float
    subscores: Subscores
    providers: List[ProviderBreakdown]
    notes: List[str] = Field(default_factory=list)

# Weights (tune as needed)
W_RECOG = 0.45
W_DETAIL = 0.25
W_CONTEXT = 0.20
W_COMP = 0.10
# consistency is applied as a mild multiplier later

# helper to clamp 0..1
clamp01 = lambda x: 0.0 if x < 0 else (1.0 if x > 1 else x)


def visibility_from_subscores(s: Subscores) -> float:
    base = (
        W_RECOG * s.recognition
        + W_DETAIL * s.detail
        + W_CONTEXT * s.context
        + W_COMP * s.competitors
    )  # 0..1
    # mild consistency multiplier: 0.85..1.0
    mult = 0.85 + 0.15 * clamp01(s.consistency)
    return round(100.0 * clamp01(base * mult), 1)


# ---------- Probe Design ----------
"""
We send the following standardized probes to each provider in JSON mode (or as
strict-JSON instructions):

1) PROFILE probe — Does the model "know" the entity?
   Prompt target JSON schema (simplified):
   {
     "recognized": bool,
     "summary": str,               # 1–3 sentences
     "facts": [str],               # 5–12 concise facts
     "category": str | null,       # coarse category/industry
     "competitors": [str]          # inferred competitors
   }

2) CONTEXT-RANK probe — Rank list placement among peers for the inferred
   category (or user-provided category). We ask the model to output a TOP-N list
   and include the rank (1-indexed) if the entity appears; else null.
   Schema:
   {
     "top_list": [str],            # length 10 ideally
     "rank_of_entity": int | null
   }

3) ALT probe — Alternatives to entity.
   Schema:
   {
     "alternatives": [str]         # 10 items
   }

4) CONSISTENCY probe — A simple re-ask/paraphrase of probe #2 to estimate
   stability of rank_of_entity. We compute variance.
"""

PROFILE_SCHEMA_DESC = (
    "Return ONLY JSON with keys: recognized (bool), summary (string, 1-3 "
    "sentences), facts (array of concise strings, 5-12 items), category (string "
    "or null), competitors (array of strings, 3-12 items)."
)

CONTEXT_SCHEMA_DESC = (
    "Return ONLY JSON with keys: top_list (array of 10 strings), "
    "rank_of_entity (integer 1-10 or null if the entity is absent)."
)

ALT_SCHEMA_DESC = (
    "Return ONLY JSON with key: alternatives (array of 10 strings)."
)

CONSISTENCY_SCHEMA_DESC = CONTEXT_SCHEMA_DESC

SYSTEM_INSTRUCTIONS = (
    "You are a careful analyst. Respond strictly in VALID JSON with no extra "
    "text. If uncertain, set values to null or empty arrays instead of guessing."
)

# ---------- Provider Adapters ----------

class ProviderResponse(BaseModel):
    profile: Dict[str, Any]
    context: Dict[str, Any]
    alt: Dict[str, Any]
    consistency: Dict[str, Any]
    raw: Dict[str, Any]
    model_name: Optional[str] = None


class BaseProvider:
    name: str = "base"

    def __init__(self, model_env: str, default_model: str):
        self.model = os.getenv(model_env, default_model)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    async def run_probes(self, entity: str, category_hint: Optional[str], *, seed: int) -> ProviderResponse:
        profile_prompt = (
            f"{PROFILE_SCHEMA_DESC}\nEntity: {entity}. If known, include competitors and a coarse category."
        )
        if category_hint:
            profile_prompt += f" Category hint: {category_hint}."

        ctx_prompt = (
            f"{CONTEXT_SCHEMA_DESC}\nTask: Provide a top-10 list of leading players for the category relevant to \n"
            f"'{entity}'. Use a widely-accepted industry framing."
        )
        if category_hint:
            ctx_prompt += f" Category hint: {category_hint}."

        alt_prompt = (
            f"{ALT_SCHEMA_DESC}\nTask: List 10 notable alternatives or substitutes for '{entity}'."
        )

        con_prompt = (
            f"{CONSISTENCY_SCHEMA_DESC}\nTask: Re-run the ranking probe from a different perspective."
        )

        # Run in parallel
        results = await asyncio.gather(
            self.ask_json(profile_prompt, "profile", seed=seed + 1),
            self.ask_json(ctx_prompt, "context", seed=seed + 2),
            self.ask_json(alt_prompt, "alt", seed=seed + 3),
            self.ask_json(con_prompt, "consistency", seed=seed + 4),
            return_exceptions=True,
        )

        raw_map: Dict[str, Any] = {}
        out: Dict[str, Dict[str, Any]] = {}
        labels = ["profile", "context", "alt", "consistency"]
        for lbl, res in zip(labels, results):
            if isinstance(res, Exception):
                raw_map[lbl] = {"error": str(res)}
                out[lbl] = {}
            else:
                raw_text, parsed = res
                raw_map[lbl] = {"text": raw_text}
                out[lbl] = parsed or {}

        return ProviderResponse(
            profile=out.get("profile", {}),
            context=out.get("context", {}),
            alt=out.get("alt", {}),
            consistency=out.get("consistency", {}),
            raw=raw_map,
            model_name=self.model,
        )


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        super().__init__("OPENAI_MODEL", default_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        # defer import to allow running without the lib
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int):
        # Use Chat Completions API with JSON output
        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                # seed is optional/if supported
            )
            # unify text extraction
            try:
                text = resp.choices[0].message.content or ""
            except Exception:
                # fallback: stitch from content if needed
                text = getattr(resp, "content", "") or str(resp)
            return text

        text = await asyncio.to_thread(_call)
        parsed = try_parse_json(text)
        return text, parsed


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self):
        super().__init__("ANTHROPIC_MODEL", default_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
        import anthropic  # type: ignore
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int):
        # Messages API — return JSON by instruction. See docs: docs.anthropic.com/en/api/messages
        def _call():
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0,
                system=SYSTEM_INSTRUCTIONS,
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic returns content as a list of blocks
            try:
                parts = getattr(msg, "content", [])
                text = "".join(getattr(p, "text", "") for p in parts)
            except Exception:
                text = str(msg)
            return text

        text = await asyncio.to_thread(_call)
        parsed = try_parse_json(text)
        return text, parsed


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self):
        super().__init__("GEMINI_MODEL", default_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        from google import genai  # type: ignore
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int):
        # JSON Mode via response_mime_type. See: ai.google.dev/api/generate-content (JSON Mode)
        from google.genai import types  # type: ignore
        def _call():
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
            )
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=cfg,
            )
            try:
                # SDK returns an object with .text or raw candidates
                text = getattr(resp, "text", None)
                if text is None:
                    text = json.dumps(resp.to_dict())
            except Exception:
                text = str(resp)
            return text

        text = await asyncio.to_thread(_call)
        parsed = try_parse_json(text)
        return text, parsed


# ---------- Core Aggregation ----------

class AnalyzeRequest(BaseModel):
    entity: str
    category: Optional[str] = None
    competitors: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=lambda: ["openai", "anthropic", "gemini"])

    @validator("entity")
    def _v_entity(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("entity required")
        if len(v) > 80:
            raise ValueError("entity too long")
        return v


class AnalyzeResponse(VisibilityResult):
    pass


async def compute_subscores_from_provider(resp: ProviderResponse, entity: str) -> Subscores:
    profile = resp.profile or {}
    context = resp.context or {}
    alt = resp.alt or {}
    consistency = resp.consistency or {}

    # recognition: based on recognized + summary length + facts count
    recognized = 1.0 if profile.get("recognized") is True else 0.0
    summary_len = len((profile.get("summary") or "").split())
    facts_len = len(profile.get("facts") or [])
    recog = clamp01(0.4 * recognized + 0.35 * min(1.0, summary_len / 120) + 0.25 * min(1.0, facts_len / 10))

    # detail: richer factual density; weigh facts + competitor list length
    comp_len = len(profile.get("competitors") or [])
    detail = clamp01(0.6 * min(1.0, facts_len / 10) + 0.4 * min(1.0, comp_len / 10))

    # context: presence + rank in a top list (1 is best)
    rank = context.get("rank_of_entity")
    if isinstance(rank, int) and 1 <= rank <= 10:
        context_score = (11 - rank) / 10.0  # 1.0 for rank 1, 0.1 for rank 10
    else:
        context_score = 0.0

    # competitors: size of alternatives list (proxy for contextual awareness)
    alt_len = len(alt.get("alternatives") or [])
    comp_score = clamp01(0.5 * min(1.0, comp_len / 10) + 0.5 * min(1.0, alt_len / 10))

    # consistency: compare rank between context and consistency probes
    rank2 = consistency.get("rank_of_entity")
    if isinstance(rank, int) and isinstance(rank2, int):
        # lower variance => higher score
        diff = abs(rank - rank2)
        cons = clamp01(1.0 - (diff / 10.0))  # diff 0 => 1.0, diff 10 => 0.0
    else:
        cons = 0.7 if context_score > 0 else 0.5  # neutral fallback

    return Subscores(
        recognition=round(recog, 4),
        detail=round(detail, 4),
        context=round(context_score, 4),
        competitors=round(comp_score, 4),
        consistency=round(cons, 4),
    )


async def analyze_entity(req: AnalyzeRequest) -> AnalyzeResponse:
    seed = seed_from_string(req.entity + "|" + (req.category or ""))

    adapters: List[BaseProvider] = []
    notes: List[str] = []
    for name in req.providers:
        try:
            if name == "openai":
                adapters.append(OpenAIProvider())
            elif name == "anthropic":
                adapters.append(AnthropicProvider())
            elif name == "gemini":
                adapters.append(GeminiProvider())
            else:
                notes.append(f"Unknown provider skipped: {name}")
        except Exception as e:
            notes.append(f"Provider {name} unavailable: {e}")

    if not adapters:
        raise HTTPException(status_code=400, detail="No providers available")

    async def run_adapter(adp: BaseProvider):
        pr = await adp.run_probes(req.entity, req.category, seed=seed)
        subs = await compute_subscores_from_provider(pr, req.entity)
        overall = visibility_from_subscores(subs)
        return ProviderBreakdown(
            provider=adp.name,
            model=adp.model,
            subscores=subs,
            overall=overall,
            probes={
                "profile": pr.profile,
                "context": pr.context,
                "alt": pr.alt,
                "consistency": pr.consistency,
            },
            raw=pr.raw,
        )

    results: List[ProviderBreakdown] = await asyncio.gather(*[run_adapter(a) for a in adapters])

    # Aggregate across providers by weighted average; weight by presence of context rank info
    weights: List[float] = []
    for r in results:
        w = 0.5 + 0.5 * (1.0 if r.probes.get("context", {}).get("rank_of_entity") else 0.0)
        weights.append(w)

    total_w = sum(weights) or 1.0
    overall = round(sum(w * r.overall for w, r in zip(weights, results)) / total_w, 1)

    # merge subscores by weighted average (normalized 0..1 internally)
    def merge_field(field: str) -> float:
        vals = [getattr(r.subscores, field) for r in results]
        # subscores are 0..1; convert to 0..100 and average with weights, then back to 0..1 for response
        agg = sum(w * v for w, v in zip(weights, vals)) / total_w
        return round(agg, 4)

    subs = Subscores(
        recognition=merge_field("recognition"),
        detail=merge_field("detail"),
        context=merge_field("context"),
        competitors=merge_field("competitors"),
        consistency=merge_field("consistency"),
    )

    # Optional: flag if highly divergent provider results (for insight, not correction)
    if len(results) >= 2:
        vals = [r.overall for r in results]
        rng = max(vals) - min(vals)
        if rng >= 25:
            notes.append("High cross-provider variance — investigate prompt framing or category.")

    return AnalyzeResponse(
        entity=req.entity,
        category=req.category,
        competitors=req.competitors,
        overall=overall,
        subscores=subs,
        providers=results,
        notes=notes,
    )


# ---------- FastAPI App ----------
app = FastAPI(title="LLM Visibility API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Health(BaseModel):
    ok: bool
    time: float


@app.get("/health", response_model=Health)
async def health():
    return Health(ok=True, time=time.time())


@app.post("/api/visibility", response_model=AnalyzeResponse)
async def visibility(req: AnalyzeRequest):
    try:
        return await analyze_entity(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Local dev ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5051")))