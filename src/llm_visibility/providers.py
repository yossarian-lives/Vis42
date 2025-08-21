import contextlib
import time
import json
from src.llm_visibility.utils.logging import get_logger
from src.llm_visibility.utils.config import SETTINGS

log = get_logger("vis42.providers")

def call_with_fallback(fn, *, provider_name, simulate_fn):
    """Robust provider call with fallback to simulation"""
    t0 = time.time()
    try:
        if SETTINGS.real_enabled:
            res = fn(timeout=SETTINGS.req_timeout, max_tokens=SETTINGS.max_tokens)
            log.info(f"{provider_name} latency={time.time()-t0:.2f}s")
            return {"ok": True, "data": res}
    except Exception as e:
        log.exception(f"{provider_name} failed, using fallback: {e}")
    
    if SETTINGS.simulation_fallback:
        with contextlib.suppress(Exception):
            sim = simulate_fn()
            return {"ok": True, "data": sim, "simulated": True}
    
    return {"ok": False, "error": f"{provider_name} failed and no fallback"}

def call_openai_robust(entity: str, category: str):
    """Robust OpenAI call with proper error handling"""
    def _call():
        import openai
        client = openai.OpenAI(api_key=SETTINGS.openai_key)
        
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with these exact fields:
{{
    "facts": ["fact1", "fact2"],
    "competitors": ["comp1", "comp2"],
    "detail_score": 0.75,
    "context_score": 0.80,
    "recognition_score": 0.85,
    "consistency_score": 0.90
}}

Use realistic scores 0.0-1.0 based on how well-known {entity} is."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an LLM visibility analyst. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=SETTINGS.max_tokens,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    
    def _simulate():
        return {
            "facts": [f"{entity} operates in the {category} space"],
            "competitors": ["Competitor A", "Competitor B"],
            "detail_score": 0.7,
            "context_score": 0.75,
            "recognition_score": 0.8,
            "consistency_score": 0.85
        }
    
    return call_with_fallback(
        _call, 
        provider_name="OpenAI", 
        simulate_fn=_simulate
    )

def call_anthropic_robust(entity: str, category: str):
    """Robust Anthropic call with proper error handling"""
    def _call():
        import anthropic
        client = anthropic.Anthropic(api_key=SETTINGS.anthropic_key)
        
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with these exact fields:
{{
    "facts": ["fact1", "fact2"],
    "competitors": ["comp1", "comp2"],
    "detail_score": 0.75,
    "context_score": 0.80,
    "recognition_score": 0.85,
    "consistency_score": 0.90
}}

Use realistic scores 0.0-1.0 based on how well-known {entity} is."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=SETTINGS.max_tokens,
            temperature=0.3,
            system="You are an LLM visibility analyst. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        return json.loads(result_text)
    
    def _simulate():
        return {
            "facts": [f"{entity} has presence in {category}"],
            "competitors": ["Competitor X", "Competitor Y"],
            "detail_score": 0.65,
            "context_score": 0.7,
            "recognition_score": 0.75,
            "consistency_score": 0.8
        }
    
    return call_with_fallback(
        _call, 
        provider_name="Anthropic", 
        simulate_fn=_simulate
    )

def call_gemini_robust(entity: str, category: str):
    """Robust Gemini call with proper error handling"""
    def _call():
        import google.generativeai as genai
        genai.configure(api_key=SETTINGS.gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with these exact fields:
{{
    "facts": ["fact1", "fact2"],
    "competitors": ["comp1", "comp2"],
    "detail_score": 0.75,
    "context_score": 0.80,
    "recognition_score": 0.85,
    "consistency_score": 0.90
}}

Use realistic scores 0.0-1.0 based on how well-known {entity} is."""

        response = model.generate_content(
            prompt + "\n\nRespond with valid JSON only.",
            generation_config={"temperature": 0.3}
        )
        
        result_text = response.text
        return json.loads(result_text)
    
    def _simulate():
        return {
            "facts": [f"{entity} is active in {category}"],
            "competitors": ["Competitor M", "Competitor N"],
            "detail_score": 0.6,
            "context_score": 0.65,
            "recognition_score": 0.7,
            "consistency_score": 0.75
        }
    
    return call_with_fallback(
        _call, 
        provider_name="Gemini", 
        simulate_fn=_simulate
    ) 