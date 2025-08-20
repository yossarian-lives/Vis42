"""
LLM Provider Management

Clean, safe wrapper functions for multiple LLM providers with automatic discovery.
"""

import streamlit as st
from contextlib import suppress
from typing import Optional

# ---- Provider Discovery ------------------------------------------------------
def get_secret(name: str) -> str | None:
    """Safely get a secret value, returning None if not found or empty"""
    with suppress(Exception):
        v = st.secrets.get(name)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

PROVIDERS = {
    "openai": {"key_name": "OPENAI_API_KEY"},
    "anthropic": {"key_name": "ANTHROPIC_API_KEY"},
    "gemini": {"key_name": "GEMINI_API_KEY"},
}

ENABLED = {p: get_secret(cfg["key_name"]) for p, cfg in PROVIDERS.items()}
ENABLED = {p: k for p, k in ENABLED.items() if k}  # keep only those with keys

# Determine mode
SIMULATION_MODE = not bool(ENABLED)

# ---- Safe Wrapper Functions (only call if enabled; never raise to UI) -------
def call_openai(prompt: str) -> str | None:
    """Safely call OpenAI API"""
    if "openai" not in ENABLED:
        return None
    try:
        # import here so missing packages don't error when provider disabled
        from openai import OpenAI
        import httpx
        client = OpenAI(
            api_key=ENABLED["openai"],
            http_client=httpx.Client(timeout=20)
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI visibility analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000
        )
        return resp.choices[0].message.content
    except Exception:
        # Silently skip on failure; you can log if you want
        return None

def call_anthropic(prompt: str) -> str | None:
    """Safely call Anthropic API"""
    if "anthropic" not in ENABLED:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ENABLED["anthropic"])
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.1,
            system="You are an AI visibility analyst. Respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        # join text parts
        return "".join(getattr(b, "text", "") for b in msg.content)
    except Exception:
        return None

def call_gemini(prompt: str) -> str | None:
    """Safely call Gemini API"""
    if "gemini" not in ENABLED:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=ENABLED["gemini"])
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            prompt + "\n\nRespond with valid JSON only.",
            generation_config={"temperature": 0.1}
        )
        return response.text
    except Exception:
        return None

# ---- Clean API Aggregation --------------------------------------------------
def analyze_with_real_apis(prompt: str) -> dict[str, str]:
    """Analyze using real LLM APIs - clean aggregation"""
    results = {}
    o = call_openai(prompt)
    if o: results["openai"] = o
    a = call_anthropic(prompt)
    if a: results["anthropic"] = a
    g = call_gemini(prompt)
    if g: results["gemini"] = g
    return results 