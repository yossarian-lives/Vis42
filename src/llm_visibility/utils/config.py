from dataclasses import dataclass
import os

def _get(name, default=None):
    # Streamlit secrets first, then env, then default
    try:
        import streamlit as st
        if "secrets" in dir(st) and st.secrets.get(name) is not None:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

@dataclass(frozen=True)
class Settings:
    openai_key: str = _get("OPENAI_API_KEY", "")
    anthropic_key: str = _get("ANTHROPIC_API_KEY", "")
    gemini_key: str = _get("GEMINI_API_KEY", "")
    real_enabled: bool = str(_get("REAL_ANALYSIS_ENABLED", "true")).lower()=="true"
    simulation_fallback: bool = str(_get("SIMULATION_FALLBACK","true")).lower()=="true"
    req_timeout: int = int(_get("REQUEST_TIMEOUT_SECS", 45))
    max_tokens: int = int(_get("MAX_PROVIDER_TOKENS", 3000))

SETTINGS = Settings() 