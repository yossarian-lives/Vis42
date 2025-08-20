"""
LLM Provider adapters for standardized probes
"""
from __future__ import annotations

import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from schemas import ProviderResponse


# JSON repair regex for extracting valid JSON from responses
JSON_REPAIR_RE = re.compile(r"\{[\s\S]*\}")

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse strict JSON. If it fails, attempt to repair by extracting 
    the first balanced-looking {...} block.
    """
    if not text.strip():
        return None
    
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Try to extract JSON block
    match = JSON_REPAIR_RE.search(text)
    if match:
        chunk = match.group(0)
        try:
            return json.loads(chunk)
        except Exception:
            pass
    
    return None


# Standard probe prompts and schemas
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


class BaseProvider:
    """Base class for LLM providers"""
    name: str = "base"

    def __init__(self, model_env: str, default_model: str):
        self.model = os.getenv(model_env, default_model)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Send a prompt and expect JSON response"""
        raise NotImplementedError

    async def run_probes(self, entity: str, category_hint: Optional[str], *, seed: int) -> ProviderResponse:
        """Run all four standardized probes in parallel"""
        
        # Construct prompts
        profile_prompt = (
            f"{PROFILE_SCHEMA_DESC}\nEntity: {entity}. If known, include competitors and a coarse category."
        )
        if category_hint:
            profile_prompt += f" Category hint: {category_hint}."

        context_prompt = (
            f"{CONTEXT_SCHEMA_DESC}\nTask: Provide a top-10 list of leading players for the category relevant to "
            f"'{entity}'. Use a widely-accepted industry framing. Include the entity's rank if it belongs in top 10."
        )
        if category_hint:
            context_prompt += f" Category hint: {category_hint}."

        alt_prompt = (
            f"{ALT_SCHEMA_DESC}\nTask: List 10 notable alternatives or substitutes for '{entity}'."
        )

        consistency_prompt = (
            f"{CONSISTENCY_SCHEMA_DESC}\nTask: Create a different top-10 ranking for the same category as '{entity}'. "
            "Use a slightly different perspective or criteria, but stay in the same industry space."
        )
        if category_hint:
            consistency_prompt += f" Category hint: {category_hint}."

        # Run probes in parallel
        try:
            results = await asyncio.gather(
                self.ask_json(profile_prompt, "profile", seed=seed + 1),
                self.ask_json(context_prompt, "context", seed=seed + 2),
                self.ask_json(alt_prompt, "alt", seed=seed + 3),
                self.ask_json(consistency_prompt, "consistency", seed=seed + 4),
                return_exceptions=True,
            )
        except Exception as e:
            # If gather itself fails, create empty response
            results = [Exception(str(e))] * 4

        # Process results
        raw_map: Dict[str, Any] = {}
        parsed_map: Dict[str, Dict[str, Any]] = {}
        labels = ["profile", "context", "alt", "consistency"]

        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                raw_map[label] = {"error": str(result)}
                parsed_map[label] = {}
            else:
                raw_text, parsed = result
                raw_map[label] = {"text": raw_text}
                parsed_map[label] = parsed or {}

        return ProviderResponse(
            profile=parsed_map.get("profile", {}),
            context=parsed_map.get("context", {}),
            alt=parsed_map.get("alt", {}),
            consistency=parsed_map.get("consistency", {}),
            raw=raw_map,
            model_name=self.model,
        )


class OpenAIProvider(BaseProvider):
    """OpenAI API provider using structured outputs"""
    name = "openai"

    def __init__(self):
        super().__init__("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("OpenAI library not installed. Run: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        def _make_request():
            try:
                # Use the newer chat completions API with JSON mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    seed=seed if hasattr(self.client.chat.completions, 'create') else None,  # Some models support seed
                )
                
                # Extract text content
                if response.choices and response.choices[0].message:
                    text = response.choices[0].message.content or ""
                else:
                    text = ""
                    
                return text
            except Exception as e:
                raise RuntimeError(f"OpenAI API error: {str(e)}")

        text = await asyncio.to_thread(_make_request)
        parsed = try_parse_json(text)
        return text, parsed


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider"""
    name = "anthropic"

    def __init__(self):
        super().__init__("ANTHROPIC_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"))
        
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Anthropic library not installed. Run: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        def _make_request():
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0,
                    system=SYSTEM_INSTRUCTIONS,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract text from response
                if hasattr(response, 'content') and response.content:
                    text = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            text += block.text
                        elif isinstance(block, dict) and 'text' in block:
                            text += block['text']
                        else:
                            text += str(block)
                else:
                    text = str(response)
                    
                return text
            except Exception as e:
                raise RuntimeError(f"Anthropic API error: {str(e)}")

        text = await asyncio.to_thread(_make_request)
        parsed = try_parse_json(text)
        return text, parsed


class GeminiProvider(BaseProvider):
    """Google Gemini API provider"""
    name = "gemini"

    def __init__(self):
        super().__init__("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("Google GenerativeAI library not installed. Run: pip install google-generativeai")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.client = genai

    async def ask_json(self, prompt: str, schema_hint: str, *, seed: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        def _make_request():
            try:
                # Configure model for JSON output
                generation_config = {
                    "temperature": 0,
                    "response_mime_type": "application/json"
                }
                
                model = self.client.GenerativeModel(
                    model_name=self.model,
                    generation_config=generation_config,
                    system_instruction=SYSTEM_INSTRUCTIONS
                )
                
                response = model.generate_content(prompt)
                
                # Extract text
                if hasattr(response, 'text'):
                    text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    else:
                        text = str(candidate)
                else:
                    text = str(response)
                    
                return text
            except Exception as e:
                raise RuntimeError(f"Gemini API error: {str(e)}")

        text = await asyncio.to_thread(_make_request)
        parsed = try_parse_json(text)
        return text, parsed