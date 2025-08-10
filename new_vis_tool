#!/usr/bin/env python3
"""
LLM Visibility & Sentiment — Terminal CLI (ChatGPT-only)

What it does:
1) Takes a business name or URL.
2) Calls ChatGPT (JSON-only) to discover context (brand, category, region, peers).
3) Calls ChatGPT (JSON-only) to estimate:
   - visibility_score (0..100)
   - sentiment_score (0..100)
   - confidence (0..100)
   - short reasons + likely peers
4) Prints a clean terminal summary with ASCII bars.

Install:
  pip install openai python-dotenv

Setup:
  .env with OPENAI_API_KEY=sk-...

Usage:
  python llm_visibility_cli.py "Port Sa'id Tel Aviv" --model gpt-4o-mini --seed 42
"""
import os, re, json, argparse, sys, time
from typing import Optional
from dotenv import load_dotenv

# ---------------- Config ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"  # swap to "gpt-4o" for max quality
DEFAULT_SEED = 42
MAX_RETRIES = 3

SYSTEM_JSON = "You are a neutral analyst. Output VALID JSON ONLY. No prose."

DISCOVER_PROMPT = (
    "Given a business name or URL, infer brief details.\n"
    "Return exactly this JSON schema:\n"
    "{\n"
    "  \"brand_canonical\":\"\",\n"
    "  \"category\":\"\",\n"
    "  \"region\":\"\",\n"
    "  \"competitors\":[],\n"
    "  \"notes\":\"\"\n"
    "}\n"
    "Rules:\n"
    "- brand_canonical: clean official name\n"
    "- category: concise (e.g., 'grocery retail', 'B2B SaaS', 'local restaurant')\n"
    "- region: country or city if obvious; else \"global\"\n"
    "- competitors: up to 5 relevant peer brands in same space/region\n"
    "- notes: 1 short line of widely-known context (no private data)\n"
)

SCORE_PROMPT_TPL = (
    "You will assess how visible and how positively referenced a brand is in typical ChatGPT answers.\n"
    "Context:\n"
    "- brand: {brand}\n"
    "- category: {category}\n"
    "- region: {region}\n"
    "- competitors_hint: {competitors}\n\n"
    "Return exactly this JSON schema:\n"
    "{\n"
    "  \"visibility_score\": 0,\n"
    "  \"sentiment_score\": 0,\n"
    "  \"confidence\": 0,\n"
    "  \"why_visibility\": \"\",\n"
    "  \"why_sentiment\": \"\",\n"
    "  \"top_peers\": []\n"
    "}\n"
    "Rules:\n"
    "- Be conservative; if brand is niche, lower visibility and confidence.\n"
    "- If information is sparse/ambiguous, reflect that in confidence.\n"
    "- Avoid marketing fluff; keep reasons tight.\n"
)

# -------------- OpenAI helper --------------
def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def chat_json(system: str, user: str, model: str, seed: Optional[int]) -> dict:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY (set it in .env).")
    client = OpenAI(api_key=OPENAI_API_KEY)

    last_err = None
    for i in range(MAX_RETRIES):
        try:
            params = dict(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if seed is not None:
                params["seed"] = seed  # supported by GPT-4o variants

            resp = client.chat.completions.create(**params)
            txt = resp.choices[0].message.content or "{}"
            data = _safe_json(txt)
            if isinstance(data, dict) and data:
                return data
            raise ValueError("Model did not return valid JSON.")
        except Exception as e:
            last_err = e
            if i < MAX_RETRIES - 1:
                time.sleep(0.8 * (2 ** i))
            else:
                raise RuntimeError(f"OpenAI call failed after retries: {e}") from e
    raise RuntimeError(f"OpenAI call failed: {last_err}")

# -------------- Pretty printing --------------
def bar(value: float, width: int = 40) -> str:
    value = max(0.0, min(100.0, float(value)))
    filled = int(round((value / 100.0) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {value:5.1f}"

def hr():
    print("-" * 64)

def sentiment_label(score: float) -> str:
    s = float(score)
    if s >= 75:  return "Positive"
    if s >= 50:  return "Mixed"
    if s >= 25:  return "Weak"
    return "Negative"

# -------------- Main logic --------------
def run_once(query: str, model: str, seed: Optional[int]) -> None:
    # 1) Discover minimal context
    user = f"{DISCOVER_PROMPT}\nBusiness: {query}\n"
    ctx = chat_json(SYSTEM_JSON, user, model, seed)

    brand = ctx.get("brand_canonical") or query.strip()
    category = ctx.get("category") or "general"
    region = ctx.get("region") or "global"
    competitors = ", ".join(ctx.get("competitors") or [])
    notes = ctx.get("notes") or ""

    # 2) Scoring
    score_prompt = SCORE_PROMPT_TPL.format(
        brand=brand, category=category, region=region, competitors=competitors or "[]"
    )
    res = chat_json(SYSTEM_JSON, score_prompt, model, seed)

    vis = float(res.get("visibility_score", 0))
    sent = float(res.get("sentiment_score", 50))
    conf = float(res.get("confidence", 50))
    why_v = (res.get("why_visibility") or "").strip()
    why_s = (res.get("why_sentiment") or "").strip()
    peers = res.get("top_peers", []) or []

    # 3) Output
    print(f"\nLLM Visibility & Sentiment — {brand}")
    hr()
    print(f"Category : {category}")
    print(f"Region   : {region}")
    if notes:
        print(f"Notes    : {notes}")
    if competitors:
        print(f"Peers(hint): {competitors}")
    hr()
    print(f"Visibility: {bar(vis)}")
    print(f"Sentiment : {bar(sent)}  ({sentiment_label(sent)})")
    print(f"Confidence: {bar(conf)}")
    hr()
    if why_v:
        print(f"Why (visibility): {why_v}")
    if why_s:
        print(f"Why (sentiment):  {why_s}")
    if peers:
        print(f"Likely peers in answers: {', '.join(peers)}")
    hr()
    # raw (for debugging/repro)
    print("RAW JSON — context:")
    print(json.dumps(ctx, ensure_ascii=False, indent=2))
    print("\nRAW JSON — scores:")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print()

def main():
    ap = argparse.ArgumentParser(description="Terminal LLM Visibility & Sentiment (ChatGPT)")
    ap.add_argument("query", help="Company name or URL (quoted)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (e.g., gpt-4o-mini, gpt-4o)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Use 0 to disable seeding")
    args = ap.parse_args()

    seed = args.seed if args.seed > 0 else None
    try:
        run_once(args.query, args.model, seed)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
