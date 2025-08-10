#!/usr/bin/env python3
"""
LLM Visibility Score (ChatGPT-only, easy mode)

Steps:
1) User enters a business name or URL.
2) Ask ChatGPT to infer context (category, region, persona, competitors).
3) Ask ChatGPT for a strict Top-N ranking and ratings on 4 criteria.
4) Calculate a Visibility Score (VIS) = 100 × (0.6·RS + 0.4·RT).
"""

import os, re, json, argparse, time
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = "gpt-4o-mini"  # cost-efficient default
DEFAULT_SEED = 42
DEFAULT_TOPN = 5

SYSTEM_DISCOVER = (
    "You are a neutral market analyst. Output VALID JSON ONLY. No prose.\n"
    "Given a business name or URL, infer:\n"
    " - brand_canonical\n"
    " - category (e.g., 'grocery retail', 'local services', 'B2B SaaS')\n"
    " - region (country or city)\n"
    " - buyer_persona (short string)\n"
    " - competitors (up to 6 relevant competitors in same region)\n"
    " - notes (short fact)\n"
    "Schema:\n"
    "{"
    "\"brand_canonical\":\"\","
    "\"category\":\"\","
    "\"region\":\"\","
    "\"buyer_persona\":\"\","
    "\"competitors\":[],"
    "\"notes\":\"\""
    "}"
)

SYSTEM_RANKER = "You are a neutral evaluator. Output VALID JSON ONLY. No prose."

TOPN_TEMPLATE = (
    "Rank the most relevant brands for {category} for a typical {buyer_persona} in {region}. "
    "Return JSON EXACTLY: {\"ranking\":[{\"brand\":\"\",\"reason\":\"\"}]} "
    "with EXACTLY {N} items, no ties. "
    "Consider (but not limited to): {brand_list}."
)

RATINGS_TEMPLATE = (
    "Rate these brands for {category} for a {buyer_persona} in {region}. "
    "Criteria (0-10): {c1}, {c2}, {c3}, {c4}. "
    "Return JSON EXACTLY: {\"ratings\":[{\"brand\":\"\",\"{c1}\":0,\"{c2}\":0,\"{c3}\":0,\"{c4}\":0,\"notes\":\"\"}]} "
    "Brands: {brand_list}."
)

# Criteria presets
CRITERIA_PRESETS = [
    ("grocery", ["selection", "pricing", "freshness", "availability"]),
    ("local",   ["reputation", "proximity", "value", "consistency"]),
    ("pizza",   ["reputation", "taste", "value", "delivery"]),
    ("saas",    ["capabilities", "enterprise", "ecosystem", "value"]),
]
DEFAULT_CRITERIA = ["capabilities", "value", "trust", "availability"]

# ---------- Helpers ----------
def choose_criteria(category: str) -> List[str]:
    cat = (category or "").lower()
    for key, crit in CRITERIA_PRESETS:
        if key in cat:
            return crit
    return DEFAULT_CRITERIA

def safe_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try: return json.loads(m.group(0))
            except: return {}
    return {}

def call_openai(system: str, user: str, model: str, seed: Optional[int]) -> dict:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    client = OpenAI(api_key=OPENAI_API_KEY)
    params = dict(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    if seed is not None: params["seed"] = seed
    resp = client.chat.completions.create(**params)
    return safe_json(resp.choices[0].message.content or "{}")

# ---------- Core steps ----------
def discover_context(business: str, model: str, seed: int) -> dict:
    prompt = f"Business: {business}\nReturn the schema exactly."
    data = call_openai(SYSTEM_DISCOVER, prompt, model, seed)
    data.setdefault("brand_canonical", business)
    data.setdefault("category", "")
    data.setdefault("region", "Unknown")
    data.setdefault("buyer_persona", "consumer")
    data["competitors"] = [c for c in data.get("competitors", []) if c][:6]
    return data

def ask_topn(category, region, persona, brands, N, model, seed) -> dict:
    return call_openai(SYSTEM_RANKER, TOPN_TEMPLATE.format(
        category=category, buyer_persona=persona, region=region,
        N=N, brand_list=", ".join(brands)
    ), model, seed)

def ask_ratings(category, region, persona, brands, criteria, model, seed) -> dict:
    c1,c2,c3,c4 = criteria
    return call_openai(SYSTEM_RANKER, RATINGS_TEMPLATE.format(
        category=category, buyer_persona=persona, region=region,
        c1=c1, c2=c2, c3=c3, c4=c4, brand_list=", ".join(brands)
    ), model, seed)

# ---------- Scoring ----------
def harmonic(n): return sum(1/i for i in range(1, n+1))
@dataclass
class Scores: RS: float; RT: float; VIS: float

def score_topn(data, primary, N):
    ranks = [b.get("brand","").lower() for b in data.get("ranking",[])]
    if primary.lower() in ranks:
        return (1/(ranks.index(primary.lower())+1))/harmonic(N)
    return 0.0

def score_ratings(data, primary, criteria):
    row = next((x for x in data.get("ratings",[]) if x.get("brand","").lower()==primary.lower()), None)
    if not row: return 0.0
    vals = [float(row.get(c,0)) for c in criteria]
    return max(0,min(1,sum(vals)/(10*len(vals)))) if vals else 0.0

def compute_vis(RS, RT): return 100*(0.6*RS + 0.4*RT)

# ---------- Main ----------
def run_once(business, model, seed, N):
    ctx = discover_context(business, model, seed)
    primary = ctx["brand_canonical"]
    brands = [primary] + ctx.get("competitors",[])
    criteria = choose_criteria(ctx["category"])
    topn = ask_topn(ctx["category"], ctx["region"], ctx["buyer_persona"], brands, N, model, seed)
    ratings = ask_ratings(ctx["category"], ctx["region"], ctx["buyer_persona"], brands, criteria, model, seed)
    RS = score_topn(topn, primary, N)
    RT = score_ratings(ratings, primary, criteria)
    return ctx, topn, ratings, Scores(RS, RT, compute_vis(RS, RT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Business name or URL")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--topn", type=int, default=DEFAULT_TOPN)
    args = ap.parse_args()

    ctx, topn, ratings, sc = run_once(args.input, args.model, args.seed, args.topn)
    print("\n-- Context --\n", json.dumps(ctx, indent=2, ensure_ascii=False))
    print("\n-- Top-N --\n", json.dumps(topn, indent=2, ensure_ascii=False))
    print("\n-- Ratings --\n", json.dumps(ratings, indent=2, ensure_ascii=False))
    print(f"\nVIS for {ctx['brand_canonical']}: {sc.VIS:.1f} (RS={sc.RS:.2f}, RT={sc.RT:.2f})")

if __name__ == "__main__":
    main()
