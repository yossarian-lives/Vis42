# streamlit visibility speed-test gauge (ChatGPT-only)
# deps: streamlit openai python-dotenv
import os, re, json, time, math
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

# ---------- config ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"   # great price/quality; change to "gpt-4o" for max quality
DEFAULT_SEED = 42
TOPN = 5

SYSTEM_DISCOVER = (
    "You are a neutral market analyst. Output VALID JSON ONLY. No prose.\n"
    "Infer for a given business name/URL:\n"
    "  - brand_canonical (clean name)\n"
    "  - category (e.g., 'grocery retail', 'local services', 'B2B SaaS')\n"
    "  - region (country or city if obvious)\n"
    "  - buyer_persona (short string)\n"
    "  - competitors (array of up to 6 relevant competitors in same region if possible)\n"
    "  - notes (1 short line)\n"
    "Schema EXACTLY:\n"
    "{\n"
    "  \"brand_canonical\":\"\",\n"
    "  \"category\":\"\",\n"
    "  \"region\":\"\",\n"
    "  \"buyer_persona\":\"\",\n"
    "  \"competitors\":[],\n"
    "  \"notes\":\"\"\n"
    "}"
)

SYSTEM_RANKER = "You are a neutral evaluator. Output VALID JSON ONLY per schema. No prose."

TOPN_TEMPLATE = (
    "Rank the most relevant brands for {category}, for a typical {buyer_persona} in {region}.\n"
    "Return JSON EXACTLY: {\"ranking\":[{\"brand\":\"\",\"reason\":\"\"}]} with EXACTLY {N} items, no ties.\n"
    "Consider (but do not limit to): {brand_list}. Be concise in 'reason'."
)

RATINGS_TEMPLATE = (
    "Rate these brands for {category} targeting a {buyer_persona} in {region}.\n"
    "Criteria (each 0-10): {c1}, {c2}, {c3}, {c4}\n"
    "Return JSON EXACTLY:\n"
    "{\n"
    "  \"ratings\": [\n"
    "    {\"brand\":\"\",\"{c1}\":0,\"{c2}\":0,\"{c3}\":0,\"{c4}\":0,\"notes\":\"\"}\n"
    "  ]\n"
    "}\n"
    "Brands: {brand_list}\n"
    "Keep 'notes' short."
)

CRITERIA_PRESETS = [
    ("grocery", ["selection", "pricing", "freshness", "availability"]),
    ("local",   ["reputation", "proximity", "value", "consistency"]),
    ("barber",  ["reputation", "proximity", "value", "consistency"]),
    ("pizza",   ["reputation", "taste", "value", "delivery"]),
    ("saas",    ["capabilities", "enterprise", "ecosystem", "value"]),
    ("startup", ["capabilities", "ecosystem", "momentum", "value"]),
]
DEFAULT_CRITERIA = ["capabilities", "value", "trust", "availability"]

# ---------- helpers ----------
def choose_criteria(category: str) -> List[str]:
    c = (category or "").lower()
    for key, crit in CRITERIA_PRESETS:
        if key in c:
            return crit
    return DEFAULT_CRITERIA

def safe_json(text: str) -> dict:
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
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    client = OpenAI(api_key=OPENAI_API_KEY)
    params = dict(
        model=model, temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}]
    )
    if seed is not None:
        params["seed"] = seed
    resp = client.chat.completions.create(**params)
    txt = resp.choices[0].message.content or "{}"
    return safe_json(txt)

def discover_context(business: str, model: str, seed: Optional[int]) -> dict:
    user = f"Business: {business}\nReturn the schema exactly."
    data = chat_json(SYSTEM_DISCOVER, user, model, seed)
    data.setdefault("brand_canonical", business.strip())
    data.setdefault("category", "")
    data.setdefault("region", "Israel")
    data.setdefault("buyer_persona", "local consumer")
    comps = [str(x).strip() for x in data.get("competitors", []) if str(x).strip()]
    data["competitors"] = comps[:6]
    return data

def build_brand_list(primary: str, competitors: List[str]) -> List[str]:
    seen, out = set(), []
    for b in [primary] + competitors:
        k = b.strip()
        if k and k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out

def ask_topn(category, region, persona, brand_list, N, model, seed):
    prompt = TOPN_TEMPLATE.format(
        category=category or "local services",
        buyer_persona=persona or "local consumer",
        region=region or "Israel",
        brand_list=", ".join(brand_list),
        N=N
    )
    return chat_json(SYSTEM_RANKER, prompt, model, seed)

def ask_ratings(category, region, persona, brand_list, criteria, model, seed):
    c1, c2, c3, c4 = criteria
    prompt = RATINGS_TEMPLATE.format(
        category=category or "local services",
        buyer_persona=persona or "local consumer",
        region=region or "Israel",
        c1=c1, c2=c2, c3=c3, c4=c4,
        brand_list=", ".join(brand_list)
    )
    return chat_json(SYSTEM_RANKER, prompt, model, seed)

def harmonic(n: int) -> float:
    return sum(1.0/i for i in range(1, n+1))

def score_topn(topn_json: dict, primary: str, N: int) -> float:
    Hn = harmonic(N)
    ranking = [str(x.get("brand","")).lower() for x in (topn_json.get("ranking") or [])]
    p = primary.lower()
    if p in ranking:
        r = ranking.index(p) + 1
        return (1.0 / r) / Hn
    return 0.0

def score_ratings(ratings_json: dict, primary: str, criteria: List[str]) -> float:
    items = ratings_json.get("ratings") or []
    p = primary.lower()
    row = next((x for x in items if str(x.get("brand","")).lower() == p), None)
    if not row:
        return 0.0
    vals = [float(row.get(c, 0)) for c in criteria]
    if not vals:
        return 0.0
    return max(0.0, min(1.0, sum(vals) / (10.0 * len(vals))))

def compute_vis(RS: float, RT: float) -> float:
    return 100.0 * (0.60*RS + 0.40*RT)

# ---------- gauge UI (HTML/CSS component) ----------
def render_gauge(value: float, spinning=False, label="Visibility"):
    # value: 0..100; map to -90..+90 degrees
    angle = -90 + (min(max(value,0),100) * 180.0/100.0)
    spin_css = """
    @keyframes spin {
      0% { transform: rotate(-90deg); }
      50% { transform: rotate(90deg); }
      100% { transform: rotate(-90deg); }
    }""" if spinning else ""

    needle_style = f"transform: rotate({angle}deg);" if not spinning else "animation: spin 1.2s linear infinite;"
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;">
      <div style="position:relative;width:260px;height:130px;overflow:hidden;">
        <div style="position:absolute;left:0;top:0;width:260px;height:260px;border-radius:50%;border:10px solid #e6e6e6;"></div>
        <!-- colored arc segments -->
        <div style="position:absolute;left:0;top:0;width:260px;height:260px;border-radius:50%;
                    background: conic-gradient(#e74c3c 0deg, #e67e22 60deg, #f1c40f 120deg, #2ecc71 180deg); clip-path: inset(0 0 50% 0 round 130px); opacity:0.85;"></div>
        <!-- needle -->
        <div style="position:absolute;left:130px;top:130px;transform-origin: 0 0; {needle_style}">
          <div style="width:120px;height:4px;background:#111;border-radius:2px;"></div>
        </div>
        <!-- hub -->
        <div style="position:absolute;left:126px;top:126px;width:8px;height:8px;border-radius:50%;background:#111;"></div>
        <!-- scale labels -->
        <div style="position:absolute;left:0;top:96px;width:100%;font:12px sans-serif;color:#666;display:flex;justify-content:space-between;">
          <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
        </div>
      </div>
      <div style="margin-top:8px;font:600 18px/1.2 sans-serif;">{label}: <span>{value:.1f}</span></div>
    </div>
    <style>{spin_css}</style>
    """
    st.components.v1.html(html, height=220)

# ---------- streamlit app ----------
st.set_page_config(page_title="LLM Visibility Speed Test", layout="centered")
st.title("⚡ LLM Visibility Speed Test")
st.caption("Paste a brand or URL → we auto-discover context, run a quick test, and show a spinning meter with your Visibility Score.")

col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_input("Brand or URL", placeholder="e.g., Shufersal / Port Sa'id Tel Aviv / Datadog")
with col2:
    model = st.text_input("Model", value=DEFAULT_MODEL)
seed = st.number_input("Seed (for repeatability)", value=DEFAULT_SEED, step=1, help="Use 0 to disable.")
run = st.button("Run Visibility Test", use_container_width=True)

if run:
    if not user_input.strip():
        st.error("Please enter a brand/URL.")
        st.stop()
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in .env")
        st.stop()

    # show spinning gauge while we call the API
    spin_placeholder = st.empty()
    with spin_placeholder:
        render_gauge(0, spinning=True, label="Testing…")

    # step 1: discover context
    try:
        ctx = discover_context(user_input.strip(), model, seed if seed>0 else None)
    except Exception as e:
        spin_placeholder.empty()
        st.error(f"Context discovery failed: {e}")
        st.stop()

    primary = ctx.get("brand_canonical") or user_input.strip()
    category = ctx.get("category") or "local services"
    region = ctx.get("region") or "Israel"
    persona = ctx.get("buyer_persona") or "local consumer"
    comps = ctx.get("competitors") or []
    brand_list = build_brand_list(primary, comps)
    criteria = choose_criteria(category)
    if len(criteria) < 4:
        criteria += (DEFAULT_CRITERIA * 2)[:4-len(criteria)]

    # step 2: top-N and ratings
    try:
        topn = ask_topn(category, region, persona, brand_list, TOPN, model, seed if seed>0 else None)
        ratings = ask_ratings(category, region, persona, brand_list, criteria, model, seed if seed>0 else None)
    except Exception as e:
        spin_placeholder.empty()
        st.error(f"Ranking failed: {e}")
        st.stop()

    # scoring
    def H(n): return sum(1.0/i for i in range(1, n+1))
    RS = 0.0
    ranking_list = [b.get("brand","") for b in (topn.get("ranking") or [])]
    if ranking_list:
        lwr = [x.lower() for x in ranking_list]
        if primary.lower() in lwr:
            r = lwr.index(primary.lower()) + 1
            RS = (1.0 / r) / H(TOPN)
    items = ratings.get("ratings") or []
    row = next((x for x in items if str(x.get("brand","")).lower()==primary.lower()), None)
    RT = 0.0
    if row:
        vals = [float(row.get(c, 0)) for c in criteria]
        if vals:
            RT = max(0.0, min(1.0, sum(vals)/(10.0*len(vals))))
    VIS = 100.0 * (0.60*RS + 0.40*RT)

    # replace spinning gauge with final gauge
    spin_placeholder.empty()
    render_gauge(VIS, spinning=False, label="Visibility")

    # details
    st.subheader("Discovered context")
    st.json(ctx)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Top-N ranking (model)")
        st.json(topn)
    with colB:
        st.subheader("Ratings (model)")
        st.json(ratings)

    st.subheader("Scores")
    st.write(f"**RS** (ranking, 0..1): {RS:.2f}")
    st.write(f"**RT** (ratings, 0..1): {RT:.2f}")
    st.write(f"**VIS** (0..100): **{VIS:.1f}**")
    st.caption(f"Criteria used: {', '.join(criteria)}")
