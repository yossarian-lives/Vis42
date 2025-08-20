import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Try to get API keys from Streamlit secrets (for cloud deployment)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
except:
    # Fallback to environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if we have real API keys
HAS_REAL_APIS = any([OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY])

# ---------- Real LLM Provider Classes (when API keys available) ----------

class OpenAIProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            self.available = False
        except Exception:
            self.available = False
    
    async def ask_json(self, prompt: str, entity: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI visibility analyst. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return try_parse_json(content)
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return None

class AnthropicProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.available = True
        except ImportError:
            self.available = False
        except Exception:
            self.available = False
    
    async def ask_json(self, prompt: str, entity: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
            
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                system="You are an AI visibility analyst. Respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            return try_parse_json(content)
        except Exception as e:
            st.error(f"Anthropic API Error: {str(e)}")
            return None

class GeminiProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.available = True
        except ImportError:
            self.available = False
        except Exception:
            self.available = False
    
    async def ask_json(self, prompt: str, entity: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
            
        try:
            response = self.model.generate_content(
                prompt + "\n\nRespond with valid JSON only.",
                generation_config={"temperature": 0.1}
            )
            
            content = response.text
            return try_parse_json(content)
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return None

# Initialize providers if API keys are available
providers = {}
if OPENAI_API_KEY:
    providers["openai"] = OpenAIProvider(OPENAI_API_KEY)
if ANTHROPIC_API_KEY:
    providers["anthropic"] = AnthropicProvider(ANTHROPIC_API_KEY)
if GEMINI_API_KEY:
    providers["gemini"] = GeminiProvider(GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="LLM Visibility Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .score-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
    }
    .provider-badge {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
    }
    .finding-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .finding-card.warning {
        border-left-color: #ffc107;
    }
    .finding-card.error {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LLM Analysis Logic (Direct Implementation) ----------

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse strict JSON. If it fails, attempt to repair by extracting the
    first balanced-looking {...} block.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def seed_from_string(s: str) -> int:
    """Generate deterministic seed from string"""
    h = 2166136261
    for ch in s:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h

def clamp01(x: float) -> float:
    """Clamp value between 0 and 1"""
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def visibility_from_subscores(subscores: Dict[str, float]) -> float:
    """Calculate overall visibility score from subscores"""
    # Weights (tune as needed)
    W_RECOG = 0.45
    W_DETAIL = 0.25
    W_CONTEXT = 0.20
    W_COMP = 0.10
    
    base = (
        W_RECOG * subscores['recognition']
        + W_DETAIL * subscores['detail']
        + W_CONTEXT * subscores['context']
        + W_COMP * subscores['competitors']
    )  # 0..1
    
    # mild consistency multiplier: 0.85..1.0
    mult = 0.85 + 0.15 * clamp01(subscores['consistency'])
    return round(100.0 * clamp01(base * mult), 1)

def simulate_llm_analysis(entity: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Simulate LLM analysis with realistic scoring"""
    # Generate deterministic but realistic scores based on entity
    seed = seed_from_string(entity + "|" + (category or ""))
    
    # Simulate different entity types and their typical scores
    entity_lower = entity.lower()
    
    # High visibility entities (well-known brands/people)
    if any(name in entity_lower for name in ['tesla', 'apple', 'google', 'microsoft', 'amazon', 'netflix']):
        base_recognition = 0.9
        base_detail = 0.85
        base_context = 0.8
        base_competitors = 0.9
        base_consistency = 0.85
    # Medium visibility entities
    elif any(name in entity_lower for name in ['startup', 'ai', 'machine learning', 'blockchain', 'crypto']):
        base_recognition = 0.7
        base_detail = 0.75
        base_context = 0.7
        base_competitors = 0.8
        base_consistency = 0.75
    # Lower visibility entities
    else:
        base_recognition = 0.5
        base_detail = 0.6
        base_context = 0.5
        base_competitors = 0.6
        base_consistency = 0.7
    
    # Add some variation based on seed
    import random
    random.seed(seed)
    
    subscores = {
        'recognition': clamp01(base_recognition + random.uniform(-0.1, 0.1)),
        'detail': clamp01(base_detail + random.uniform(-0.1, 0.1)),
        'context': clamp01(base_context + random.uniform(-0.1, 0.1)),
        'competitors': clamp01(base_competitors + random.uniform(-0.1, 0.1)),
        'consistency': clamp01(base_consistency + random.uniform(-0.05, 0.05))
    }
    
    overall = visibility_from_subscores(subscores)
    
    # Generate realistic analysis data
    if 'tesla' in entity_lower:
        summary = "Tesla is an American electric vehicle and clean energy company founded by Elon Musk. Known for innovative electric cars, battery energy storage, and solar products."
        facts = [
            "Founded in 2003 by Martin Eberhard and Marc Tarpenning",
            "Elon Musk joined as chairman in 2004 and became CEO in 2008",
            "First electric car was the Tesla Roadster (2008)",
            "Model S launched in 2012, Model 3 in 2017",
            "Pioneered over-the-air software updates for vehicles",
            "Built Gigafactories for battery production",
            "Market cap often exceeds traditional automakers",
            "Developed Autopilot advanced driver-assistance system"
        ]
        competitors = ["Ford", "General Motors", "Nissan", "Rivian", "Lucid Motors", "Volkswagen", "BMW", "Hyundai"]
        industry_rank = 10
    elif 'apple' in entity_lower:
        summary = "Apple Inc. is an American multinational technology company that designs, develops, and sells consumer electronics, computer software, and online services."
        facts = [
            "Founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne",
            "Headquartered in Cupertino, California",
            "Revolutionized personal computing with Macintosh (1984)",
            "Introduced iPhone in 2007, changing mobile industry",
            "iPad launched in 2010, creating tablet market",
            "Apple Watch debuted in 2015",
            "One of world's most valuable companies",
            "Known for premium design and user experience"
        ]
        competitors = ["Samsung", "Microsoft", "Google", "Amazon", "Sony", "Dell", "HP", "Lenovo"]
        industry_rank = 1
    else:
        # Generic analysis for other entities
        summary = f"{entity} is a notable entity in the {category or 'general'} space with varying levels of recognition across different knowledge bases."
        facts = [
            f"{entity} has established presence in the market",
            "Multiple sources provide information about this entity",
            "Industry recognition varies by region and sector",
            "Competitive landscape includes several players",
            "Technology and innovation play key roles"
        ]
        competitors = ["Competitor A", "Competitor B", "Competitor C", "Alternative 1", "Alternative 2"]
        industry_rank = random.randint(5, 15)
    
    return {
        "entity": entity,
        "category": category or "auto-detected",
        "overall": overall,
        "subscores": subscores,
        "summary": summary,
        "facts": facts,
        "competitors": competitors,
        "industry_rank": industry_rank,
        "providers": [
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "overall": overall,
                "subscores": subscores
            }
        ],
        "notes": [
            f"Analysis based on simulated LLM responses for {entity}",
            "Scores reflect typical recognition patterns across major AI models"
        ]
    }

def analyze_visibility(entity: str, category: Optional[str], providers: List[str]) -> Dict[str, Any]:
    """Analyze entity visibility using real APIs or simulation"""
    
    # If we have real API keys and selected providers, use them
    if HAS_REAL_APIS and any(p in providers for p in ["openai", "anthropic", "gemini"]):
        return asyncio.run(analyze_with_real_apis(entity, category, providers))
    else:
        # Fallback to simulation
        time.sleep(1)
        return simulate_llm_analysis(entity, category)

async def analyze_with_real_apis(entity: str, category: Optional[str], selected_providers: List[str]) -> Dict[str, Any]:
    """Analyze using real LLM APIs"""
    
    # Create analysis prompt
    prompt = f"""
    Analyze the visibility of "{entity}" in the {category or 'general'} space.
    
    Provide a JSON response with this structure:
    {{
        "summary": "Brief description of the entity",
        "facts": ["fact1", "fact2", "fact3"],
        "competitors": ["competitor1", "competitor2"],
        "industry_rank": 5,
        "subscores": {{
            "recognition": 0.85,
            "detail": 0.75,
            "context": 0.8,
            "competitors": 0.9,
            "consistency": 0.85
        }}
    }}
    
    Base scores on how well-known and detailed the information is about this entity.
    """
    
    results = {}
    provider_results = []
    
    # Query each selected provider
    for provider_name in selected_providers:
        if provider_name in providers:
            provider = providers[provider_name]
            result = await provider.ask_json(prompt, entity)
            
            if result:
                results[provider_name] = result
                provider_results.append({
                    "provider": provider_name,
                    "model": provider_name,
                    "overall": visibility_from_subscores(result.get("subscores", {})),
                    "subscores": result.get("subscores", {})
                })
    
    # If no real API results, fallback to simulation
    if not results:
        return simulate_llm_analysis(entity, category)
    
    # Combine results (for now, use first successful result as base)
    first_result = list(results.values())[0]
    
    return {
        "entity": entity,
        "category": category or "auto-detected",
        "overall": visibility_from_subscores(first_result.get("subscores", {})),
        "subscores": first_result.get("subscores", {}),
        "summary": first_result.get("summary", ""),
        "facts": first_result.get("facts", []),
        "competitors": first_result.get("competitors", []),
        "industry_rank": first_result.get("industry_rank", 10),
        "providers": provider_results,
        "notes": [
            f"Analysis using real LLM APIs: {', '.join(selected_providers)}",
            "Scores based on actual AI model responses"
        ]
    }

# ---------- Streamlit UI Functions ----------

def create_radar_chart(data):
    """Create a radar chart for subscores"""
    categories = ['Recognition', 'Detail', 'Context', 'Competitors', 'Consistency']
    values = [
        data['subscores']['recognition'] * 100,
        data['subscores']['detail'] * 100,
        data['subscores']['context'] * 100,
        data['subscores']['competitors'] * 100,
        data['subscores']['consistency'] * 100
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Visibility Subscores',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Visibility Subscores Breakdown",
        height=400
    )
    
    return fig

def create_provider_comparison(data):
    """Create provider comparison chart"""
    if not data.get('providers') or len(data['providers']) < 2:
        return None
        
    providers = [p['provider'] for p in data['providers']]
    scores = [p['overall'] for p in data['providers']]
    
    fig = px.bar(
        x=providers,
        y=scores,
        title="Provider Comparison",
        labels={'x': 'Provider', 'y': 'Score'},
        color=scores,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 LLM Visibility Analyzer</h1>
        <p>Analyze brand & topic presence across AI knowledge spaces</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Status indicator
        if HAS_REAL_APIS:
            st.success("✅ Real LLM APIs Available")
            st.info(f"Connected to: {', '.join(providers.keys())}")
        else:
            st.warning("⚠️ Using Simulation Mode")
            st.info("Add API keys to .streamlit/secrets.toml for real analysis")
        
        st.divider()
        
        # Provider Selection
        st.subheader("🤖 LLM Providers")
        
        if HAS_REAL_APIS:
            # Real providers
            openai_enabled = st.checkbox("OpenAI (GPT-4o)", value="openai" in providers)
            anthropic_enabled = st.checkbox("Anthropic (Claude)", value="anthropic" in providers)
            gemini_enabled = st.checkbox("Google (Gemini)", value="gemini" in providers)
            
            providers_selected = []
            if openai_enabled and "openai" in providers:
                providers_selected.append("openai")
            if anthropic_enabled and "anthropic" in providers:
                providers_selected.append("anthropic")
            if gemini_enabled and "gemini" in providers:
                providers_selected.append("gemini")
        else:
            # Simulation mode
            st.info("Simulation mode - all providers enabled")
            providers_selected = ["openai", "anthropic", "gemini"]
            
        if not providers_selected:
            st.warning("Please select at least one provider")
            return
            
        st.divider()
        
        # Category Selection
        st.subheader("📂 Category (Optional)")
        category = st.selectbox(
            "Choose category or auto-detect",
            ["", "technology", "automotive", "person", "brand", "concept", "startup", "enterprise", "media"]
        )
        
        st.divider()
        
        # Recent Analyses
        if 'recent_analyses' in st.session_state:
            st.subheader("📚 Recent Analyses")
            for analysis in st.session_state.recent_analyses[-5:]:
                st.write(f"• {analysis['entity']}: {analysis['score']}/100")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Entity Analysis")
        entity = st.text_input(
            "Enter entity to analyze",
            placeholder="e.g., Tesla, Apple, Elon Musk, Machine Learning",
            help="Enter a company, person, concept, or topic to analyze"
        )
        
        if st.button("🚀 Analyze Visibility", type="primary", use_container_width=True):
            if not entity.strip():
                st.error("Please enter an entity to analyze")
                return
                
            if not providers_selected:
                st.error("Please select at least one LLM provider")
                return
            
            # Show loading
            with st.spinner("🔍 Analyzing visibility across LLMs..."):
                result = analyze_visibility(entity, category, providers_selected)
                
                if result:
                    # Store in session state for recent analyses
                    if 'recent_analyses' not in st.session_state:
                        st.session_state.recent_analyses = []
                    
                    st.session_state.recent_analyses.append({
                        'entity': entity,
                        'score': result['overall'],
                        'timestamp': datetime.now()
                    })
                    
                    # Display results
                    display_results(result)
                else:
                    st.error("Analysis failed. Please try again.")
    
    with col2:
        st.subheader("📊 Quick Stats")
        if 'recent_analyses' in st.session_state and st.session_state.recent_analyses:
            total_analyses = len(st.session_state.recent_analyses)
            avg_score = sum(a['score'] for a in st.session_state.recent_analyses) / total_analyses
            
            st.metric("Total Analyses", total_analyses)
            st.metric("Average Score", f"{avg_score:.1f}/100")
            
            # Top performers
            top_entities = sorted(st.session_state.recent_analyses, key=lambda x: x['score'], reverse=True)[:3]
            st.write("🏆 Top Performers:")
            for i, entity in enumerate(top_entities, 1):
                st.write(f"{i}. {entity['entity']}: {entity['score']}/100")
        else:
            st.info("No analyses yet. Start by analyzing an entity!")

def display_results(data):
    """Display analysis results in a beautiful format"""
    st.success(f"✅ Analysis complete for **{data['entity']}**!")
    
    # Overall Score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="score-display">{data['overall']}</div>
            <div style="text-align: center; font-size: 1.2rem; color: #666;">
                Overall Visibility Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary and Facts
    if data.get('summary'):
        st.subheader("📝 Summary")
        st.write(data['summary'])
    
    if data.get('facts'):
        st.subheader("🔍 Key Facts")
        for fact in data['facts']:
            st.write(f"• {fact}")
    
    # Subscores
    st.subheader("📊 Detailed Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        radar_fig = create_radar_chart(data)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        # Metrics grid
        subscores = data['subscores']
        
        metrics_data = [
            ("Recognition", subscores['recognition'] * 100, "How well the LLM recognizes the entity"),
            ("Detail", subscores['detail'] * 100, "Depth of factual information available"),
            ("Context", subscores['context'] * 100, "Understanding of industry position"),
            ("Competitors", subscores['competitors'] * 100, "Awareness of alternatives"),
            ("Consistency", subscores['consistency'] * 100, "Stability of responses")
        ]
        
        for metric, value, description in metrics_data:
            st.metric(metric, f"{value:.1f}/100", help=description)
    
    # Competitors
    if data.get('competitors'):
        st.subheader("🏆 Competitors & Alternatives")
        competitors_text = ", ".join(data['competitors'])
        st.write(competitors_text)
    
    # Industry Ranking
    if data.get('industry_rank'):
        st.subheader("📈 Industry Position")
        st.write(f"Ranked #{data['industry_rank']} in relevant industry category")
    
    # Provider Details
    if data.get('providers'):
        st.subheader("🤖 Provider Analysis")
        
        # Provider comparison chart
        provider_fig = create_provider_comparison(data)
        if provider_fig:
            st.plotly_chart(provider_fig, use_container_width=True)
        
        # Provider details
        for provider in data['providers']:
            with st.expander(f"📋 {provider['provider'].title()} Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model:** {provider['model']}")
                    st.write(f"**Score:** {provider['overall']}/100")
                
                with col2:
                    st.write(f"**Recognition:** {provider['subscores']['recognition']:.3f}")
                    st.write(f"**Detail:** {provider['subscores']['detail']:.3f}")
                    st.write(f"**Context:** {provider['subscores']['context']:.3f}")
    
    # Key Findings
    if data.get('notes'):
        st.subheader("💡 Analysis Notes")
        for note in data['notes']:
            st.info(note)
    
    # Raw Data (collapsible)
    with st.expander("🔍 Raw Analysis Data"):
        st.json(data)
    
    # Export Options
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV"):
            export_to_csv(data)
    
    with col2:
        if st.button("📄 Export Summary"):
            export_summary(data)

def export_to_csv(data):
    """Export results to CSV"""
    # Create DataFrame
    df_data = {
        'Metric': ['Overall Score', 'Recognition', 'Detail', 'Context', 'Competitors', 'Consistency'],
        'Value': [
            data['overall'],
            data['subscores']['recognition'] * 100,
            data['subscores']['detail'] * 100,
            data['subscores']['context'] * 100,
            data['subscores']['competitors'] * 100,
            data['subscores']['consistency'] * 100
        ]
    }
    
    df = pd.DataFrame(df_data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{data['entity']}_visibility_analysis.csv",
        mime="text/csv"
    )

def export_summary(data):
    """Export summary report"""
    summary = f"""
LLM Visibility Analysis Report
==============================

Entity: {data['entity']}
Category: {data.get('category', 'Auto-detected')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SCORE: {data['overall']}/100

Summary: {data.get('summary', 'N/A')}

Key Facts:
"""
    
    if data.get('facts'):
        for fact in data['facts']:
            summary += f"- {fact}\n"
    
    summary += f"""
Detailed Breakdown:
- Recognition: {data['subscores']['recognition'] * 100:.1f}/100
- Detail: {data['subscores']['detail'] * 100:.1f}/100
- Context: {data['subscores']['context'] * 100:.1f}/100
- Competitors: {data['subscores']['competitors'] * 100:.1f}/100
- Consistency: {data['subscores']['consistency'] * 100:.1f}/100

Competitors: {', '.join(data.get('competitors', []))}
Industry Rank: #{data.get('industry_rank', 'N/A')}

Provider Results:
"""
    
    for provider in data.get('providers', []):
        summary += f"- {provider['provider'].title()}: {provider['overall']}/100\n"
    
    if data.get('notes'):
        summary += "\nNotes:\n"
        for note in data['notes']:
            summary += f"- {note}\n"
    
    # Download button
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=f"{data['entity']}_visibility_summary.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main() 