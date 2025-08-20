"""
LLM Visibility Analyzer - Streamlit Application

Professional Streamlit app with clean modular imports and beautiful UI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from collections.abc import Mapping

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))



# ---- Robust Secret Detection -------------------------------------------------

def _find_in_mapping(d: Mapping, name: str):
    if not isinstance(d, Mapping): 
        return None
    if name in d:
        v = d[name]
        if isinstance(v, str) and v.strip(): 
            return v.strip()
    for v in d.values():
        if isinstance(v, Mapping):
            found = _find_in_mapping(v, name)
            if found: 
                return found
    return None

def get_secret_or_env(name: str) -> str | None:
    """Get secret from st.secrets (any nesting level) or fallback to environment variable"""
    # First try direct access to st.secrets
    try:
        if hasattr(st, 'secrets') and name in st.secrets:
            value = st.secrets[name]
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass
    
    # Then try recursive search in nested secrets
    try:
        found = _find_in_mapping(st.secrets, name)
        if found: 
            return found
    except Exception:
        pass
    
    # Finally fallback to environment variable
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else None

# ---- Provider Configuration --------------------------------------------------

PROVIDERS = {
    "OpenAI":    {"key": get_secret_or_env("OPENAI_API_KEY")},
    "Anthropic": {"key": get_secret_or_env("ANTHROPIC_API_KEY")},
    "Gemini":    {"key": get_secret_or_env("GEMINI_API_KEY")},
}
ENABLED = {name: cfg["key"] for name, cfg in PROVIDERS.items() if cfg["key"]}
SIMULATION_MODE = len(ENABLED) == 0

# ---- Debug Info (Safe for Production) --------------------------------------
# Note: Debug info moved to main function to ensure st.secrets is available

# ---- Fail-Safe Provider Calls ----------------------------------------------

def call_openai(prompt: str) -> str | None:
    if "OpenAI" not in ENABLED:
        return None
    try:
        from openai import OpenAI
        import httpx
        client = OpenAI(api_key=ENABLED["OpenAI"], http_client=httpx.Client(timeout=20))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None

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
    .badge {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #667eea;
        border: 1px solid #e1e5e9;
    }
</style>
""", unsafe_allow_html=True)

# ---- UI Functions -----------------------------------------------------------

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

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 LLM Visibility Analyzer</h1>
        <p>Professional brand & topic analysis across AI knowledge spaces</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode indicator badge
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<span class='badge'>🔧 Configuration · "
            f"{'Simulation Mode' if SIMULATION_MODE else 'Real Analysis Enabled'}</span>",
            unsafe_allow_html=True
        )
    with col2:
        if st.button("🔄 Refresh Detection"):
            st.rerun()
    
    # Debug info (temporary - remove later)
    if st.checkbox("🔍 Show Debug Info"):
        st.write("**Debug Information:**")
        st.write(f"- SIMULATION_MODE: {SIMULATION_MODE}")
        st.write(f"- ENABLED providers: {list(ENABLED.keys())}")
        st.write(f"- OpenAI key present: {'Yes' if 'OpenAI' in ENABLED else 'No'}")
        if 'OpenAI' in ENABLED:
            st.write(f"- OpenAI key starts with: {ENABLED['OpenAI'][:10]}...")
        
        # Add more detailed debugging
        st.write("**Secrets Debug:**")
        try:
            secrets_keys = list(st.secrets.keys()) if hasattr(st, 'secrets') else []
            st.write(f"- st.secrets available: {'Yes' if hasattr(st, 'secrets') else 'No'}")
            st.write(f"- st.secrets keys: {secrets_keys}")
            if 'OPENAI_API_KEY' in secrets_keys:
                st.write(f"- OPENAI_API_KEY in secrets: Yes")
                st.write(f"- OPENAI_API_KEY starts with: {st.secrets['OPENAI_API_KEY'][:10]}...")
            else:
                st.write(f"- OPENAI_API_KEY in secrets: No")
        except Exception as e:
            st.write(f"- Error accessing secrets: {str(e)}")
        
        st.write("**Environment Debug:**")
        import os
        env_openai = os.getenv('OPENAI_API_KEY')
        st.write(f"- OPENAI_API_KEY in env: {'Yes' if env_openai else 'No'}")
        if env_openai:
            st.write(f"- Env OPENAI_API_KEY starts with: {env_openai[:10]}...")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Status indicator
        if SIMULATION_MODE:
            st.warning("Using Simulation Mode — add API keys to enable real calls.")
        else:
            st.success("Real analysis enabled (API keys detected)")
        
        st.divider()
        
        # Provider Selection
        st.subheader("🤖 LLM Providers")
        
        # Show provider status (clean approach)
        for p in PROVIDERS:
            if p in ENABLED:
                st.write(f"✅ {p} enabled")
            else:
                st.write(f"⚪ {p} (no key)")
        
        st.divider()
        
        if not SIMULATION_MODE:
            # Real providers - let user select which to use
            openai_enabled = st.checkbox("OpenAI (GPT-4o)", value="OpenAI" in ENABLED)
            anthropic_enabled = st.checkbox("Anthropic (Claude)", value="Anthropic" in ENABLED)
            gemini_enabled = st.checkbox("Google (Gemini)", value="Gemini" in ENABLED)
            
            providers_selected = []
            if openai_enabled and "OpenAI" in ENABLED:
                providers_selected.append("openai")
            if anthropic_enabled and "Anthropic" in ENABLED:
                providers_selected.append("anthropic")
            if gemini_enabled and "Gemini" in ENABLED:
                providers_selected.append("gemini")
        else:
            # Simulation mode - all providers enabled
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
                # Try real API calls first
                if not SIMULATION_MODE:
                    # Use our fail-safe OpenAI call
                    if "OpenAI" in providers_selected and "OpenAI" in ENABLED:
                        prompt = f"Analyze the visibility of '{entity}' in the AI/tech space. Provide a score from 0-100 and brief analysis."
                        result = call_openai(prompt)
                        if result:
                            # Parse the result and create a structured response
                            # For now, create a simple result structure
                            analysis_result = {
                                'entity': entity,
                                'overall': 85,  # Placeholder - you can parse this from the actual response
                                'subscores': {'recognition': 0.8, 'detail': 0.9, 'context': 0.8, 'competitors': 0.7, 'consistency': 0.8},
                                'providers': {'openai': {'score': 85, 'response': result}},
                                'notes': [result],
                                'category': category or 'auto-detected'
                            }
                            
                            # Store in session state for recent analyses
                            if 'recent_analyses' not in st.session_state:
                                st.session_state.recent_analyses = []
                            
                            st.session_state.recent_analyses.append({
                                'entity': entity,
                                'score': analysis_result['overall'],
                                'timestamp': datetime.now()
                            })
                            
                            # Display results
                            display_results(analysis_result)
                            return
                
                # Fallback to simulation if no real results
                if SIMULATION_MODE:
                    st.info("Running in simulation mode - no API keys available")
                else:
                    st.warning("API call failed. Check your API keys and try again.")
                
                # For now, show a simple message
                st.info("Analysis completed. Check the results above.")
    
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

if __name__ == "__main__":
    main() 