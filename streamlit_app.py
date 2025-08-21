"""
LLM Visibility Analyzer - Minimal Working Baseline
Clean, robust, and always functional.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.llm_visibility.utils.config import SETTINGS
from src.llm_visibility.utils.logging import get_logger
from src.llm_visibility.providers import call_openai_robust, call_anthropic_robust, call_gemini_robust
from src.llm_visibility.scoring import score

log = get_logger("vis42.ui")

# Page config
st.set_page_config(
    page_title="LLM Visibility Analyzer",
    page_icon="🎯",
    layout="wide"
)

@st.cache_data
def get_provider_status():
    """Cache provider metadata to avoid repeated checks"""
    return {
        "openai": bool(SETTINGS.openai_key),
        "anthropic": bool(SETTINGS.anthropic_key),
        "gemini": bool(SETTINGS.gemini_key)
    }

def render_score_chart(scores):
    """Render the score breakdown chart"""
    df = pd.DataFrame([
        {"Metric": k, "Score": v} for k, v in scores.items() if k != "Overall"
    ])
    
    fig = px.bar(
        df, 
        x="Metric", 
        y="Score",
        color="Score",
        color_continuous_scale="RdYlGn",
        title="Visibility Score Breakdown",
        range_color=[0, 100]
    )
    fig.update_layout(height=400)
    return fig

def main():
    log.info("App starting up")
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🎯 LLM Visibility Analyzer</h1>
        <p>Minimal Working Baseline - Always Functional</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Provider status
    provider_status = get_provider_status()
    real_providers = sum(provider_status.values())
    
    if real_providers == 0 and SETTINGS.real_enabled:
        st.warning("⚠️ No API keys configured. Running in simulation mode.")
        st.info("Add your API keys to `.streamlit/secrets.toml` for real analysis.")
    
    # Main input
    col1, col2 = st.columns([2, 1])
    with col1:
        entity = st.text_input("Entity to analyze", placeholder="e.g., Patagonia, Snowflake, Messi")
    with col2:
        category = st.text_input("Category/Industry", placeholder="e.g., Outdoor, Tech, Sports", value="Technology")
    
    run_button = st.button("🚀 Analyze", type="primary", disabled=not entity)
    
    # Main analysis flow
    if run_button and entity:
        log.info(f"Starting analysis for {entity} in {category}")
        
        # State management
        if "analysis_state" not in st.session_state:
            st.session_state.analysis_state = "idle"
        
        st.session_state.analysis_state = "fetching"
        
        with st.spinner("🔍 Analyzing visibility across providers..."):
            try:
                # Call all providers
                results = {}
                
                # OpenAI
                if provider_status["openai"]:
                    openai_result = call_openai_robust(entity, category)
                    if openai_result["ok"]:
                        results["OpenAI"] = openai_result
                
                # Anthropic
                if provider_status["anthropic"]:
                    anthropic_result = call_anthropic_robust(entity, category)
                    if anthropic_result["ok"]:
                        results["Anthropic"] = anthropic_result
                
                # Gemini
                if provider_status["gemini"]:
                    gemini_result = call_gemini_robust(entity, category)
                    if gemini_result["ok"]:
                        results["Gemini"] = gemini_result
                
                # Fallback to simulation if no real results
                if not results and SETTINGS.simulation_fallback:
                    log.info("Using simulation fallback")
                    results = {
                        "Simulation": {
                            "ok": True,
                            "data": {
                                "facts": [f"{entity} operates in the {category} space"],
                                "competitors": ["Competitor A", "Competitor B"],
                                "detail_score": 0.6,
                                "context_score": 0.7,
                                "recognition_score": 0.75,
                                "consistency_score": 0.8
                            },
                            "simulated": True
                        }
                    }
                
                if results:
                    st.session_state.analysis_state = "scored"
                    st.session_state.results = results
                    log.info(f"Analysis completed with {len(results)} providers")
                else:
                    st.session_state.analysis_state = "error"
                    st.error("❌ All providers failed. Check your configuration.")
                    
            except Exception as e:
                log.exception(f"Analysis failed: {e}")
                st.session_state.analysis_state = "error"
                st.error(f"❌ Analysis failed: {str(e)}")
    
    # Render results based on state
    if st.session_state.get("analysis_state") == "scored" and "results" in st.session_state:
        st.success("✅ Analysis complete!")
        
        results = st.session_state.results
        
        # Show provider results
        st.subheader("📊 Provider Results")
        for provider_name, result in results.items():
            with st.expander(f"{provider_name} {'(Simulated)' if result.get('simulated') else ''}"):
                if result["ok"]:
                    data = result["data"]
                    
                    # Show facts and competitors
                    if data.get("facts"):
                        st.write("**Key Facts:**")
                        for fact in data["facts"]:
                            st.write(f"• {fact}")
                    
                    if data.get("competitors"):
                        st.write("**Competitors:**")
                        for comp in data["competitors"]:
                            st.write(f"• {comp}")
                    
                    # Score the response
                    scores = score(data)
                    
                    # Overall score highlight
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="Overall Score",
                            value=f"{scores['Overall']:.1f}",
                            delta=f"{scores['Overall'] - 50:.1f}"
                        )
                    
                    # Score breakdown chart
                    st.plotly_chart(render_score_chart(scores), use_container_width=True)
                    
                    # Detailed scores table
                    st.write("**Detailed Scores:**")
                    score_df = pd.DataFrame([
                        {"Metric": k, "Score": f"{v:.1f}"} for k, v in scores.items()
                    ])
                    st.dataframe(score_df, use_container_width=True)
                    
                    # Confidence indicator
                    if result.get("simulated"):
                        st.info("🎭 This result was simulated due to missing API keys or provider failure.")
                    elif scores["Overall"] < 30:
                        st.warning("⚠️ Low confidence: This entity may have limited visibility data.")
                else:
                    st.error(f"Provider failed: {result.get('error', 'Unknown error')}")
    
    elif st.session_state.get("analysis_state") == "error":
        st.error("❌ Analysis encountered an error. Please try again.")
    
    elif st.session_state.get("analysis_state") == "fetching":
        st.info("🔄 Analysis in progress...")
    
    else:
        st.info("💡 Enter an entity and category above, then click Analyze to get started.")
    
    # Debug panel in sidebar
    if st.sidebar.checkbox("🔧 Show debug info"):
        st.sidebar.subheader("Debug Information")
        st.sidebar.json({
            "Provider Status": provider_status,
            "Settings": {
                "Real Analysis": SETTINGS.real_enabled,
                "Simulation Fallback": SETTINGS.simulation_fallback,
                "Request Timeout": SETTINGS.req_timeout,
                "Max Tokens": SETTINGS.max_tokens
            },
            "Analysis State": st.session_state.get("analysis_state", "idle")
        })
        
        if st.sidebar.button("Clear Session"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main() 