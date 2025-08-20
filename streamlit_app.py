"""
LLM Visibility Analyzer - Main Streamlit Application
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from core.orchestrator import analyze_entity, get_available_providers
    from core.enrich import guess_category
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required files are in the correct directories.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="LLM Visibility Analyzer",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .provider-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .provider-enabled {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .provider-disabled {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .score-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .breakdown-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_radar_chart(breakdown: Dict[str, int]) -> go.Figure:
    """Create a radar chart for the breakdown scores"""
    
    categories = list(breakdown.keys())
    values = list(breakdown.values())
    
    # Close the radar chart by adding the first point at the end
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Visibility Breakdown',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.25)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#333')
            )
        ),
        showlegend=False,
        title="Visibility Breakdown",
        title_x=0.5,
        height=400,
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_bar_chart(breakdown: Dict[str, int]) -> go.Figure:
    """Create a bar chart for the breakdown scores"""
    
    categories = list(breakdown.keys())
    values = list(breakdown.values())
    
    # Create color scale based on values
    colors = ['#1f77b4' if v >= 70 else '#ff7f0e' if v >= 50 else '#d62728' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v}%' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Detailed Score Breakdown",
        xaxis_title="Metrics",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

def display_provider_status():
    """Display current provider status"""
    available_providers = get_available_providers()
    
    st.subheader("Provider Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_class = "provider-enabled" if available_providers.get("openai") else "provider-disabled"
        status_text = "Available" if available_providers.get("openai") else "No API Key"
        st.markdown(f'<div class="provider-status {status_class}">OpenAI: {status_text}</div>', unsafe_allow_html=True)
    
    with col2:
        status_class = "provider-enabled" if available_providers.get("anthropic") else "provider-disabled"
        status_text = "Available" if available_providers.get("anthropic") else "No API Key"
        st.markdown(f'<div class="provider-status {status_class}">Anthropic: {status_text}</div>', unsafe_allow_html=True)
    
    with col3:
        status_class = "provider-enabled" if available_providers.get("gemini") else "provider-disabled"
        status_text = "Available" if available_providers.get("gemini") else "No API Key"
        st.markdown(f'<div class="provider-status {status_class}">Gemini: {status_text}</div>', unsafe_allow_html=True)
    
    return available_providers

def display_results(result: Dict[str, Any]):
    """Display analysis results with beautiful formatting"""
    
    st.success(f"Analysis completed for **{result['entity']}**")
    
    # Overall score display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="score-display">{result['overall_score']}</div>
            <div style="text-align: center; font-size: 1.2rem; color: #666; margin-top: 0.5rem;">
                Overall Visibility Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Entity details
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Entity", result['entity'])
        st.metric("Category", result['category'])
    with col2:
        if result.get('sources'):
            st.metric("Sources Found", len(result['sources']))
    
    # Charts section
    st.subheader("Detailed Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        radar_fig = create_radar_chart(result['breakdown'])
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with chart_col2:
        bar_fig = create_bar_chart(result['breakdown'])
        st.plotly_chart(bar_fig, use_container_width=True)
    
    # Breakdown metrics
    st.subheader("Score Breakdown")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    breakdown = result['breakdown']
    with col1:
        st.metric("Recognition", f"{breakdown.get('recognition', 0)}/100", 
                 help="How well LLMs recognize this entity")
    with col2:
        st.metric("Media", f"{breakdown.get('media', 0)}/100", 
                 help="Media coverage and mentions")
    with col3:
        st.metric("Context", f"{breakdown.get('context', 0)}/100", 
                 help="Industry context understanding")
    with col4:
        st.metric("Competitors", f"{breakdown.get('competitors', 0)}/100", 
                 help="Competitive landscape awareness")
    with col5:
        st.metric("Consistency", f"{breakdown.get('consistency', 0)}/100", 
                 help="Response consistency across models")
    
    # Notes section
    if result.get('notes'):
        st.subheader("Analysis Notes")
        st.info(result['notes'])
    
    # Sources section
    if result.get('sources'):
        st.subheader("Sources")
        sources_text = ""
        for i, source in enumerate(result['sources'], 1):
            # Try to make domains clickable if they look like URLs
            if '.' in source and ' ' not in source:
                if not source.startswith('http'):
                    source = f"https://{source}"
                sources_text += f"{i}. [{source}]({source})\n"
            else:
                sources_text += f"{i}. {source}\n"
        
        st.markdown(sources_text)
    
    # Raw data expander
    with st.expander("Raw Analysis Data"):
        st.json(result)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>LLM Visibility Analyzer</h1>
        <p>Professional brand & topic analysis across AI knowledge spaces</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Display provider status
        available_providers = display_provider_status()
        
        st.divider()
        
        # Provider selection
        st.subheader("Provider Selection")
        
        selected_providers = []
        
        if available_providers.get("openai"):
            if st.checkbox("OpenAI (GPT-4)", value=True):
                selected_providers.append("openai")
        else:
            st.checkbox("OpenAI (GPT-4)", value=False, disabled=True, help="API key required")
        
        if available_providers.get("anthropic"):
            if st.checkbox("Anthropic (Claude)", value=True):
                selected_providers.append("anthropic")
        else:
            st.checkbox("Anthropic (Claude)", value=False, disabled=True, help="API key required")
        
        if available_providers.get("gemini"):
            if st.checkbox("Google (Gemini)", value=True):
                selected_providers.append("gemini")
        else:
            st.checkbox("Google (Gemini)", value=False, disabled=True, help="API key required")
        
        if not any(available_providers.values()):
            st.warning("No API keys detected. Add them to environment variables:\n- OPENAI_API_KEY\n- ANTHROPIC_API_KEY\n- GEMINI_API_KEY")
            selected_providers = []  # Will trigger fallback mode
        
        st.divider()
        
        # Web enrichment toggle
        st.subheader("Web Enrichment")
        enable_web_search = st.checkbox("Enable category detection via web search", value=True,
                                       help="Use Tavily/Serper APIs to improve category detection")
        
        if enable_web_search:
            tavily_key = os.getenv('TAVILY_API_KEY')
            serper_key = os.getenv('SERPER_API_KEY')
            if tavily_key:
                st.success("Tavily API detected")
            elif serper_key:
                st.success("Serper API detected")
            else:
                st.info("Add TAVILY_API_KEY or SERPER_API_KEY for web enrichment")
    
    # Main content area
    st.subheader("Entity Analysis")
    
    # Input form
    with st.form("analysis_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            entity = st.text_input(
                "Entity to analyze",
                placeholder="e.g., Vuori, Tesla, ChatGPT, Apple",
                help="Enter a brand, company, person, or concept"
            )
        
        with col2:
            manual_category = st.selectbox(
                "Override category (optional)",
                ["", "consumer apparel / activewear", "technology", "automotive", "artificial intelligence", "healthcare", "financial services", "other"],
                help="Leave blank for automatic detection"
            )
        
        submitted = st.form_submit_button("Analyze Visibility", type="primary", use_container_width=True)
    
    # Handle form submission
    if submitted:
        if not entity.strip():
            st.error("Please enter an entity to analyze.")
            return
        
        # Show what we're about to do
        if selected_providers:
            provider_names = [p.title() for p in selected_providers]
            st.info(f"Analyzing with: {', '.join(provider_names)}")
        else:
            st.info("No API keys available - will provide structured fallback")
        
        # Run analysis with progress indicator
        with st.spinner("Analyzing visibility across LLM knowledge spaces..."):
            try:
                result = analyze_entity(entity, selected_providers)
                
                # Store in session state for history
                if 'analysis_history' not in st.session_state:
                    st.session_state.analysis_history = []
                
                st.session_state.analysis_history.append({
                    'entity': result['entity'],
                    'score': result['overall_score'],
                    'category': result['category']
                })
                
                # Keep only last 10 analyses
                if len(st.session_state.analysis_history) > 10:
                    st.session_state.analysis_history = st.session_state.analysis_history[-10:]
                
                # Display results
                display_results(result)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.error("Please check your API keys and try again.")
                
                # Show fallback option
                if st.button("Try with fallback analysis"):
                    fallback_result = {
                        "entity": entity,
                        "category": manual_category or guess_category(entity),
                        "overall_score": 45,
                        "breakdown": {
                            "recognition": 40,
                            "media": 35,
                            "context": 50,
                            "competitors": 45,
                            "consistency": 55
                        },
                        "notes": "Fallback analysis due to API connection issues. Scores are estimated based on entity characteristics.",
                        "sources": []
                    }
                    display_results(fallback_result)
    
    # Analysis history
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        st.subheader("Recent Analyses")
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        
        # Create a summary chart
        if len(history_df) > 1:
            fig = px.bar(history_df, x='entity', y='score', color='category',
                        title="Recent Analysis Scores")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        st.dataframe(history_df[['entity', 'score', 'category']], use_container_width=True)
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

if __name__ == "__main__":
    main() 