import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

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

# API configuration
API_BASE_URL = "http://localhost:5051"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_visibility(entity, category, providers):
    """Analyze entity visibility using the API"""
    try:
        payload = {
            "entity": entity,
            "providers": providers
        }
        if category:
            payload["category"] = category
            
        response = requests.post(
            f"{API_BASE_URL}/api/visibility",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

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
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info("Make sure your API is running on port 5051")
            return
        
        st.divider()
        
        # Provider Selection
        st.subheader("🤖 LLM Providers")
        openai_enabled = st.checkbox("OpenAI (GPT-4o)", value=True)
        anthropic_enabled = st.checkbox("Anthropic (Claude)")
        gemini_enabled = st.checkbox("Google (Gemini)")
        
        providers = []
        if openai_enabled:
            providers.append("openai")
        if anthropic_enabled:
            providers.append("anthropic")
        if gemini_enabled:
            providers.append("gemini")
            
        if not providers:
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
                
            if not providers:
                st.error("Please select at least one LLM provider")
                return
            
            # Show loading
            with st.spinner("🔍 Analyzing visibility across LLMs..."):
                result = analyze_visibility(entity, category, providers)
                
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
                    st.error("Analysis failed. Please check your API connection and try again.")
    
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
    with st.expander("🔍 Raw API Response"):
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

Detailed Breakdown:
- Recognition: {data['subscores']['recognition'] * 100:.1f}/100
- Detail: {data['subscores']['detail'] * 100:.1f}/100
- Context: {data['subscores']['context'] * 100:.1f}/100
- Competitors: {data['subscores']['competitors'] * 100:.1f}/100
- Consistency: {data['subscores']['consistency'] * 100:.1f}/100

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