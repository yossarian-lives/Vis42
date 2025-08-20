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

# ---- Schema and Validation --------------------------------------------------

# Unified schema that all providers MUST return
VISIBILITY_SCHEMA = {
    "type": "object",
    "required": ["entity", "category", "overall_score", "breakdown", "notes", "sources"],
    "properties": {
        "entity": {"type": "string"},
        "category": {"type": "string"},
        "overall_score": {"type": "number", "minimum": 0, "maximum": 100},
        "breakdown": {
            "type": "object",
            "required": ["recognition", "media", "context", "competitors", "consistency"],
            "properties": {
                "recognition": {"type": "number", "minimum": 0, "maximum": 100},
                "media": {"type": "number", "minimum": 0, "maximum": 100},
                "context": {"type": "number", "minimum": 0, "maximum": 100},
                "competitors": {"type": "number", "minimum": 0, "maximum": 100},
                "consistency": {"type": "number", "minimum": 0, "maximum": 100}
            },
            "additionalProperties": False
        },
        "notes": {"type": "string", "maxLength": 600},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8
        }
    },
    "additionalProperties": False
}

def validate_result(data: dict) -> bool:
    """Validate that a result matches our schema"""
    try:
        # Simple validation without external jsonschema dependency
        if not isinstance(data, dict):
            return False
        
        required_fields = ["entity", "category", "overall_score", "breakdown", "notes", "sources"]
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate breakdown
        breakdown = data.get("breakdown", {})
        breakdown_fields = ["recognition", "media", "context", "competitors", "consistency"]
        for field in breakdown_fields:
            if field not in breakdown:
                return False
            if not isinstance(breakdown[field], (int, float)) or breakdown[field] < 0 or breakdown[field] > 100:
                return False
        
        # Validate overall_score
        if not isinstance(data["overall_score"], (int, float)) or data["overall_score"] < 0 or data["overall_score"] > 100:
            return False
        
        return True
    except Exception:
        return False

def get_fallback_result(entity: str, reason: str = "Structured fallback due to unparseable provider output.") -> dict:
    """Return a structured fallback result when providers fail"""
    return {
        "entity": entity,
        "category": "unknown",
        "overall_score": 40,
        "breakdown": {
            "recognition": 40,
            "media": 40,
            "context": 40,
            "competitors": 40,
            "consistency": 60
        },
        "notes": reason,
        "sources": []
    }

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

# ---- Fail-Safe Provider Calls ----------------------------------------------

def call_openai(prompt: str) -> str | None:
    if "OpenAI" not in ENABLED:
        return None
    try:
        from openai import OpenAI
        import httpx
        client = OpenAI(api_key=ENABLED["OpenAI"], http_client=httpx.Client(timeout=20))
        # Try gpt-4o-mini first, fallback to gpt-3.5-turbo if not available
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
        except Exception as model_error:
            # Fallback to gpt-3.5-turbo if gpt-4o-mini fails
            st.info("⚠️ gpt-4o-mini not available, trying gpt-3.5-turbo...")
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
        return resp.choices[0].message.content
    except Exception as e:
        # Log the error for debugging
        st.error(f"OpenAI API call failed: {str(e)}")
        return None

def parse_openai_response(response: str, entity: str, category: str) -> dict:
    """Parse OpenAI response and convert to structured schema format"""
    try:
        # Try to extract structured information from the response
        # This is a simple parser - you can enhance it based on your needs
        
        # Default values
        result = {
            "entity": entity,
            "category": category or "auto-detected",
            "overall_score": 50,  # Default middle score
            "breakdown": {
                "recognition": 50,
                "media": 50,
                "context": 50,
                "competitors": 50,
                "consistency": 50
            },
            "notes": response[:600],  # Truncate to max length
            "sources": []
        }
        
        # Try to extract score if mentioned
        import re
        score_match = re.search(r'(\d{1,3})/100|score[:\s]*(\d{1,3})|(\d{1,3})\s*out\s*of\s*100', response, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1) or score_match.group(2) or score_match.group(3))
            if 0 <= score <= 100:
                result["overall_score"] = score
        
        # Try to extract category if not provided
        if not category or category == "auto-detected":
            category_keywords = {
                "Technology": ["tech", "software", "ai", "artificial intelligence", "machine learning"],
                "Finance": ["finance", "banking", "investment", "crypto", "blockchain"],
                "Healthcare": ["health", "medical", "pharma", "biotech"],
                "Education": ["education", "learning", "academic", "university"],
                "Entertainment": ["entertainment", "media", "gaming", "film", "music"]
            }
            
            response_lower = response.lower()
            for cat, keywords in category_keywords.items():
                if any(keyword in response_lower for keyword in keywords):
                    result["category"] = cat
                    break
        
        return result
    except Exception as e:
        st.warning(f"Failed to parse OpenAI response: {str(e)}")
        return get_fallback_result(entity, f"Failed to parse response: {str(e)}")

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
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
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
    .score-chart {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header"><h1>🔍 LLM Visibility Analyzer</h1><p>Professional brand & topic analysis across AI knowledge spaces</p></div>', unsafe_allow_html=True)
    
    # Configuration section
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
        
        # Provider status
        st.subheader("🤖 LLM Providers")
        for p in ["OpenAI", "Anthropic", "Gemini"]:
            if p in ENABLED:
                st.markdown(f"✅ **{p}** enabled")
            else:
                st.markdown(f"⚪ **{p}** (no key)")
        
        # Test API Key button
        if st.button("🧪 Test API Key"):
            if "OpenAI" in ENABLED:
                with st.spinner("Testing OpenAI API..."):
                    try:
                        # Test with a simple call first
                        test_result = call_openai("Say 'Hello World' in one word.")
                        if test_result:
                            st.success(f"✅ API Test Successful: {test_result}")
                        else:
                            st.error("❌ API Test Failed - check the error above")
                    except Exception as e:
                        st.error(f"❌ Test failed with exception: {str(e)}")
                        st.info(f"Exception type: {type(e).__name__}")
            else:
                st.warning("No OpenAI API key available for testing")
    
    # Main content area
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
    
    # Debug captions
    st.caption("Secrets present: " + ", ".join(sorted(getattr(st, "secrets", {}).keys())))
    st.caption("Enabled providers: " + ", ".join(sorted(ENABLED.keys())) if ENABLED else "Enabled providers: none")
    
    # Entity analysis section
    st.header("🎯 Entity Analysis")
    
    # Input form
    with st.form("analysis_form"):
        entity = st.text_input("Enter entity to analyze", placeholder="e.g., Apple, ChatGPT, Tesla")
        category = st.selectbox(
            "📂 Category (Optional)",
            ["", "Technology", "Finance", "Healthcare", "Education", "Entertainment", "Other"],
            help="Choose category or auto-detect"
        )
        
        # Provider selection
        if ENABLED:
            st.subheader("Select Providers")
            providers_selected = []
            for p in ENABLED.keys():
                if st.checkbox(f"Use {p}", value=p in ENABLED):
                    providers_selected.append(p)
        else:
            providers_selected = []
            st.info("Simulation mode - all providers enabled")
        
        submitted = st.form_submit_button("🔍 Analyze Visibility")
    
    # Analysis results
    if submitted and entity:
        if not entity.strip():
            st.warning("Please enter an entity to analyze.")
            return
        
        # Show loading
        with st.spinner("🔍 Analyzing visibility across LLMs..."):
            # Try real API calls first
            if not SIMULATION_MODE:
                # Use our fail-safe OpenAI call
                if "OpenAI" in providers_selected and "OpenAI" in ENABLED:
                    prompt = f"""Analyze the visibility of '{entity}' in the AI/tech space. 
                    
Provide a comprehensive analysis including:
1. Overall visibility score (0-100)
2. Recognition level
3. Media presence
4. Context understanding
5. Competitive positioning
6. Brand consistency

Format your response with clear sections and scores."""
                    
                    # Show what we're trying to do
                    st.info(f"🔍 Attempting OpenAI API call with model fallback...")
                    
                    try:
                        result = call_openai(prompt)
                        if result:
                            # Parse the result using our schema
                            analysis_result = parse_openai_response(result, entity, category)
                            
                            # Validate the result
                            if validate_result(analysis_result):
                                st.success("✅ Analysis completed successfully with valid schema!")
                            else:
                                st.warning("⚠️ Analysis completed but schema validation failed - using fallback")
                                analysis_result = get_fallback_result(entity, "Schema validation failed")
                            
                            # Store in session state for recent analyses
                            if 'recent_analyses' not in st.session_state:
                                st.session_state.recent_analyses = []
                            
                            st.session_state.recent_analyses.append({
                                'entity': entity,
                                'score': analysis_result['overall_score'],
                                'timestamp': datetime.now()
                            })
                            
                            # Display results
                            display_results(analysis_result)
                            return
                        else:
                            # Show specific error information
                            st.error("❌ OpenAI API call returned no result")
                            st.info("💡 **Next Steps:**")
                            st.info("1. Use the '🧪 Test API Key' button above to test your key")
                            st.info("2. Check the error message that should appear above")
                            st.info("3. Verify your API key has credits and access")
                            return
                    except Exception as e:
                        st.error(f"❌ Error during OpenAI API call: {str(e)}")
                        st.info("💡 **Error Details:**")
                        st.info(f"Exception type: {type(e).__name__}")
                        st.info(f"Error message: {str(e)}")
                        return
            
            # Fallback to simulation if no real results
            if SIMULATION_MODE:
                st.info("Running in simulation mode - no API keys available")
                # Use fallback result for simulation
                analysis_result = get_fallback_result(entity, "Simulation mode - no real API calls")
                display_results(analysis_result)
            else:
                st.warning("API call failed. Check your API keys and try again.")
                st.info("💡 **Troubleshooting Tips:**")
                st.info("1. Verify your API key is valid and has credits")
                st.info("2. Check if the model 'gpt-4o-mini' is available")
                st.info("3. Ensure your OpenAI account has access to the API")
    
    # Quick stats section
    st.header("📊 Quick Stats")
    if 'recent_analyses' in st.session_state and st.session_state.recent_analyses:
        df = pd.DataFrame(st.session_state.recent_analyses)
        st.dataframe(df)
    else:
        st.info("No analyses yet. Start by analyzing an entity above.")

def display_results(results):
    """Display analysis results in a structured format using the schema"""
    st.header(f"📈 Analysis Results: {results['entity']}")
    
    # Overall score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Overall Visibility Score", f"{results['overall_score']}/100")
    
    # Breakdown scores
    st.subheader("📊 Detailed Breakdown")
    
    # Create a radar chart for the breakdown
    breakdown = results['breakdown']
    categories = list(breakdown.keys())
    values = list(breakdown.values())
    
    # Create the radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Visibility Scores',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Visibility Breakdown Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual metrics
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, len(categories), 2):
            if i < len(categories):
                st.metric(categories[i].title(), f"{values[i]}/100")
    with col2:
        for i in range(1, len(categories), 2):
            if i < len(categories):
                st.metric(categories[i].title(), f"{values[i]}/100")
    
    # Notes and sources
    if results.get('notes'):
        st.subheader("📝 Analysis Notes")
        st.write(results['notes'])
    
    if results.get('sources') and len(results['sources']) > 0:
        st.subheader("🔗 Sources")
        for source in results['sources']:
            st.write(f"• {source}")
    
    # Category info
    st.info(f"📂 **Category:** {results.get('category', 'Unknown')}")

if __name__ == "__main__":
    main() 