"""
LLM Visibility Analyzer - DIAGNOSTIC VERSION
This version includes all diagnostics to find and fix the simulation fallback issue
"""

import streamlit as st
import os
import json
import traceback
from typing import Tuple, Dict, Any, Optional

# Page config
st.set_page_config(
    page_title="LLM Visibility Analyzer - Diagnostic",
    page_icon="🔍",
    layout="wide"
)

# Utility functions
def mask(s):
    """Mask API key for display"""
    return "•••" if not s else f"{s[:3]}•••{s[-4:]}"

def get_api_keys():
    """Get all API keys with fallback"""
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    return {
        "openai": openai_key,
        "anthropic": anthropic_key, 
        "gemini": gemini_key
    }

def get_config():
    """Get configuration flags"""
    real_enabled = str(st.secrets.get("REAL_ANALYSIS_ENABLED", os.getenv("REAL_ANALYSIS_ENABLED", "true"))).lower() == "true"
    sim_fallback = str(st.secrets.get("SIMULATION_FALLBACK", os.getenv("SIMULATION_FALLBACK", "true"))).lower() == "true"
    openai_model = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    return {
        "real_enabled": real_enabled,
        "sim_fallback": sim_fallback,
        "openai_model": openai_model
    }

def openai_sanity_check(entity: str, api_key: str, model: str) -> Tuple[bool, str]:
    """Test OpenAI API with detailed error reporting"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise brand-visibility assistant."},
                {"role": "user", "content": f"In one sentence, say what {entity} is."}
            ],
            temperature=0.2,
            max_tokens=50,
            timeout=45
        )
        
        text = response.choices[0].message.content.strip()
        return True, f"Model: {model} · OK · '{text[:120]}'"
        
    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}"
        return False, f"OpenAI error: {error_details}"

def openai_visibility_analysis(entity: str, category: str, api_key: str, model: str) -> Dict[str, Any]:
    """Full OpenAI visibility analysis with error handling"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with this exact structure:
{{
    "entity": "{entity}",
    "category": "{category}",
    "overall_score": 75,
    "breakdown": {{
        "recognition": 80,
        "media": 70,
        "context": 75,
        "competitors": 85,
        "consistency": 80
    }},
    "notes": "Brief analysis of {entity}'s visibility",
    "sources": ["source1.com", "source2.com"]
}}

Base scores on how well-known {entity} is. Use realistic values 0-100."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an LLM visibility analyst. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Validate required fields
        if "overall_score" not in result or "breakdown" not in result:
            raise ValueError("Missing required fields in response")
        
        return {
            "ok": True,
            "provider": "OpenAI",
            "simulated": False,
            "result": result,
            "note": f"Real OpenAI analysis using {model}"
        }
        
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return {
            "ok": False,
            "provider": "OpenAI", 
            "simulated": True,
            "error": str(e),
            "trace": tb,
            "note": f"OpenAI failed: {type(e).__name__}: {e}"
        }

def anthropic_sanity_check(entity: str, api_key: str) -> Tuple[bool, str]:
    """Test Anthropic API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": f"In one sentence, say what {entity} is."}]
        )
        
        text = response.content[0].text.strip()
        return True, f"Claude OK · '{text[:120]}'"
        
    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}"
        return False, f"Anthropic error: {error_details}"

def generate_realistic_simulation(entity: str, category: str, reason: str) -> Dict[str, Any]:
    """Generate simulation with clear reasoning"""
    import hashlib
    import random
    
    # Consistent but varied scores
    seed = int(hashlib.md5(entity.lower().encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    # Entity-based scoring
    entity_lower = entity.lower()
    if any(term in entity_lower for term in ['tesla', 'apple', 'google', 'microsoft']):
        base_score = random.randint(80, 95)
    elif any(term in entity_lower for term in ['vuori', 'anthropic', 'openai']):
        base_score = random.randint(65, 85)
    else:
        base_score = random.randint(40, 75)
    
    # Generate breakdown
    variation = 15
    breakdown = {
        "recognition": max(0, min(100, base_score + random.randint(-variation, variation))),
        "media": max(0, min(100, base_score + random.randint(-variation, variation))),
        "context": max(0, min(100, base_score + random.randint(-variation, variation))),
        "competitors": max(0, min(100, base_score + random.randint(-variation, variation))),
        "consistency": max(0, min(100, base_score + random.randint(-10, 10)))
    }
    
    return {
        "ok": True,
        "provider": "Simulation",
        "simulated": True,
        "result": {
            "entity": entity,
            "category": category,
            "overall_score": base_score,
            "breakdown": breakdown,
            "notes": f"SIMULATION: {reason}. Add valid API keys for real analysis.",
            "sources": ["simulation.demo", "mock.data"]
        },
        "note": f"Simulation because: {reason}"
    }

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🔍 LLM Visibility Analyzer</h1>
        <p>Diagnostic Version - Find and Fix API Issues</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    api_keys = get_api_keys()
    config = get_config()
    
    # DIAGNOSTIC PANEL (always visible)
    with st.expander("🔎 Diagnostics", expanded=True):
        st.write("**API Key Status:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write({
                "OPENAI_API_KEY_present": bool(api_keys["openai"]),
                "OPENAI_API_KEY_masked": mask(api_keys["openai"] or ""),
            })
        
        with col2:
            st.write({
                "ANTHROPIC_API_KEY_present": bool(api_keys["anthropic"]),
                "ANTHROPIC_API_KEY_masked": mask(api_keys["anthropic"] or ""),
            })
        
        with col3:
            st.write({
                "GEMINI_API_KEY_present": bool(api_keys["gemini"]),
                "GEMINI_API_KEY_masked": mask(api_keys["gemini"] or ""),
            })
        
        st.write("**Configuration:**")
        st.write({
            "REAL_ANALYSIS_ENABLED": config["real_enabled"],
            "SIMULATION_FALLBACK": config["sim_fallback"],
            "OPENAI_MODEL": config["openai_model"]
        })
    
    # API Sanity Tests
    st.subheader("🧪 API Sanity Tests")
    
    test_entity = "Tesla"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test OpenAI Connection"):
            if api_keys["openai"]:
                with st.spinner("Testing OpenAI..."):
                    ok, note = openai_sanity_check(test_entity, api_keys["openai"], config["openai_model"])
                    if ok:
                        st.success(f"✅ {note}")
                    else:
                        st.error(f"❌ {note}")
            else:
                st.error("❌ No OpenAI API key found")
    
    with col2:
        if st.button("Test Anthropic Connection"):
            if api_keys["anthropic"]:
                with st.spinner("Testing Anthropic..."):
                    ok, note = anthropic_sanity_check(test_entity, api_keys["anthropic"])
                    if ok:
                        st.success(f"✅ {note}")
                    else:
                        st.error(f"❌ {note}")
            else:
                st.error("❌ No Anthropic API key found")
    
    st.divider()
    
    # Main Analysis Interface
    st.subheader("🎯 Visibility Analysis")
    
    entity = st.text_input(
        "Enter entity to analyze",
        placeholder="e.g., Tesla, Apple, Vuori",
        value="Tesla"
    )
    
    category = st.text_input(
        "Category",
        placeholder="e.g., automotive, technology",
        value="general"
    )
    
    # Provider selection
    providers_to_test = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if api_keys["openai"] and st.checkbox("OpenAI", value=True):
            providers_to_test.append("openai")
    
    with col2:
        if api_keys["anthropic"] and st.checkbox("Anthropic", value=False):
            providers_to_test.append("anthropic")
    
    with col3:
        if api_keys["gemini"] and st.checkbox("Gemini", value=False):
            providers_to_test.append("gemini")
    
    # Analysis button
    if st.button("🚀 Run Full Analysis", type="primary"):
        if not entity:
            st.error("Please enter an entity")
            return
        
        st.subheader("📊 Analysis Results")
        
        results = []
        
        # Test each provider
        if "openai" in providers_to_test:
            st.info("🤖 Testing OpenAI full analysis...")
            
            if not config["real_enabled"]:
                result = generate_realistic_simulation(entity, category, "REAL_ANALYSIS_ENABLED=false")
            elif not api_keys["openai"]:
                result = generate_realistic_simulation(entity, category, "No OpenAI API key")
            else:
                result = openai_visibility_analysis(entity, category, api_keys["openai"], config["openai_model"])
                
                if not result["ok"] and config["sim_fallback"]:
                    # Fallback to simulation
                    result = generate_realistic_simulation(entity, category, result["note"])
            
            results.append(result)
        
        # Display results with full diagnostics
        for result in results:
            provider = result["provider"]
            simulated = result.get("simulated", False)
            
            # Color code the provider name
            if simulated:
                provider_display = f"🎭 {provider} (SIMULATED)"
                border_color = "#ffc107"  # Yellow for simulation
            else:
                provider_display = f"✅ {provider} (REAL)"
                border_color = "#28a745"  # Green for real
            
            with st.container():
                st.markdown(f"""
                <div style="border: 3px solid {border_color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                """, unsafe_allow_html=True)
                
                st.subheader(provider_display)
                
                # Show the diagnostic note prominently
                if result.get("note"):
                    if simulated:
                        st.warning(f"⚠️ **Why Simulation?** {result['note']}")
                    else:
                        st.success(f"✅ **Status:** {result['note']}")
                
                # Show error details if available
                if result.get("error"):
                    st.error(f"**Error:** {result['error']}")
                    
                    if result.get("trace"):
                        with st.expander("🔍 Error Traceback"):
                            st.code(result["trace"])
                
                # Show results if available
                if result.get("result"):
                    analysis_result = result["result"]
                    
                    # Main score
                    score = analysis_result["overall_score"]
                    st.metric("Overall Visibility Score", f"{score}/100")
                    
                    # Check for the dreaded score 43
                    if score == 43:
                        st.error("🚨 **FOUND THE ISSUE!** Score 43 indicates fallback to default simulation values")
                        st.error("This means your API calls are failing silently. Check the error details above.")
                    
                    # Breakdown
                    breakdown = analysis_result.get("breakdown", {})
                    if breakdown:
                        st.write("**Breakdown Scores:**")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Recognition", f"{breakdown.get('recognition', 0)}/100")
                        with col2:
                            st.metric("Media", f"{breakdown.get('media', 0)}/100")
                        with col3:
                            st.metric("Context", f"{breakdown.get('context', 0)}/100")
                        with col4:
                            st.metric("Competitors", f"{breakdown.get('competitors', 0)}/100")
                        with col5:
                            st.metric("Consistency", f"{breakdown.get('consistency', 0)}/100")
                    
                    # Notes and sources
                    if analysis_result.get("notes"):
                        st.write(f"**📝 Notes:** {analysis_result['notes']}")
                    
                    if analysis_result.get("sources"):
                        st.write(f"**🔗 Sources:** {', '.join(analysis_result['sources'])}")
                    
                    # Raw JSON
                    with st.expander("🔍 Raw JSON Response"):
                        st.json(analysis_result)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Summary and next steps
        if results:
            st.subheader("🔧 Next Steps")
            
            simulated_count = sum(1 for r in results if r.get("simulated", False))
            real_count = len(results) - simulated_count
            
            if simulated_count > 0:
                st.warning(f"⚠️ {simulated_count} provider(s) fell back to simulation")
                st.write("**To fix simulation fallbacks:**")
                st.write("1. Check the error messages above for specific API issues")
                st.write("2. Verify your API keys are correct and have sufficient credits")
                st.write("3. Ensure the model names are accessible to your account")
                st.write("4. Check for rate limiting or permission issues")
            
            if real_count > 0:
                st.success(f"✅ {real_count} provider(s) working correctly with real API calls")

if __name__ == "__main__":
    main() 