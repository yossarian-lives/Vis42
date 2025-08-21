"""
Simple Working LLM Visibility Analyzer
This version should fix the consistent score 43 issue by bypassing complex imports
"""

import streamlit as st
import json
import os
import hashlib
import random
from typing import Dict, Any, Optional

# Page config
st.set_page_config(
    page_title="LLM Visibility Analyzer",
    page_icon="🎯",
    layout="wide"
)

def get_api_key(provider: str) -> Optional[str]:
    """Get API key from secrets or environment"""
    key_name = f"{provider.upper()}_API_KEY"
    
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            key = st.secrets[key_name]
            if isinstance(key, str) and key.strip():
                return key.strip()
    except:
        pass
    
    # Try environment variable
    key = os.getenv(key_name)
    if key and key.strip():
        return key.strip()
    
    return None

def call_openai_simple(entity: str, category: str) -> Optional[Dict[str, Any]]:
    """Simple OpenAI API call"""
    try:
        import openai
        
        api_key = get_api_key("OPENAI")
        if not api_key:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with these exact fields:
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
    "notes": "Brief analysis",
    "sources": ["example.com"]
}}

Use realistic scores 0-100 based on how well-known {entity} is."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an LLM visibility analyst. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Ensure required fields
        if "overall_score" not in result or "breakdown" not in result:
            return None
        
        return result
        
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

def call_anthropic_simple(entity: str, category: str) -> Optional[Dict[str, Any]]:
    """Simple Anthropic API call"""
    try:
        import anthropic
        
        api_key = get_api_key("ANTHROPIC")
        if not api_key:
            return None
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""Analyze the LLM visibility of "{entity}" in the {category} space.

Return ONLY a JSON object with these exact fields:
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
    "notes": "Brief analysis",
    "sources": ["example.com"]
}}

Use realistic scores 0-100 based on how well-known {entity} is."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            temperature=0.3,
            system="You are an LLM visibility analyst. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = json.loads(result_text)
        
        return result
        
    except Exception as e:
        st.error(f"Anthropic error: {e}")
        return None

def generate_smart_simulation(entity: str, category: str) -> Dict[str, Any]:
    """Generate realistic simulation results"""
    # Use entity name for consistent but varied results
    seed = int(hashlib.md5(entity.lower().encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    entity_lower = entity.lower()
    
    # Determine base score range based on entity recognition
    if any(term in entity_lower for term in ['tesla', 'apple', 'google', 'microsoft', 'amazon', 'netflix']):
        score_range = (80, 95)
        note = f"{entity} is a highly recognized global brand with excellent visibility"
    elif any(term in entity_lower for term in ['vuori', 'anthropic', 'openai', 'claude', 'chatgpt']):
        score_range = (65, 85)
        note = f"{entity} is well-known in specific communities with good visibility"
    elif any(term in entity_lower for term in ['ai', 'startup', 'tech', 'software']):
        score_range = (50, 75)
        note = f"{entity} has moderate visibility in the {category} space"
    else:
        score_range = (30, 60)
        note = f"{entity} has limited visibility, may benefit from increased digital presence"
    
    overall_score = random.randint(*score_range)
    
    # Generate breakdown scores with some variation
    variation = 20
    breakdown = {
        "recognition": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "media": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "context": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "competitors": max(0, min(100, overall_score + random.randint(-variation, variation))),
        "consistency": max(0, min(100, overall_score + random.randint(-10, 10)))
    }
    
    return {
        "entity": entity,
        "category": category,
        "overall_score": overall_score,
        "breakdown": breakdown,
        "notes": note + " (Simulation mode - add API keys for real analysis)",
        "sources": ["simulation.demo", "mock.data", "example.source"]
    }

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🎯 LLM Visibility Analyzer</h1>
        <p>Simple Working Version - Fixed Score Issue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API keys
    openai_key = get_api_key("OPENAI")
    anthropic_key = get_api_key("ANTHROPIC")
    gemini_key = get_api_key("GEMINI")
    
    st.subheader("🔑 API Key Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if openai_key:
            st.success(f"✅ OpenAI: {openai_key[:8]}...")
        else:
            st.error("❌ OpenAI: Missing")
    
    with col2:
        if anthropic_key:
            st.success(f"✅ Anthropic: {anthropic_key[:8]}...")
        else:
            st.error("❌ Anthropic: Missing")
    
    with col3:
        if gemini_key:
            st.success(f"✅ Gemini: {gemini_key[:8]}...")
        else:
            st.error("❌ Gemini: Missing")
    
    # Quick test button
    if st.button("🧪 Quick API Test"):
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say 'test successful'"}],
                    max_tokens=10
                )
                st.success(f"✅ OpenAI test: {response.choices[0].message.content}")
            except Exception as e:
                st.error(f"❌ OpenAI test failed: {e}")
        else:
            st.info("No OpenAI key to test")
    
    st.divider()
    
    # Main interface
    st.subheader("🎯 Entity Analysis")
    
    entity = st.text_input(
        "Enter entity to analyze",
        placeholder="e.g., Tesla, Apple, Vuori",
        help="Enter a brand, company, person, or topic"
    )
    
    category = st.text_input(
        "Category (optional)",
        placeholder="e.g., automotive, technology, activewear",
        value="general"
    )
    
    # Provider selection
    providers = []
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if openai_key and st.checkbox("Use OpenAI", value=True):
            providers.append("openai")
    
    with col2:
        if anthropic_key and st.checkbox("Use Anthropic", value=True):
            providers.append("anthropic")
    
    with col3:
        if gemini_key and st.checkbox("Use Gemini", value=False):
            providers.append("gemini")
    
    # Analysis button
    if st.button("🚀 Analyze Visibility", type="primary", disabled=not entity):
        if not entity:
            st.error("Please enter an entity")
            return
        
        st.subheader("📊 Analysis Results")
        
        results = []
        
        # Try each selected provider
        with st.spinner("Analyzing..."):
            if "openai" in providers:
                st.info("🤖 Calling OpenAI...")
                result = call_openai_simple(entity, category)
                if result:
                    st.success("✅ OpenAI analysis complete")
                    results.append(("OpenAI", result))
                else:
                    st.warning("⚠️ OpenAI analysis failed")
            
            if "anthropic" in providers:
                st.info("🤖 Calling Anthropic...")
                result = call_anthropic_simple(entity, category)
                if result:
                    st.success("✅ Anthropic analysis complete")
                    results.append(("Anthropic", result))
                else:
                    st.warning("⚠️ Anthropic analysis failed")
        
        # If no API results, use simulation
        if not results:
            st.info("🎭 No API responses - using simulation mode")
            sim_result = generate_smart_simulation(entity, category)
            results.append(("Simulation", sim_result))
        
        # Display results
        for provider_name, result in results:
            with st.expander(f"📋 {provider_name} Results", expanded=True):
                # Main score
                score = result["overall_score"]
                st.metric("Overall Visibility Score", f"{score}/100")
                
                # Score color indicator
                if score >= 80:
                    color = "#28a745"  # Green
                elif score >= 60:
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red
                
                st.markdown(f"""
                <div style="background: {color}; height: 10px; border-radius: 5px; margin: 10px 0;"></div>
                """, unsafe_allow_html=True)
                
                # Breakdown
                st.subheader("Detailed Breakdown")
                breakdown = result.get("breakdown", {})
                
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
                if result.get("notes"):
                    st.write("**📝 Analysis Notes:**")
                    st.write(result["notes"])
                
                if result.get("sources"):
                    st.write("**🔗 Sources:**")
                    st.write(", ".join(result["sources"]))
                
                # Raw JSON (collapsible)
                with st.expander("🔍 Raw JSON Response"):
                    st.json(result)
                
                # Check for the dreaded score 43
                if score == 43:
                    st.error("🚨 **ISSUE DETECTED:** Consistent score 43 suggests API calls are failing!")
                    st.error("This typically means:")
                    st.error("- API key is invalid or expired")
                    st.error("- API quota exceeded")
                    st.error("- Network/firewall blocking requests")
                    st.error("- Model access restrictions")
        
        # Download results
        if results:
            st.subheader("📤 Export Results")
            
            # Create CSV data
            csv_data = []
            for provider_name, result in results:
                csv_data.append({
                    "Provider": provider_name,
                    "Entity": result["entity"],
                    "Category": result["category"],
                    "Overall_Score": result["overall_score"],
                    "Recognition": result["breakdown"].get("recognition", 0),
                    "Media": result["breakdown"].get("media", 0),
                    "Context": result["breakdown"].get("context", 0),
                    "Competitors": result["breakdown"].get("competitors", 0),
                    "Consistency": result["breakdown"].get("consistency", 0),
                    "Notes": result.get("notes", ""),
                    "Sources": "; ".join(result.get("sources", []))
                })
            
            import pandas as pd
            df = pd.DataFrame(csv_data)
            
            csv_string = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_string,
                file_name=f"{entity}_visibility_analysis.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 