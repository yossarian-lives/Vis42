"""
Debug script to test API keys and identify the scoring issue
Run this in your Streamlit app to debug the consistent score 43 problem
"""

import streamlit as st
import os
import json

def debug_api_keys():
    """Debug API key detection and usage"""
    st.title("🔍 API Key Debugging")
    
    # Check Streamlit secrets
    st.subheader("1. Streamlit Secrets Check")
    try:
        if hasattr(st, 'secrets'):
            st.success("✅ st.secrets is available")
            
            # Check each key
            openai_key = st.secrets.get("OPENAI_API_KEY")
            anthropic_key = st.secrets.get("ANTHROPIC_API_KEY")
            gemini_key = st.secrets.get("GEMINI_API_KEY")
            
            st.write("**Keys in secrets:**")
            st.write(f"- OPENAI_API_KEY: {'Found' if openai_key else 'Missing'}")
            if openai_key:
                st.write(f"  - Starts with: {openai_key[:10]}...")
                st.write(f"  - Length: {len(openai_key)}")
            
            st.write(f"- ANTHROPIC_API_KEY: {'Found' if anthropic_key else 'Missing'}")
            if anthropic_key:
                st.write(f"  - Starts with: {anthropic_key[:10]}...")
                st.write(f"  - Length: {len(anthropic_key)}")
            
            st.write(f"- GEMINI_API_KEY: {'Found' if gemini_key else 'Missing'}")
            if gemini_key:
                st.write(f"  - Starts with: {gemini_key[:10]}...")
                st.write(f"  - Length: {len(gemini_key)}")
        else:
            st.error("❌ st.secrets is not available")
    except Exception as e:
        st.error(f"❌ Error accessing secrets: {e}")
    
    # Check environment variables
    st.subheader("2. Environment Variables Check")
    env_openai = os.getenv("OPENAI_API_KEY")
    env_anthropic = os.getenv("ANTHROPIC_API_KEY")
    env_gemini = os.getenv("GEMINI_API_KEY")
    
    st.write("**Environment variables:**")
    st.write(f"- OPENAI_API_KEY: {'Found' if env_openai else 'Missing'}")
    st.write(f"- ANTHROPIC_API_KEY: {'Found' if env_anthropic else 'Missing'}")
    st.write(f"- GEMINI_API_KEY: {'Found' if env_gemini else 'Missing'}")
    
    # Test actual API calls
    st.subheader("3. API Connection Tests")
    
    # Test OpenAI
    if st.button("🧪 Test OpenAI API"):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ No OpenAI API key found")
            else:
                st.info(f"🔑 Using key: {api_key[:10]}...")
                
                import openai
                client = openai.OpenAI(api_key=api_key)
                
                # Simple test call
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say 'API test successful'"}],
                    max_tokens=10
                )
                
                result = response.choices[0].message.content
                st.success(f"✅ OpenAI API test successful: {result}")
                
                # Test with JSON mode
                st.info("Testing JSON mode...")
                json_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Return only JSON"},
                        {"role": "user", "content": 'Return JSON: {"test": true, "score": 85}'}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=50
                )
                
                json_result = json_response.choices[0].message.content
                st.success(f"✅ JSON mode test: {json_result}")
                
                # Parse the JSON
                parsed = json.loads(json_result)
                st.write(f"Parsed JSON: {parsed}")
                
        except Exception as e:
            st.error(f"❌ OpenAI API test failed: {e}")
            st.code(str(e))
    
    # Test Anthropic
    if st.button("🧪 Test Anthropic API"):
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("❌ No Anthropic API key found")
            else:
                st.info(f"🔑 Using key: {api_key[:10]}...")
                
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Say 'API test successful'"}]
                )
                
                result = response.content[0].text
                st.success(f"✅ Anthropic API test successful: {result}")
                
        except Exception as e:
            st.error(f"❌ Anthropic API test failed: {e}")
            st.code(str(e))
    
    # Full visibility test
    st.subheader("4. Full Visibility Analysis Test")
    
    test_entity = st.text_input("Test entity", value="Tesla")
    
    if st.button("🚀 Test Full Analysis"):
        if not test_entity:
            st.error("Enter a test entity")
        else:
            st.info(f"Testing analysis for: {test_entity}")
            
            # Test with our fixed functions
            try:
                from core.orchestrator import analyze_entity
                
                result = analyze_entity(test_entity, ["openai"])
                
                st.success("✅ Analysis completed!")
                st.json(result)
                
                # Check if we're getting the consistent score 43
                score = result.get("overall_score", 0)
                if score == 43:
                    st.error("🚨 FOUND THE ISSUE: Getting consistent score 43!")
                    st.error("This suggests the API calls are failing and falling back to simulation/default values")
                else:
                    st.success(f"✅ Got varied score: {score} (this is good!)")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    debug_api_keys() 