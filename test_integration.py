"""
Test integration between fixed OpenAI adapter and diagnostic Streamlit app
"""

from providers.openai_adapter import test_openai_connection, analyze_with_openai

def test_integration():
    """Test that the fixed adapter works with the diagnostic app"""
    print("🧪 Testing Integration")
    print("=" * 40)
    
    # Test 1: Connection test
    print("1. Testing OpenAI connection...")
    ok, msg = test_openai_connection("Tesla")
    print(f"   Result: {ok}")
    print(f"   Message: {msg}")
    print()
    
    # Test 2: Analysis test
    print("2. Testing OpenAI analysis...")
    result = analyze_with_openai("Tesla", "automotive")
    
    print(f"   Simulated: {result.get('simulated', False)}")
    print(f"   Provider: {result.get('provider', 'Unknown')}")
    
    if result.get('simulated'):
        print(f"   Reason: {result.get('reason', 'Unknown')}")
        if result.get('error'):
            print(f"   Error: {result['error']}")
        if result.get('api_key_present'):
            print(f"   API Key: Present (shows: {result.get('api_key_preview', 'N/A')})")
        else:
            print("   API Key: Not present")
    else:
        print("   ✅ Real API call successful!")
        print(f"   Overall Score: {result.get('overall_score', 'N/A')}")
        print(f"   Notes: {result.get('notes', 'N/A')}")
    
    print()
    
    # Test 3: Check for score 43
    if result.get('result') and result['result'].get('overall_score') == 43:
        print("🚨 WARNING: Score 43 detected! This indicates the old simulation fallback.")
    else:
        print("✅ No score 43 detected - system is working correctly!")
    
    print()
    print("🎯 Next Steps:")
    print("1. Refresh your Streamlit app (it should now show the diagnostic version)")
    print("2. Click 'Test OpenAI Connection' to see detailed error messages")
    print("3. Run a full analysis to see if you get varied scores instead of 43")
    print("4. Check the diagnostics panel for API key status and configuration")

if __name__ == "__main__":
    test_integration() 