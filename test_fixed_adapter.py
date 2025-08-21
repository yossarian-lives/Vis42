"""
Test script for the fixed OpenAI adapter
Demonstrates proper error handling and fallback behavior
"""

from openai_adapter_fixed import test_openai_connection, analyze_with_openai, get_fallback_result

def test_connection():
    """Test the connection function"""
    print("🔌 Testing OpenAI connection...")
    ok, msg = test_openai_connection("Tesla")
    print(f"Result: {ok}")
    print(f"Message: {msg}")
    print()

def test_analysis():
    """Test the analysis function"""
    print("🔍 Testing OpenAI analysis...")
    result = analyze_with_openai("Tesla", "automotive")
    
    print(f"Simulated: {result.get('simulated', False)}")
    print(f"Provider: {result.get('provider', 'Unknown')}")
    
    if result.get('simulated'):
        print(f"Reason: {result.get('reason', 'Unknown')}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        if result.get('api_key_present'):
            print(f"API Key: Present (shows: {result.get('api_key_preview', 'N/A')})")
        else:
            print("API Key: Not present")
    else:
        print("✅ Real API call successful!")
        print(f"Overall Score: {result.get('overall_score', 'N/A')}")
        print(f"Notes: {result.get('notes', 'N/A')}")
    
    print()

def test_fallback():
    """Test the fallback function"""
    print("🎭 Testing fallback generation...")
    fallback = get_fallback_result("Tesla", "automotive", "API key invalid")
    
    print(f"Entity: {fallback['entity']}")
    print(f"Category: {fallback['category']}")
    print(f"Overall Score: {fallback['overall_score']}")
    print(f"Breakdown: {fallback['breakdown']}")
    print(f"Notes: {fallback['notes']}")
    print(f"Simulated: {fallback['simulated']}")
    print()

def main():
    print("🧪 Testing Fixed OpenAI Adapter")
    print("=" * 40)
    
    test_connection()
    test_analysis()
    test_fallback()
    
    print("✅ All tests completed!")
    print("\nKey improvements in this fixed adapter:")
    print("1. ✅ Proper error handling with specific guidance")
    print("2. ✅ No silent fallbacks - always shows why simulation occurred")
    print("3. ✅ Modern OpenAI SDK usage with timeouts")
    print("4. ✅ Comprehensive response validation")
    print("5. ✅ Detailed error reporting with tracebacks")
    print("6. ✅ Clear distinction between real API calls and simulations")

if __name__ == "__main__":
    main() 