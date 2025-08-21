#!/usr/bin/env python3
"""
Setup script for LLM Visibility Analyzer MWB
Run this to verify your setup and get started quickly.
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and report success/failure"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_file_exists(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} missing: {path}")
        return False

def main():
    print("🎯 LLM Visibility Analyzer - MWB Setup Check")
    print("=" * 50)
    
    # Check key files
    print("\n📁 Checking required files...")
    files_to_check = [
        ("streamlit_app.py", "Main Streamlit app"),
        ("src/llm_visibility/utils/config.py", "Configuration module"),
        ("src/llm_visibility/utils/logging.py", "Logging module"),
        ("src/llm_visibility/providers.py", "Provider wrappers"),
        ("src/llm_visibility/scoring.py", "Scoring function"),
        (".streamlit/secrets.toml", "Secrets configuration"),
        ("requirements.txt", "Dependencies"),
        ("tests/test_scoring.py", "Test suite")
    ]
    
    all_files_exist = True
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing. Please check the setup.")
        return
    
    # Check Python environment
    print(f"\n🐍 Python version: {sys.version}")
    
    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
    else:
        print("⚠️  Virtual environment not detected")
        print("   Consider running: python -m venv .venv")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        from src.llm_visibility.utils.config import SETTINGS
        print("✅ Configuration imported")
        
        from src.llm_visibility.utils.logging import get_logger
        print("✅ Logging imported")
        
        from src.llm_visibility.providers import call_openai_robust
        print("✅ Providers imported")
        
        from src.llm_visibility.scoring import score
        print("✅ Scoring imported")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return
    
    # Check API keys
    print("\n🔑 Checking API key configuration...")
    if SETTINGS.openai_key and SETTINGS.openai_key != "sk-...":
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  OpenAI API key not configured (will use simulation)")
    
    if SETTINGS.anthropic_key and SETTINGS.anthropic_key != "sk-ant-...":
        print("✅ Anthropic API key configured")
    else:
        print("⚠️  Anthropic API key not configured (will use simulation)")
    
    if SETTINGS.gemini_key and SETTINGS.gemini_key != "AIza...":
        print("✅ Gemini API key configured")
    else:
        print("⚠️  Gemini API key not configured (will use simulation)")
    
    # Run tests
    print("\n🧪 Running tests...")
    if run_command("py -m pytest tests/test_scoring.py -q", "Running scoring tests"):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 MWB Setup Complete!")
    print("\n🚀 To run the app:")
    print("   streamlit run streamlit_app.py --server.runOnSave true --logger.level=debug")
    print("\n📚 For more information, see MWB_README.md")
    print("\n💡 The app will work in simulation mode even without API keys!")

if __name__ == "__main__":
    main() 