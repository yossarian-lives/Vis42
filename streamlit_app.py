"""
LLM Visibility Analyzer - Streamlit Application Entry Point

This is the main entry point for Streamlit Cloud deployment.
It imports and runs the main application from the src directory.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main Streamlit app
from llm_visibility.streamlit.app import main

if __name__ == "__main__":
    main() 