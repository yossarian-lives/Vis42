# LLM Visibility Analyzer - Project Structure

## 📁 Professional Directory Organization

```
llm_visibility_easy/
├── 📁 src/                          # Source code (clean, modular)
│   ├── __init__.py                  # Main package initialization
│   └── 📁 llm_visibility/           # Core package
│       ├── __init__.py              # Package exports
│       ├── 📁 api/                  # FastAPI backend
│       │   ├── __init__.py
│       │   └── main.py              # FastAPI application
│       ├── 📁 streamlit/            # Streamlit frontend
│       │   ├── __init__.py
│       │   └── app.py               # Main Streamlit application
│       └── 📁 utils/                # Shared utilities
│           ├── __init__.py          # Utility exports
│           ├── providers.py         # LLM provider management
│           ├── analysis.py          # Analysis engine
│           ├── scoring.py           # Scoring algorithms
│           └── json_utils.py        # JSON parsing utilities
├── 📁 docs/                         # Documentation
│   ├── PROJECT_STRUCTURE.md        # This file
│   └── DEPLOYMENT.md                # Deployment guide
├── 📁 scripts/                      # Utility scripts
│   ├── serve_frontend.py            # Frontend development server
│   ├── test_api.ps1                 # API testing script
│   ├── test_frontend.ps1            # Frontend testing script
│   └── 📁 frontend/                 # Legacy HTML frontend
│       └── index.html               # HTML/CSS/JS frontend
├── 📁 legacy/                       # Legacy files (archived)
│   ├── new ver vistool/             # Old version
│   ├── touch.env                    # Old environment file
│   └── setup                        # Old setup script
├── 📁 tests/                        # Test files (future)
├── 📁 .streamlit/                   # Streamlit configuration
│   └── secrets.toml                 # API keys (gitignored)
├── streamlit_app.py                 # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore rules
```

## 🏗️ Architecture Overview

### **Core Modules**

1. **`src/llm_visibility/utils/providers.py`**
   - Clean API key discovery
   - Safe wrapper functions for OpenAI, Anthropic, Gemini
   - Automatic provider detection and initialization

2. **`src/llm_visibility/utils/analysis.py`**
   - Main analysis engine
   - Real API integration with fallback simulation
   - Deterministic scoring for consistent results

3. **`src/llm_visibility/utils/scoring.py`**
   - Visibility score calculation algorithms
   - Subscore weighting and aggregation
   - Utility functions for score normalization

4. **`src/llm_visibility/utils/json_utils.py`**
   - Robust JSON parsing with error recovery
   - LLM response cleaning and validation

### **Applications**

1. **`src/llm_visibility/streamlit/app.py`**
   - Professional Streamlit application
   - Interactive UI with charts and visualizations
   - Export functionality (CSV, text summaries)

2. **`src/llm_visibility/api/main.py`**
   - FastAPI backend for programmatic access
   - RESTful API endpoints
   - Swagger/OpenAPI documentation

### **Entry Points**

1. **`streamlit_app.py`**
   - Main entry point for Streamlit application
   - Clean import structure with path management
   - Run with: `streamlit run streamlit_app.py`

## 🔧 Key Benefits

### **Modularity**
- **Separation of concerns** - Each module has a single responsibility
- **Reusable components** - Utilities can be imported across applications
- **Easy testing** - Individual modules can be tested in isolation

### **Professional Structure**
- **Industry standard** - Follows Python packaging best practices
- **Scalable** - Easy to add new providers, analysis methods, or applications
- **Maintainable** - Clear organization makes code easy to understand and modify

### **Clean Imports**
```python
# Before (monolithic)
from streamlit_app import analyze_visibility

# After (modular)
from src.llm_visibility.utils.analysis import analyze_visibility
from src.llm_visibility.utils.providers import ENABLED, SIMULATION_MODE
```

### **Development Workflow**
1. **Core logic** → `src/llm_visibility/utils/`
2. **Applications** → `src/llm_visibility/{streamlit,api}/`
3. **Scripts** → `scripts/`
4. **Documentation** → `docs/`
5. **Legacy** → `legacy/` (archived, not in active development)

## 🚀 Usage Examples

### **Running the Streamlit App**
```bash
streamlit run streamlit_app.py
```

### **Using the Analysis Engine Programmatically**
```python
from src.llm_visibility.utils.analysis import analyze_visibility

result = analyze_visibility("Tesla", "automotive", ["openai"])
print(f"Visibility Score: {result['overall']}/100")
```

### **Adding a New LLM Provider**
1. Add provider config to `src/llm_visibility/utils/providers.py`
2. Implement safe wrapper function
3. Update aggregation logic
4. Provider automatically available in all applications

This structure provides a **professional, scalable foundation** for the LLM Visibility Analyzer while maintaining clean separation between different components. 