# LLM Visibility Analyzer - Minimal Working Baseline (MWB)

A clean, robust, and always-functional baseline for the LLM Visibility Analyzer.

## 🎯 What This Gives You

- **Always Works**: Simulation fallback when API keys are missing
- **Clean Architecture**: Separated concerns with robust error handling
- **Consistent Scoring**: Pure, testable scoring function
- **Real-time Logging**: Debug information in terminal and UI
- **Single Config**: Works locally and on Streamlit Cloud

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.streamlit/secrets.toml`:
```toml
# LLM Providers
OPENAI_API_KEY = "sk-your-actual-key"
ANTHROPIC_API_KEY = "sk-ant-your-actual-key"
GEMINI_API_KEY = "AIza-your-actual-key"

# Feature flags
REAL_ANALYSIS_ENABLED = true
SIMULATION_FALLBACK = true

# Tuning
REQUEST_TIMEOUT_SECS = 45
MAX_PROVIDER_TOKENS = 3000
```

### 3. Run the App
```bash
streamlit run streamlit_app.py --server.runOnSave true --logger.level=debug
```

## 🏗️ Architecture

```
src/llm_visibility/
├── utils/
│   ├── config.py      # Single source of truth for settings
│   └── logging.py     # Robust logging system
├── providers.py       # Robust LLM provider wrappers
└── scoring.py         # Pure scoring function

streamlit_app.py       # Clean UI with state management
```

## 🔧 Key Features

### Robust Provider Calls
- Automatic fallback to simulation
- Uniform response format: `{ok, data, simulated?}`
- Timeout and error handling

### Clean State Management
- `idle` → `fetching` → `scored` → `error`
- No more "half-loaded" states
- Clear user feedback

### Deterministic Scoring
- Pure function: `score(response) → dict`
- Weighted algorithm with consistency multiplier
- Comprehensive test coverage

## 🧪 Testing

```bash
# Run tests
pytest -q

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## 📊 What You'll See

1. **Clean UI**: Modern, responsive design
2. **Provider Status**: Clear indication of available APIs
3. **Analysis Results**: Expandable sections per provider
4. **Score Visualization**: Interactive charts and metrics
5. **Debug Panel**: Configuration and state information

## 🚨 Troubleshooting

### Port Already in Use
```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```

### Import Errors
- Ensure you're in the virtual environment
- Check that `src/` is in your Python path
- Verify all dependencies are installed

### API Key Issues
- Check `.streamlit/secrets.toml` format
- Verify keys are valid and have credits
- Use simulation mode for testing

## 🔄 Next Steps

This MWB provides a solid foundation. You can now:

1. **Add Real API Keys**: Get actual results from LLM providers
2. **Extend Scoring**: Add more sophisticated algorithms
3. **Enhance UI**: Add more visualization options
4. **Scale**: Add more providers or analysis types

## 📝 Development Workflow

```bash
# Make changes
git add -A
git commit -m "feat: add new feature"

# Pre-commit hooks run automatically
# Tests run in CI on push/PR

# Deploy to Streamlit Cloud
git push origin main
```

---

**Remember**: This baseline is designed to always work. If something breaks, the simulation fallback ensures users can still see the app in action while you debug the issue. 