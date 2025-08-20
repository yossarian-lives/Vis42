# 🔍 LLM Visibility Analyzer

**Professional brand & entity visibility analysis across multiple LLM providers**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

## 🎯 Overview

The LLM Visibility Analyzer is a comprehensive tool for analyzing how well brands, people, concepts, and entities are recognized across different Large Language Model providers. It provides both real-time analysis using actual LLM APIs and intelligent simulation for consistent results.

### ✨ Key Features

- **🤖 Multi-Provider Support** - OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini)
- **📊 Comprehensive Scoring** - Recognition, Detail, Context, Competitors, Consistency
- **🎨 Beautiful UI** - Interactive Streamlit interface with charts and visualizations
- **🔄 Smart Fallback** - Automatic simulation mode when API keys aren't available
- **📤 Export Options** - CSV data and text summaries
- **🛡️ Error-Safe** - Robust error handling with graceful degradation
- **🏗️ Professional Structure** - Clean, modular codebase following best practices

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm_visibility_easy

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)

Add your API keys to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-openai-key"
ANTHROPIC_API_KEY = "anthropic-your-key"
GEMINI_API_KEY = "AIza-your-gemini-key"
```

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` and start analyzing!

## 📁 Project Structure

```
llm_visibility_easy/
├── 📁 src/                    # Clean, modular source code
│   └── 📁 llm_visibility/     # Core package
│       ├── 📁 api/            # FastAPI backend
│       ├── 📁 streamlit/      # Streamlit frontend  
│       └── 📁 utils/          # Shared utilities
├── 📁 docs/                   # Documentation
├── 📁 scripts/                # Utility scripts
├── 📁 legacy/                 # Archived files
├── streamlit_app.py           # Main entry point
└── requirements.txt           # Dependencies
```

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed architecture information.

## 🔧 Usage

### Streamlit Interface

1. **Enter an entity** - Company, person, concept, or topic
2. **Select providers** - Choose which LLM APIs to use (if available)
3. **Choose category** - Optional categorization for better analysis
4. **Analyze** - Get comprehensive visibility scores and insights
5. **Export** - Download results as CSV or text summary

### Programmatic Usage

```python
from src.llm_visibility.utils.analysis import analyze_visibility

# Analyze an entity
result = analyze_visibility("Tesla", "automotive", ["openai"])

print(f"Overall Score: {result['overall']}/100")
print(f"Summary: {result['summary']}")
print(f"Competitors: {', '.join(result['competitors'])}")
```

## 📊 Scoring Methodology

The analyzer evaluates entities across five key dimensions:

- **Recognition** (45%) - How well the LLM recognizes the entity
- **Detail** (25%) - Depth of factual information available  
- **Context** (20%) - Understanding of industry position
- **Competitors** (10%) - Awareness of alternatives and competition
- **Consistency** (multiplier) - Stability across multiple queries

**Overall Score** = Weighted average × Consistency multiplier

## 🌐 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set `streamlit_app.py` as your main file
4. Add API keys in the Secrets section
5. Deploy!

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions.

### Local Development

```bash
# Run Streamlit app
streamlit run streamlit_app.py

# Run FastAPI backend (optional)
cd src/llm_visibility/api
uvicorn main:app --reload --port 5051
```

## 🔑 API Keys

The application works in two modes:

### **Real Analysis Mode**
- Requires API keys for OpenAI, Anthropic, or Gemini
- Provides actual LLM-based analysis
- More accurate and up-to-date results

### **Simulation Mode** 
- No API keys required
- Intelligent simulation based on entity recognition patterns
- Consistent, deterministic results for testing and demos

## 🛠️ Development

### Adding New LLM Providers

1. **Add provider config** in `src/llm_visibility/utils/providers.py`
2. **Implement safe wrapper** function
3. **Update aggregation** logic
4. **Test integration** across all applications

### Project Philosophy

- **Clean Architecture** - Separation of concerns with modular design
- **Error Safety** - Never crash the UI, always provide graceful fallbacks
- **Professional Quality** - Industry-standard code organization and practices
- **User Experience** - Beautiful, intuitive interfaces with helpful feedback

## 📈 Roadmap

- [ ] **Additional Providers** - Cohere, Hugging Face, local models
- [ ] **Advanced Analytics** - Trend analysis, comparative studies
- [ ] **API Enhancements** - Rate limiting, caching, batch processing
- [ ] **Testing Suite** - Comprehensive unit and integration tests
- [ ] **Documentation** - API docs, tutorials, examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o API
- **Anthropic** for Claude API  
- **Google** for Gemini API
- **Streamlit** for the amazing UI framework
- **FastAPI** for the high-performance backend framework

---

**Built with ❤️ for the LLM community**
