# RUN.md - LLM Visibility Analyzer Setup

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional but Recommended)
```bash
# Add to your shell profile or create .env file
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here" 
export GEMINI_API_KEY="AIza-your-gemini-key-here"

# Optional: Web enrichment APIs
export TAVILY_API_KEY="tvly-your-tavily-key"
export SERPER_API_KEY="your-serper-key"
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🔧 Configuration Options

### API Keys
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **Anthropic**: Get from https://console.anthropic.com/
- **Gemini**: Get from https://ai.google.dev/
- **Tavily** (optional): Get from https://tavily.com/
- **Serper** (optional): Get from https://serper.dev/

### Environment Variables
The app will automatically detect API keys from:
1. Environment variables
2. Streamlit secrets (`.streamlit/secrets.toml`)
3. `.env` file (if using python-dotenv)

### Fallback Mode
If no API keys are provided, the app runs in structured fallback mode with realistic simulated scores.

## 📁 Project Structure
```
.
├─ core/
│  ├─ schema.py           # JSON schema validation
│  ├─ prompt.py           # LLM prompt templates  
│  ├─ orchestrator.py     # Main coordination logic
│  └─ enrich.py           # Web search enrichment
├─ providers/
│  ├─ openai_adapter.py   # OpenAI API integration
│  ├─ anthropic_adapter.py # Anthropic API integration
│  └─ gemini_adapter.py   # Gemini API integration
├─ utils/
│  ├─ entity.py           # Entity name normalization
│  └─ json_utils.py       # Robust JSON parsing
├─ streamlit_app.py       # Main Streamlit UI
├─ requirements.txt       # Python dependencies
└─ RUN.md                 # This file
```

## ✅ Features

### ✅ JSON-only outputs enforced
- All providers return validated JSON matching unified schema
- Automatic fallback for parsing failures
- No "as of 2023 I don't know..." prose in UI

### ✅ Fallbacks keep charts alive  
- Structured fallback results when providers fail
- Charts always render with meaningful data
- Graceful degradation without crashes

### ✅ Vuori case treated as apparel
- Entity normalization: "VOURI" → "Vuori"  
- Category detection: "consumer apparel / activewear"
- Web enrichment for accurate categorization

### ✅ Provider toggles
- Dynamic provider selection in sidebar
- Real-time status indicators
- Works with any combination of available APIs

### ✅ Category enrichment optional
- Automatic category detection from entity names
- Optional web search via Tavily/Serper APIs
- Manual category override in UI

## 🎯 Usage Examples

### Basic Analysis
1. Enter entity: "Vuori"  
2. Select providers (or use all available)
3. Click "Analyze Visibility"
4. View comprehensive results with charts

### Advanced Configuration
1. Set manual category override
2. Enable/disable web enrichment  
3. Select specific providers
4. Export results for further analysis

## 🐛 Troubleshooting

### Common Issues
- **"Import error"**: Ensure all files are in correct directories
- **"No API key"**: Check environment variables are set correctly
- **"Analysis failed"**: Verify API keys are valid and have credits
- **"Timeout"**: Check internet connection, APIs have 20s timeout

### Debug Mode
The app includes built-in debug information showing:
- Provider availability status
- API key detection results  
- Error messages with suggestions

## 🔒 Security Notes
- API keys are never printed or logged
- All keys handled through environment variables
- Timeout and error handling prevent hanging
- No external dependencies beyond official API clients 