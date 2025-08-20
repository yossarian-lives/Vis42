# RUN.md - Mission-Critical LLM Visibility Analyzer

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional but Recommended)
```bash
# Core LLM Provider API Keys
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here" 
export GEMINI_API_KEY="AIza-your-gemini-key-here"

# Optional: Web enrichment APIs for enhanced category detection
export TAVILY_API_KEY="tvly-your-tavily-key"
export SERPER_API_KEY="your-serper-key"

# Optional: App configuration
export APP_URL="https://your-deployed-app.com"  # For LinkedIn sharing
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🎯 Mission-Critical Features

### Analysis Modes

**Basic Analysis**: Traditional visibility scoring with 5 core metrics
- Recognition, Media, Context, Competitors, Consistency
- Single query per provider
- Fast results (15-30 seconds)

**Mission-Critical Analysis**: Advanced multi-variant approach
- 5 specialized query types: Recognition, Ranking, Sentiment, Frequency, Comparison
- Multiple queries per provider for bias reduction
- Comprehensive scoring including share of voice and market position
- LinkedIn-ready results with sentiment breakdown
- Enhanced reporting (60-90 seconds)

### Key Metrics

1. **Mention Frequency** (30% weight) - How often the brand appears in LLM responses
2. **Market Ranking** (25% weight) - Position relative to competitors in top-10 lists
3. **Sentiment Analysis** (20% weight) - Positive/neutral/negative perception
4. **Brand Recognition** (15% weight) - Basic awareness and identification
5. **Competitive Strength** (10% weight) - Market differentiation and advantages

### Query Best Practices (Built-in)

- **Neutral Prompts**: Unbiased, informational queries
- **Multiple Variants**: 5 different query types to reduce prompt bias
- **Consistency Checks**: Cross-validation across providers
- **Anchor Brands**: Baseline entities for sanity checking
- **Frequency Sampling**: Multiple runs for statistical significance

## 🔧 Configuration Options

### API Keys
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **Anthropic**: Get from https://console.anthropic.com/
- **Gemini**: Get from https://ai.google.dev/
- **Tavily** (optional): Get from https://tavily.com/
- **Serper** (optional): Get from https://serper.dev/

### Environment Variables
The app automatically detects API keys from:
1. Environment variables
2. Streamlit secrets (`.streamlit/secrets.toml`)
3. `.env` file (if using python-dotenv)

### Fallback Modes
- **No API Keys**: Structured simulation with realistic scores
- **API Failures**: Automatic fallback with error reporting
- **Mixed Success**: Partial results with provider status indicators

## 📁 Enhanced Project Structure
```
.
├─ core/
│  ├─ schema.py           # Unified JSON schema validation
│  ├─ prompt.py           # Multi-variant prompt templates
│  ├─ orchestrator.py     # Mission-critical orchestration
│  └─ enrich.py           # Web search category enrichment
├─ providers/
│  ├─ openai_adapter.py   # OpenAI with custom prompt support
│  ├─ anthropic_adapter.py # Anthropic with custom prompt support
│  └─ gemini_adapter.py   # Gemini with custom prompt support
├─ utils/
│  ├─ entity.py           # Entity normalization ("VOURI" → "Vuori")
│  └─ json_utils.py       # Robust JSON coercion
├─ streamlit_app.py       # Enhanced mission-critical UI
├─ requirements.txt       # Python dependencies
└─ RUN.md                 # This file
```

## ✅ Mission-Critical Features Delivered

### ✅ Multi-Variant Query System
- Recognition analysis with accuracy scoring
- Market ranking with competitive positioning
- Sentiment analysis with positive/negative breakdown
- Mention frequency with share of voice calculation
- Competitive comparison with market differentiation

### ✅ Enhanced Scoring Methodology
- **Frequency-based**: Measures actual mention rates vs competitors
- **Position-aware**: Rankings in industry top-10 lists
- **Sentiment-weighted**: Positive/neutral/negative perception analysis
- **Bias-reduced**: Multiple query variants prevent prompt engineering issues
- **Statistically sound**: Median aggregation across providers

### ✅ LinkedIn-Ready Sharing
- Auto-generated share text with performance messaging
- Downloadable report cards (JSON format)
- One-click LinkedIn sharing integration
- Professional visibility cards for social media

### ✅ Advanced UI Features
- Mission-critical gauge visualizations
- Progress tracking for comprehensive analysis
- Sentiment breakdown charts
- Frequency comparison graphs
- Analysis history with trend visualization
- Quick example buttons for testing

### ✅ Robust Error Handling
- Graceful API failures with structured fallbacks
- Provider status indicators
- Simulation mode for demo/testing
- Comprehensive logging and debug information

## 🎯 Usage Examples

### Mission-Critical Analysis Workflow
1. **Enter entity**: "Vuori" (automatically normalized from "VOURI")
2. **Select mode**: "Mission-Critical Analysis"
3. **Choose providers**: OpenAI + Anthropic + Gemini
4. **Run analysis**: 5 query variants × 3 providers = 15 total queries
5. **Review results**: Comprehensive scoring with LinkedIn sharing
6. **Export/Share**: Download report or share on LinkedIn

### Query Types Executed
1. **Recognition**: "How well do you know Vuori in the activewear space?"
2. **Ranking**: "List top 10 activewear companies, include Vuori if applicable"
3. **Sentiment**: "What are the strengths and weaknesses of Vuori?"
4. **Frequency**: "How often does Vuori appear in activewear discussions?"
5. **Comparison**: "Compare Vuori with main activewear competitors"

### Scoring Calculation
```
Frequency: 65% mention rate = 65 points
Ranking: #3 position = 80 points  
Sentiment: Positive overall = 75 points
Recognition: High accuracy = 85 points
Competitive: Strong differentiation = 70 points

Overall = (65×0.3 + 80×0.25 + 75×0.2 + 85×0.15 + 70×0.1) = 74.25 ≈ 74/100
```

## 🐛 Troubleshooting

### Common Issues
- **"Import error"**: Ensure all files are in correct directories per structure above
- **"No API key"**: Set environment variables or add to `.streamlit/secrets.toml`
- **"Analysis failed"**: Check API keys have credits and correct permissions
- **"Timeout"**: Network issues or API rate limits, try fewer providers

### Debug Mode
Enable in sidebar to see:
- Provider availability and key status
- Query execution details
- Error messages with suggestions
- Fallback trigger reasons

### Performance Tips
- **Basic mode**: Use for quick checks (15-30 seconds)
- **Mission-critical mode**: Use for comprehensive analysis (60-90 seconds)
- **Single provider**: Faster results, less comprehensive
- **Multiple providers**: Slower but more reliable and bias-reduced

## 🔒 Security & Best Practices
- API keys never logged or displayed
- All requests use HTTPS with timeouts
- Structured error handling prevents crashes
- No external dependencies beyond official API clients
- Rate limiting respect with automatic fallbacks

## 📈 Advanced Features
- **Trend Analysis**: Track visibility changes over time
- **Competitive Benchmarking**: Compare against industry leaders
- **Sentiment Monitoring**: Track positive/negative perception shifts
- **Share of Voice**: Measure relative mention frequency
- **Export Capabilities**: JSON reports, CSV history, LinkedIn posts

---

**Mission accomplished!** 🎯 Your LLM Visibility Analyzer now provides enterprise-grade brand intelligence across AI knowledge spaces. 