# LLM Visibility Tool

A production-ready tool that measures how visible a brand, topic, or person is inside major Large Language Models (ChatGPT/OpenAI, Anthropic Claude, Google Gemini). The tool provides a comprehensive 0-100 visibility score with detailed subscores and cross-provider analysis.

## Features

- **Multi-Provider Analysis**: Query OpenAI, Anthropic, and Google Gemini simultaneously
- **Standardized Scoring**: Consistent 0-100 visibility scores with detailed subscores
- **Real-time Analysis**: Parallel API calls with deterministic results
- **Interactive Frontend**: Modern SPA with gauges, charts, and export functionality
- **Robust Backend**: FastAPI with comprehensive error handling and caching
- **Export Capabilities**: CSV export, PDF printing, and shareable links

## Architecture

### Frontend (SPA)
- Vanilla HTML/CSS/JavaScript
- Responsive design with dark theme
- Interactive score gauges and provider breakdowns
- Real-time analysis with loading states
- Export to CSV, print to PDF, shareable links
- Local storage for session persistence

### Backend (FastAPI)
- **FastAPI** framework with async support
- **Multi-provider adapters** for OpenAI, Anthropic, and Gemini
- **Standardized probes**: Profile, Context Ranking, Alternatives, Consistency
- **Sophisticated scoring** with weighted subscores and aggregation
- **Caching layer** to reduce API costs and improve performance
- **Rate limiting** and error handling

## Scoring Methodology

The tool uses four standardized probes sent to each LLM provider:

1. **Profile Probe**: Entity recognition, summary, facts, category, competitors
2. **Context Ranking Probe**: Top-10 industry ranking with entity position
3. **Alternatives Probe**: List of alternatives/substitutes
4. **Consistency Probe**: Re-ranking to measure result stability

### Subscores (0.0 to 1.0):
- **Recognition** (45%): Based on entity recognition + summary richness + fact count
- **Detail** (25%): Factual density and competitor awareness
- **Context** (20%): Industry ranking position (1st place = 1.0, 10th place = 0.1)
- **Competitors** (10%): Alternative/competitor list completeness
- **Consistency**: Cross-probe variance (applied as 0.85-1.0 multiplier)

### Overall Score (0-100):
```
base_score = (0.45 * recognition + 0.25 * detail + 0.20 * context + 0.10 * competitors)
final_score = base_score * (0.85 + 0.15 * consistency) * 100
```

## Quick Start

### Prerequisites
- Python 3.10+
- API Keys for:
  - OpenAI API
  - Anthropic API  
  - Google Gemini API

### Backend Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # or preferred model

export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_MODEL="claude-3-sonnet-20240229"  # or preferred model

export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-1.5-flash"  # or preferred model

# Optional configuration
export REQUEST_TIMEOUT_SEC="30"
export CACHE_TTL_SEC="21600"  # 6 hours
export CORS_ORIGINS="*"  # Adjust for production
```

3. **Run the server**:
```bash
uvicorn main:app --reload --port 5051
```

### Frontend Setup

1. **Serve the frontend**:
```bash
# Simple HTTP server
python -m http.server 8000

# Or use any static file server
npx serve .
```

2. **Update API configuration** in `index.html`:
```javascript
const API_BASE = 'http://localhost:5051';  // Update for production
```

3. **Access the application**:
Open `http://localhost:8000` in your browser.

## API Reference

### Health Check
```http
GET /health
```

Response:
```json
{
  "ok": true,
  "time": 1703123456.789
}
```

### Visibility Analysis
```http
POST /api/visibility
Content-Type: application/json

{
  "entity": "Tesla",
  "category": "Electric Vehicles",
  "competitors": ["BMW", "Mercedes"],
  "providers": ["openai", "anthropic", "gemini"]
}
```

Response:
```json
{
  "entity": "Tesla",
  "category": "Electric Vehicles",
  "competitors": ["BMW", "Mercedes"],
  "overall": 87.3,
  "subscores": {
    "recognition": 0.92,
    "detail": 0.85,
    "context": 0.90,
    "competitors": 0.78,
    "consistency": 0.89
  },
  "providers": [
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "subscores": { "..." },
      "overall": 89.1,
      "probes": {
        "profile": { "recognized": true, "summary": "...", "facts": [...], "competitors": [...] },
        "context": { "top_list": [...], "rank_of_entity": 2 },
        "alt": { "alternatives": [...] },
        "consistency": { "rank_of_entity": 1 }
      },
      "raw": { "profile": { "text": "..." }, "..." }
    }
  ],
  "notes": []
}
```

## Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest test_api.py -v
```

### Test Categories
- **Unit Tests**: JSON parsing, scoring logic, request validation
- **Integration Tests**: Provider adapters, API endpoints
- **Contract Tests**: Request/response schemas

## Deployment

### Docker Deployment

1. **Build and run**:
```bash
docker build -t llm-visibility-api .
docker run -p 5051:5051 \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  -e GEMINI_API_KEY="your-key" \
  llm-visibility-api
```

### Production Considerations

1. **Environment Variables**:
```bash
# Required API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Optional configuration
REQUEST_TIMEOUT_SEC=25
CACHE_TTL_SEC=21600
CORS_ORIGINS=https://yourdomain.com
PORT=5051
```

2. **Security**:
- Never expose API keys to frontend
- Use environment-specific CORS origins
- Implement proper rate limiting for production
- Add authentication if needed

3. **Monitoring**:
- Check `/health` endpoint
- Monitor API usage and costs
- Log performance metrics
- Set up alerting for failures

4. **Frontend Deployment**:
- Deploy to static hosting (Vercel, Netlify, S3+CloudFront)
- Update `API_BASE` to production backend URL
- Configure proper CORS on backend

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (required) |
| `ANTHROPIC_MODEL` | `claude-3-sonnet-20240229` | Claude model to use |
| `GEMINI_API_KEY` | - | Google Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `REQUEST_TIMEOUT_SEC` | `30` | Request timeout in seconds |
| `CACHE_TTL_SEC` | `21600` | Cache TTL (6 hours) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `PORT` | `5051` | Server port |

### Scoring Weights

The scoring weights are configurable in `scoring.py`:

```python
W_RECOGNITION = 0.45  # Recognition subscore weight
W_DETAIL = 0.25       # Detail subscore weight  
W_CONTEXT = 0.20      # Context ranking weight
W_COMPETITORS = 0.10  # Competitors awareness weight
```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Verify all three API keys are set correctly
   - Check API key permissions and quotas
   - Ensure models are available in your region

2. **Timeout Errors**:
   - Increase `REQUEST_TIMEOUT_SEC`
   - Check network connectivity
   - Verify API endpoints are accessible

3. **Rate Limiting**:
   - Implement exponential backoff
   - Check API usage limits
   - Consider upgrading API plans

4. **CORS Issues**:
   - Set proper `CORS_ORIGINS` for your frontend domain
   - Verify frontend API_BASE configuration

5. **JSON Parsing Errors**:
   - Models occasionally return malformed JSON
   - The tool includes automatic JSON repair
   - Check provider-specific response formats

### Performance Optimization

1. **Caching**: Results are cached for 6 hours by default
2. **Parallel Execution**: All providers are queried simultaneously
3. **Request Batching**: Consider batching multiple entities
4. **Model Selection**: Faster models (e.g., `-mini` versions) for development

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git