# LLM Visibility API

A FastAPI backend that queries multiple LLM providers (OpenAI, Anthropic, Gemini) with standardized, JSON-structured probes and converts their answers into stable, interpretable 0–100 "visibility" scores.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, and Gemini with graceful fallbacks
- **Deterministic Scoring**: Temperature=0, JSON-mode outputs, optional seeds
- **Parallel Execution**: Async processing for speed
- **Robust JSON Parsing**: Auto-repair for malformed responses
- **Transparent Scoring**: Recognition, detail, context-rank, competitor-recall, and consistency factors
- **CORS Enabled**: Ready for local/SPA usage

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with your API keys:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
ANTHROPIC_MODEL=claude-sonnet-4-20250514

GEMINI_API_KEY=your-gemini-key-here
GEMINI_MODEL=gemini-2.5-flash

# Server Configuration
PORT=5051
```

### 3. Run the Server

```bash
uvicorn main:app --reload --port 5051
```

Or use the built-in runner:

```bash
python main.py
```

## API Usage

### Health Check

```bash
curl http://localhost:5051/health
```

### Analyze Entity Visibility

```bash
curl -X POST "http://localhost:5051/api/visibility" \
  -H "Content-Type: application/json" \
  -d '{
    "entity": "Tesla",
    "category": "automotive",
    "competitors": ["Ford", "GM", "Toyota"],
    "providers": ["openai", "anthropic"]
  }'
```

## Scoring Algorithm

The visibility score (0-100) is calculated from these subscores:

- **Recognition (45%)**: Entity recognition + summary quality + facts count
- **Detail (25%)**: Factual density + competitor information
- **Context (20%)**: Ranking position in industry top lists
- **Competitors (10%)**: Alternative/competitor awareness
- **Consistency**: Multi-probe stability (applied as multiplier)

## Probe Types

1. **Profile Probe**: Entity recognition, summary, facts, category, competitors
2. **Context-Rank Probe**: Top-10 industry ranking with entity position
3. **Alternatives Probe**: List of 10 notable alternatives/substitutes
4. **Consistency Probe**: Re-run ranking for stability measurement

## Response Format

```json
{
  "entity": "Tesla",
  "category": "automotive",
  "overall": 87.5,
  "subscores": {
    "recognition": 0.95,
    "detail": 0.88,
    "context": 0.90,
    "competitors": 0.85,
    "consistency": 0.92
  },
  "providers": [...],
  "notes": [...]
}
```

## Development

- **Port**: 5051 (configurable via PORT env var)
- **Auto-reload**: Enabled in development mode
- **CORS**: Configured for local development (tighten for production)

## Notes

- Review provider ToS and safety policies before production use
- The API gracefully handles provider failures and missing API keys
- Seeds are generated deterministically from entity + category for reproducibility
