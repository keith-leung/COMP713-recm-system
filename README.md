# COMP713 Movie Recommendation System

A two-stage movie recommendation system combining offline pre-computation with online LLM-powered freeform recommendations. Uses natural conversation to understand user vibe and recommends movies without explicit preference questioning or structured categorization.

## What's New (v2.0)

- **Freeform Recommendations**: Removed hardcoded categorization - LLM generates natural recommendations based on conversation
- **Smart Fallback**: Automatic NanoGPT fallback when local Ollama is unavailable
- **Speed Optimization**: 35% faster - reduced from 2 LLM calls/round to 1
- **External Prompts**: All prompts configurable via `prompts.json` - no hardcoded prompts
- **Readline Support**: Arrow keys and terminal editing now work properly
- **62 Passing Tests**: Including freeform recommendation and conversation flow tests

## Architecture

```
OFFLINE STAGE                             ONLINE STAGE
─────────────────────────────────          ─────────────────────────────
  data/movies_*.json (5 chunks)               main.py --approach semantic
  data/user_ratings_*.json (14 chunks)           |
          |                                      v
          v                                 Load index.json
  process_recommendations.py                     |
   Phase 1: Index movies by genre/mood/era       v
   Phase 2: Aggregate user ratings/segments  Match user features to files
   Phase 3: Generate recommendation files        |
          |                                      v
          v                                 Load matched .json files
  _state/movies_index.json (intermediate)        |
  _state/user_stats.json   (intermediate)        v
          |                                 Return ranked recommendations
          v                                 with explanations
  shared_recommendations/
   41 files: 9 segments, 5 moods,
   18 genres, 7 eras, 2 fallbacks,
   1 index.json

NEW: Freeform Recommendations via LLM
  interactive_recommender.py
   - Natural conversation (no structured extraction)
   - LLM generates movie recommendations directly
   - Smart fallback: Local Ollama → NanoGPT API
   - prompts.json external configuration
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy config template and add your API key
cp config.example.json config.json

# Semantic approach (default) - cold start, no features known
python main.py

# With known user features
python main.py --segment gamer --top 3
python main.py --mood thoughtful --genre Drama
python main.py --era 90s --mood exciting --segment student --top 5

# Multi-value parameters (comma-separated)
python main.py --genre "Action,Sci-Fi" --mood exciting

# Free-text semantic query
python main.py --query "something deep and philosophical"

# Original collaborative filtering (requires numpy)
python main.py --user "David Smith" --approach prime

# Interactive mode with LLM-powered freeform recommendations
python interactive_recommender.py

# Demo mode - see 10 different recommendation scenarios
python demo_recommendations.py

# Run unit tests (62 tests)
python -m pytest test_recommendations.py -v

# Run conversation flow test (requires LLM API)
python test_conversation_flow.py
```

## CLI Parameters

| Flag | Description | Examples |
|------|-------------|---------|
| `--approach` | Algorithm: `semantic` (default) or `prime` | `--approach prime` |
| `--user` | Username (for prime; label-only for semantic) | `--user "David Smith"` |
| `--segment` | User demographic segment | `gamer`, `student`, `parent`, `boomer`, `millennial`, `gen_z`, `female`, `male` |
| `--mood` | Desired mood | `exciting`, `relaxing`, `intense`, `thoughtful`, `emotional` |
| `--genre` | Preferred genre (comma-separated for multiple) | `Action`, `"Action,Sci-Fi"`, `Comedy` |
| `--era` | Preferred era | `Classic`, `60s-70s`, `80s`, `80s-90s`, `90s`, `2000s`, `Modern` |
| `--query` | Free-text for keyword matching | `"deep philosophical"`, `"fun action"` |
| `--top` | Number of results (default 5) | `--top 10` |

## Project Structure

```
.
├── main.py                          # Entry point - CLI with both approaches
├── interactive_recommender.py       # LLM-powered interactive recommender (NEW: Freeform)
├── demo_recommendations.py          # Demo scenarios showing different recommendations
├── test_recommendations.py          # Unit tests (62 tests)
├── test_conversation_flow.py        # Conversation flow integration test
├── process_recommendations.py       # Offline pipeline (3-phase incremental)
├── generate_user_ratings.py         # Synthetic user data generator
├── config.json                      # LLM API configuration (gitignored)
├── config.example.json              # LLM API configuration template
├── prompts.json                     # External prompt configuration (NEW)
├── requirements.txt                 # Python dependencies
├── prime/                           # Collaborative filtering module
│   ├── __init__.py
│   ├── compute_scores.py            #   Pearson correlation
│   ├── collaborative_filtering.py   #   Find similar users
│   └── movie_recommender_prime.py   #   Generate recommendations
├── data/                            # Input datasets
│   ├── ratings.json                 #   Small test set (8 users, 6 movies) for prime
│   ├── movies_001.json ~ 005.json   #   1,000 movies in chunks
│   └── user_ratings_001.json ~ 014  #   4,308 users in chunks
├── shared_recommendations/          # Pre-computed recommendation files
│   ├── index.json                   #   Master registry (load this first)
│   ├── segment_*.json (9)           #   By demographic (gamer, student, etc.)
│   ├── mood_*.json (5)              #   By mood (exciting, thoughtful, etc.)
│   ├── genre_*.json (18)            #   By genre (Action, Drama, etc.)
│   ├── era_*.json (7)               #   By era (Classic, 90s, Modern, etc.)
│   └── fallback_*.json (2)          #   Popular & acclaimed (cold start)
└── skills/
    └── movie-recommendation-generator/
        └── SKILL.md                 #   Claude Code Skill definition
```

## Configuration

### config.json

Copy `config.example.json` to `config.json`:

```json
{
  "llm": {
    "provider": "openai_compatible",
    "api_base": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "qwen-turbo",
    "temperature": 0.3,
    "fallback": {
      "enabled": true,
      "api_base": "https://nano-gpt.com/api/v1",
      "api_key": "your-api-key-here",
      "model": "gpt-4o-mini"
    },
    "prompts_file": "prompts.json"
  },
  "recommendation": {
    "min_rounds": 3,
    "max_rounds": 10,
    "fallback_on_insufficient_info": true,
    "default_top_n": 5
  }
}
```

**NEW: Smart Fallback**
- Primary: Local Ollama at `localhost:11434`
- Automatic fallback to NanoGPT when primary fails
- After first timeout, uses fallback directly for all subsequent calls
- Fast 0.5s timeout for local LLM, 30s timeout for fallback

### prompts.json (NEW)

All LLM prompts are now external and configurable:

```json
{
  "conversation_chain": {
    "system_prompt": "You are a friendly assistant...",
    "topic_seeds": ["what they do for fun on weekends", ...]
  },
  "recommendation_chain": {
    "system_prompt": "You are a movie recommendation expert..."
  }
}
```

**No hardcoded prompts in code** - everything is in `prompts.json`.

## How Semantic Recommendations Work

1. **Load `index.json`** to discover all available recommendation files and their `match_keywords`
2. **Direct matching**: If `--segment`, `--mood`, `--genre`, or `--era` are provided, look up the exact file by type+tag. Comma-separated values load multiple files.
3. **Keyword matching**: If `--query` is provided, tokenize it and score each file by how many tokens match its `match_keywords`; pick the top matches
4. **Cold start fallback**: If no features are provided at all, load `fallback_popular.json`
5. **Multi-source merge**: When multiple files match, load all of them and deduplicate results by `item_id`
6. **Explainable output**: Every recommendation includes `why_recommended` text and the source category

## Interactive Recommender (Freeform LLM-Powered)

The `interactive_recommender.py` uses LangChain with an LLM for:

1. **Natural conversation** - Has flowing dialogue about hobbies, weekend plans, friends, food, music, etc.
2. **Vibe understanding** - LLM naturally detects user preferences from tone and context
3. **Freeform recommendations** - LLM generates personalized movie suggestions based on the entire conversation
4. **Smart fallback** - Automatically switches to NanoGPT when local Ollama is unavailable

### Key Features (v2.0)

- **Freeform recommendations** - No structured extraction, LLM generates natural recommendations
- **Single LLM call per round** - 35% faster than previous version (2 calls → 1 call)
- **Smart fallback** - Remembers primary failure, uses fallback directly afterward
- **External prompts** - All prompts configurable via `prompts.json`
- **Varied conversation** - Random topic seeds ensure every session starts differently
- **Conversation memory** - Full history of both user messages and AI responses tracked for context
- **Readline support** - Arrow keys, Home/End, Ctrl+A/E for terminal editing
- **File-only logging** - Debug logs written to `logs/` directory, no console spam

### Usage

```bash
# Run interactive mode
python interactive_recommender.py

# The conversation will:
# 1. Start with a friendly greeting
# 2. Chat naturally about hobbies, interests, lifestyle
# 3. Generate freeform movie recommendations when ready
```

## Two Approaches

| | **Prime** (Collaborative Filtering) | **Semantic** (Pre-computed) | **Interactive** (Freeform LLM) |
|---|---|---|---|
| Algorithm | Pearson correlation between users | Keyword/tag matching to pre-generated files | LLM natural conversation |
| Data source | `data/ratings.json` (small set) | `shared_recommendations/` (41 files) | Full conversation context |
| Cold start | Cannot handle (needs rating history) | Handled via fallback files | Handled naturally |
| Explainability | Returns movie titles only | Returns titles + `why_recommended` | Natural explanations |
| Dependencies | numpy | langchain, langchain-openai | langchain, langchain-openai |

## Test Results

```
62 passed in 1.04s
```

| Test Category | Tests | Status |
|--------------|-------|--------|
| Index & File Loading | 10 | ✅ PASS |
| Keyword Matching | 4 | ✅ PASS |
| Cold Start | 2 | ✅ PASS |
| Single Feature Matching | 16 | ✅ PASS |
| Multi-Feature Matching | 3 | ✅ PASS |
| Free-Text Query | 6 | ✅ PASS |
| Prime Approach | 5 | ✅ PASS |
| Recommendation Quality | 5 | ✅ PASS |
| Real-World Scenarios | 6 | ✅ PASS |
| LLM Parser (NEW) | 2 | ✅ PASS |
| Conversation Flow (NEW) | 2 | ✅ PASS |

## Performance Comparison

| Metric | Previous Version | Current Version (v2.0) | Improvement |
|--------|-----------------|----------------------|-------------|
| LLM calls per conversation round | 2 (extraction + response) | 1 (conversation only) | 50% reduction |
| Average response time | ~6s per round | ~3.9s per round | 35% faster |
| Structured extraction | Required (segment/mood/genre/era) | Removed (freeform) | Simplified |
| Hardcoded prompts | In code | External (prompts.json) | Configurable |
| Terminal editing | Broken (escape sequences) | Working (readline) | Fixed |

## Agile Development Improvements

This repo demonstrates agile development practices:

1. **Test-Driven Development**: 62 automated tests ensure reliability during refactoring
2. **External Configuration**: Prompts and settings externalized for easy iteration
3. **Smart Fallback**: Graceful degradation when services are unavailable
4. **Continuous Refactoring**: Moved from structured extraction to freeform recommendations
5. **Performance Optimization**: 35% speed improvement through architectural changes
6. **User Experience**: Readline support for better terminal interaction
