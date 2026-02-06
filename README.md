# COMP713 Movie Recommendation System

A two-stage movie recommendation system combining offline pre-computation with online LLM-powered conversation. Uses emotional inference from casual chat to recommend movies without explicit preference questioning.

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

# Interactive mode with LLM-powered preference extraction
python interactive_recommender.py

# Demo mode - see 10 different recommendation scenarios
python demo_recommendations.py

# Run unit tests (61 tests)
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
├── interactive_recommender.py       # LLM-powered interactive recommender
├── demo_recommendations.py          # Demo scenarios showing different recommendations
├── test_recommendations.py          # Unit tests (61 tests)
├── test_conversation_flow.py        # Conversation flow integration test
├── process_recommendations.py       # Offline pipeline (3-phase incremental)
├── generate_user_ratings.py         # Synthetic user data generator
├── config.example.json              # LLM API configuration template
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

## How Semantic Recommendations Work

1. **Load `index.json`** to discover all available recommendation files and their `match_keywords`
2. **Direct matching**: If `--segment`, `--mood`, `--genre`, or `--era` are provided, look up the exact file by type+tag. Comma-separated values load multiple files.
3. **Keyword matching**: If `--query` is provided, tokenize it and score each file by how many tokens match its `match_keywords`; pick the top matches
4. **Cold start fallback**: If no features are provided at all, load `fallback_popular.json`
5. **Multi-source merge**: When multiple files match, load all of them and deduplicate results by `item_id`
6. **Explainable output**: Every recommendation includes `why_recommended` text and the source category

## Interactive Recommender (LLM-Powered)

The `interactive_recommender.py` uses LangChain with an LLM to:

1. **Engage in casual conversation** - Has natural, flowing dialogue about hobbies, weekend plans, friends, food, music, etc.
2. **Infer preferences emotionally** - Detects segment, mood, genre, era from tone, vocabulary, and context (not explicit keywords)
3. **Generate diverse follow-ups** - Explores different life areas each round, never repeats topics
4. **Normalize LLM output** - Handles combined values like `Action/Sci-Fi` by splitting and loading both genre files
5. **Handle cold start gracefully** - Falls back to popular movies when insufficient information gathered

### Key Features

- **Varied greetings** - Random topic seeds + high-temperature conversation LLM ensure every session starts differently
- **Dual-temperature LLM** - Low temperature (0.3) for precise preference extraction, high temperature (0.8) for natural conversation
- **Multi-value support** - When LLM detects multiple genres (e.g. `Action/Sci-Fi`), both are used for recommendations
- **Conversation memory** - Full history of both user messages and AI responses tracked for context
- **Emotional inference** - Detects "gamer" from "grinding ranked matches", "tired" from "just want to chill"
- **Topic diversity** - Tracks discussed topics to avoid repetition
- **File-only logging** - Debug logs written to `logs/` directory, no console spam
- **Configurable rounds** - Default 3 rounds minimum via `config.json`

### Configuration

Copy `config.example.json` to `config.json` and edit:

```json
{
  "llm": {
    "api_base": "https://your-api-endpoint.com/v1",
    "api_key": "your-api-key",
    "model": "your-model-name",
    "temperature": 0.3
  },
  "recommendation": {
    "min_rounds": 3,
    "max_rounds": 10,
    "default_top_n": 5
  }
}
```

### Usage

```bash
# Test API connection
python interactive_recommender.py --test-config

# Run interactive mode
python interactive_recommender.py

# Demo mode with predefined preferences
python interactive_recommender.py --demo
```

## Two Approaches

| | **Prime** (Collaborative Filtering) | **Semantic** (Pre-computed) |
|---|---|---|
| Algorithm | Pearson correlation between users | Keyword/tag matching to pre-generated files |
| Data source | `data/ratings.json` (small set) | `shared_recommendations/` (41 files) |
| Cold start | Cannot handle (needs rating history) | Handled via fallback files |
| Explainability | Returns movie titles only | Returns titles + `why_recommended` |
| Dependencies | numpy | langchain, langchain-openai |

## Test Results

```
61 passed in 0.86s
```

| Scenario | Matched Files | Candidates | Sample Results |
|----------|--------------|------------|----------------|
| Cold Start (no features) | 1 | 20 | Popular movies fallback |
| Gamer + Action + Exciting | 3 | 54 | Segment + mood + genre merged |
| Student + Thriller + Exciting | 3 | 51 | Multi-source deduplication |
| Parent + Comedy + Relaxing | 3 | 53 | Cross-category matching |
| 90s Nostalgia | 1 | 8 | Era-specific results |
| "deep philosophical" (query) | 3 | 42 | Keyword matching across files |
| Horror + Intense | 2 | 40 | Mood + genre combination |
| "sci-fi adventure" (query) | 3 | 42 | Free-text semantic search |
| Romantic + Emotional | 2 | 40 | Mood-driven selection |
| Classic Era | 1 | 20 | Single era filter |
