# Movie Recommendation System: Final Report

**Dawen Liang**
University of the Potomac
COMP713: Advanced Artificial Intelligence
Dr. Jimmy Tsai
February 2026

---

## Abstract

This report presents the design, implementation, and evaluation of a two-stage movie recommendation system that separates offline data processing from online LLM-powered user interaction. The system addresses three fundamental challenges in recommendation engines: LLM context window limitations, the cold start problem, and the explainability requirement. By pre-computing 41 recommendation files across 4 dimensions (user segment, mood, genre, era) and using an LLM for real-time emotional inference through casual conversation, the system delivers personalized, explainable movie recommendations without requiring explicit user preference questionnaires. The implementation includes 61 unit tests (100% passing), a conversational integration test, and supports both semantic matching and traditional collaborative filtering approaches.

---

## 1. Introduction

### 1.1 Motivation

Traditional recommendation systems face a tension between data richness and computational efficiency. Collaborative filtering approaches like ALS (Alternating Least Squares) require large rating matrices and cannot handle new users. Content-based approaches require explicit feature engineering. Modern LLM-based systems offer natural language understanding but are constrained by context window limits -- loading thousands of movies and user ratings directly into a prompt is impractical.

This project proposes a hybrid approach: use offline computation to distill large datasets into compact, semantically-tagged recommendation files, then use an LLM online to match users to these files through natural conversation.

### 1.2 Key Innovation

Unlike systems that use cosine similarity, matrix factorization, or direct LLM prompting, this system:

1. **Separates concerns**: Heavy data processing happens offline; the LLM only handles conversation and semantic matching
2. **Uses emotional inference**: The LLM detects user preferences from casual conversation tone, not explicit keyword matching
3. **Provides explainability**: Every recommendation includes a `why_recommended` field with transparent reasoning

---

## 2. System Architecture

### 2.1 Two-Stage Pipeline

```mermaid
flowchart LR
    subgraph OFFLINE["Offline Stage"]
        M["movies_*.json<br/>(5 chunks, 1000 movies)"] --> P["process_recommendations.py"]
        U["user_ratings_*.json<br/>(14 chunks, 4308 users)"] --> P
        P --> S1["Phase 1: Index Movies"]
        S1 --> S2["Phase 2: Aggregate Users"]
        S2 --> S3["Phase 3: Generate Files"]
        S3 --> R["shared_recommendations/<br/>41 JSON files"]
    end

    subgraph ONLINE["Online Stage"]
        USER["User"] --> LLM["LLM Conversation<br/>(qwen-turbo)"]
        LLM --> EXTRACT["Preference Extraction<br/>(segment, mood, genre, era)"]
        EXTRACT --> NORM["Value Normalization<br/>(split Action/Sci-Fi -> 2 files)"]
        NORM --> MATCH["Semantic Matching"]
        R --> IDX["index.json"]
        IDX --> MATCH
        MATCH --> LOAD["Load Matched Files"]
        LOAD --> DEDUP["Merge & Deduplicate"]
        DEDUP --> REC["Ranked Recommendations<br/>with Explanations"]
    end
```

### 2.2 Data Flow Summary

| Stage | Input | Processing | Output |
|-------|-------|-----------|--------|
| Offline Phase 1 | 5 movie chunk files | Index by genre, mood, era | `_state/movies_index.json` |
| Offline Phase 2 | 14 user rating chunks | Aggregate by segment, track high ratings | `_state/user_stats.json` |
| Offline Phase 3 | Both state files | Generate ranked recommendations | 41 JSON files + `index.json` |
| Online | User conversation | Emotional inference + semantic match | Top-N recommendations |

### 2.3 Recommendation File Distribution

```mermaid
pie title Recommendation Files by Category (41 total)
    "Genre (18)" : 18
    "Segment (9)" : 9
    "Era (7)" : 7
    "Mood (5)" : 5
    "Fallback (2)" : 2
```

---

## 3. Offline Processing Pipeline

### 3.1 Incremental Processing Design

The critical design principle is **never load all data at once**. Each chunk is processed independently, with intermediate state accumulated in `_state/` files between chunks.

```mermaid
flowchart TD
    A["Load movies_001.json"] --> B["Update _state/movies_index.json"]
    B --> C["Load movies_002.json"]
    C --> D["Update _state/movies_index.json"]
    D --> E["... repeat for all chunks"]
    E --> F["Final movies_index.json<br/>contains all 1000 movies"]

    G["Load user_ratings_001.json"] --> H["Update _state/user_stats.json"]
    H --> I["Load user_ratings_002.json"]
    I --> J["Update _state/user_stats.json"]
    J --> K["... repeat for all chunks"]
    K --> L["Final user_stats.json<br/>contains all 4308 users"]

    F --> M["Generate 41 recommendation files"]
    L --> M
```

This approach guarantees the system can handle arbitrarily large datasets without context overflow, as only one chunk (approximately 200 items) is in memory at any time.

### 3.2 Multi-Dimensional Tagging

Each movie is tagged across three dimensions, and users are tagged by demographic segment:

```mermaid
flowchart TD
    MOVIE["Movie: The Matrix (1999)"] --> G["Genre: Action, Sci-Fi"]
    MOVIE --> MO["Mood: Revolutionary, Action-packed"]
    MOVIE --> E["Era: 90s"]

    USER["User: id0042"] --> S["Segment: gamer, millennial"]
    USER --> R["Ratings: M014=4.8, M023=3.2, ..."]

    G --> GF["genre_action.json<br/>genre_sci_fi.json"]
    MO --> MF["mood_exciting.json"]
    E --> EF["era_90s.json"]
    S --> SF["segment_gamer.json<br/>segment_millennial.json"]
```

### 3.3 Recommendation Scoring

For segment-based recommendations, movies are ranked by how many users in that segment rated them 4+ stars. For mood/genre/era-based recommendations, movies are ranked by high-rating count, then average rating. This produces transparent scores that can be directly explained to users.

| Dimension | Ranking Method | Example Explanation |
|-----------|---------------|---------------------|
| Segment | % of segment users rating 4+ | "1% of gamer users rated this 4+ stars" |
| Mood | High-rating count, then avg rating | "Perfect for a relaxing mood. Rated 3.8/5.0" |
| Genre | High-rating count, then avg rating | "Top-rated Action movie. Average rating: 3.7/5.0" |
| Era | High-rating count, then avg rating | "Top-rated 90s movie. Average rating: 4.1/5.0" |

---

## 4. Online Interactive System

### 4.1 Conversation Architecture

The interactive recommender uses LangChain with dual LLM configurations:

```mermaid
flowchart TD
    subgraph LLM_CONFIG["Dual LLM Configuration"]
        CONV["Conversation LLM<br/>temperature=0.8<br/>Creative, varied responses"]
        EXTRACT["Extraction LLM<br/>temperature=0.3<br/>Precise, structured output"]
    end

    USER["User Input"] --> CONV
    CONV --> RESP["Conversational Response<br/>(displayed to user)"]
    USER --> EXTRACT
    EXTRACT --> PREFS["Structured Preferences<br/>{segment, mood, genre, era}"]
    PREFS --> NORM["Normalization Layer<br/>Action/Sci-Fi -> [Action, Sci-Fi]"]
    NORM --> MERGE["Accumulated Profile"]
```

The separation of conversation generation (high temperature for variety) from preference extraction (low temperature for precision) is a key architectural decision. Without this, the system either produces repetitive greetings or unreliable preference extraction.

### 4.2 Conversation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Conversation LLM
    participant E as Extraction LLM
    participant S as Semantic Matcher

    Note over C: Random topic seed selected
    C->>U: "What's a hobby you've always wanted to try?"
    U->>C: "I've been playing a lot of video games lately"
    U->>E: (same input, parallel)
    E-->>E: Extract: segment=gamer, mood=null
    C->>U: "That's cool! What's the most immersive game recently?"
    U->>C: "Competitive stuff like Valorant, just hit Diamond!"
    U->>E: (same input, parallel)
    E-->>E: Extract: mood=exciting, genre=Action
    C->>U: "Nice! What do you do to unwind after gaming?"
    U->>C: "Chill with some sci-fi shows"
    U->>E: (same input, parallel)
    E-->>E: Extract: genre=Action,Sci-Fi, era=Modern

    Note over S: Round >= min_rounds, preferences detected
    S->>S: Load segment_gamer.json
    S->>S: Load mood_exciting.json
    S->>S: Load genre_action.json
    S->>S: Load genre_sci_fi.json
    S->>S: Load era_modern.json
    S->>S: Merge & deduplicate
    S->>U: Top 5 recommendations with explanations
```

### 4.3 Multi-Value Normalization

A significant challenge with smaller LLMs (qwen-turbo) is that they often return combined values like `Action/Sci-Fi` instead of a single valid genre. The normalization layer handles this:

```mermaid
flowchart LR
    RAW["LLM Output:<br/>genre='Action/Sci-Fi'"] --> SPLIT["Split by / , and or"]
    SPLIT --> V1["'Action'"]
    SPLIT --> V2["'Sci-Fi'"]
    V1 --> CHECK1{"Valid genre?"}
    V2 --> CHECK2{"Valid genre?"}
    CHECK1 -->|Yes| OUT["Output: 'Action,Sci-Fi'"]
    CHECK2 -->|Yes| OUT
    OUT --> LOAD1["Load genre_action.json<br/>(20 movies)"]
    OUT --> LOAD2["Load genre_sci_fi.json<br/>(20 movies)"]
    LOAD1 --> MERGE["Merge: 40 candidates<br/>(deduplicated)"]
    LOAD2 --> MERGE
```

This normalization applies to all four preference dimensions (segment, mood, genre, era), not just genre. Each dimension is validated against the known valid options from the index.

### 4.4 Conversation History Management

A critical bug discovered during testing was that the LLM's own responses were not tracked in conversation history. This caused the LLM to repeat questions, having no memory of what it already asked.

**Before fix** (broken):
```
conversation_history = [
    "You: I like video games",
    "AI: User seems to be a gamer (internal reasoning)",
    "You: Yeah mostly competitive stuff"
]
```

**After fix** (working):
```
conversation_history = [
    "Assistant: What's a hobby you've always wanted to try?",
    "You: I like video games",
    "Assistant: That's cool! What's the most immersive game recently?",
    "You: Yeah mostly competitive stuff"
]
```

The fix ensures the LLM sees its own previous messages, preventing repetition and enabling contextual follow-ups.

---

## 5. Semantic Matching Engine

### 5.1 Matching Strategy

```mermaid
flowchart TD
    INPUT["User Preferences:<br/>segment=gamer<br/>mood=exciting<br/>genre=Action,Sci-Fi"] --> SPLIT["Split comma-separated values"]

    SPLIT --> S1["segment: gamer"]
    SPLIT --> S2["mood: exciting"]
    SPLIT --> S3a["genre: Action"]
    SPLIT --> S3b["genre: Sci-Fi"]

    S1 --> LOOKUP["Index Lookup"]
    S2 --> LOOKUP
    S3a --> LOOKUP
    S3b --> LOOKUP

    LOOKUP --> F1["segment_gamer.json (20 movies)"]
    LOOKUP --> F2["mood_exciting.json (20 movies)"]
    LOOKUP --> F3["genre_action.json (20 movies)"]
    LOOKUP --> F4["genre_sci_fi.json (20 movies)"]

    F1 --> MERGE["Merge & Deduplicate by item_id"]
    F2 --> MERGE
    F3 --> MERGE
    F4 --> MERGE

    MERGE --> RESULT["68 unique candidates<br/>ranked by source order"]

    NONE["No preferences detected"] --> FALLBACK["fallback_popular.json<br/>(20 movies)"]
```

### 5.2 Multi-Source Deduplication

When multiple recommendation files are loaded, the same movie may appear in multiple files. The system deduplicates by `item_id`, keeping the first occurrence (which preserves the source priority ordering).

| Scenario | Files Loaded | Raw Movies | After Dedup |
|----------|-------------|------------|-------------|
| Gamer + Action + Exciting | 3 | 60 | 54 |
| Gamer + Action + Sci-Fi + Exciting | 4 | 80 | 68 |
| Cold Start | 1 | 20 | 20 |
| 90s Nostalgia | 1 | 8 | 8 |

---

## 6. Collaborative Filtering (Prime Approach)

The system preserves the original collaborative filtering module as an alternative approach.

### 6.1 Pearson Correlation

The `prime/` module computes user similarity using the Pearson correlation coefficient:

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

Where $x_i$ and $y_i$ are ratings from two users on commonly-rated movies. The system finds users most similar to the input user, then recommends movies those similar users rated highly but the input user hasn't seen.

### 6.2 Comparison of Approaches

```mermaid
flowchart LR
    subgraph PRIME["Prime (Collaborative Filtering)"]
        direction TB
        P1["Input: User name"]
        P2["Find similar users<br/>(Pearson correlation)"]
        P3["Recommend unseen movies<br/>that similar users liked"]
        P1 --> P2 --> P3
    end

    subgraph SEMANTIC["Semantic (Pre-computed)"]
        direction TB
        S1["Input: Preferences<br/>(segment, mood, genre, era)"]
        S2["Match to pre-computed files<br/>(index.json lookup)"]
        S3["Return ranked movies<br/>with explanations"]
        S1 --> S2 --> S3
    end
```

| Characteristic | Prime | Semantic |
|---------------|-------|----------|
| Algorithm | Pearson correlation | Tag matching to pre-computed files |
| Cold start handling | Cannot (needs rating history) | Fallback files |
| Explainability | Movie titles only | Titles + `why_recommended` |
| Dataset | 8 users, 6 movies | 4,308 users, 1,000 movies |
| Dependencies | numpy | langchain, langchain-openai |
| Scalability | O(n*m) per query | O(1) file lookup |

---

## 7. Testing and Validation

### 7.1 Test Suite Overview

```mermaid
pie title Test Distribution (61 tests)
    "Index & File Loading (10)" : 10
    "Keyword Matching (4)" : 4
    "Cold Start (2)" : 2
    "Single Feature Matching (16)" : 16
    "Multi-Feature Matching (3)" : 3
    "Free-Text Query (6)" : 6
    "Prime Approach (5)" : 5
    "Recommendation Quality (5)" : 5
    "Real-World Scenarios (6)" : 6
    "LLM Parser (1)" : 1
    "Conversation Flow (3)" : 3
```

### 7.2 Unit Test Coverage

| Test Category | Tests | What's Verified |
|--------------|-------|-----------------|
| Index Loading | 4 | File exists, required fields present, counts match, all types present |
| File Loading | 6 | Each file type loads correctly, recommendation structure valid |
| Keyword Matching | 4 | Single keyword, multiple keywords, no matches, fallback exclusion |
| Cold Start | 2 | Returns results, uses fallback file |
| Single Feature | 16 | 3 segments, 5 moods, 6 genres, 5 eras each return correct files |
| Multi-Feature | 3 | Segment+mood, all features, mood+genre combinations |
| Free-Text Query | 6 | 5 query types return results, typo handling |
| Prime Approach | 5 | 3 known users, unknown user error, unseen movie filter |
| Quality Checks | 5 | Valid years, genres present, moods present, explanations present, ranked |
| Scenarios | 6 | Gamer+action, student+thriller, parent+comedy, cold start, 90s, philosophical |
| LLM Parser | 1 | Preference extraction from explicit input |

### 7.3 Integration Test

The conversation flow test (`test_conversation_flow.py`) validates the LLM interaction end-to-end:

1. Sends 3 simulated user messages through the LLM
2. Verifies all 3 generated responses are **unique** (no repetition)
3. Verifies no response contains **explicit movie questions**
4. Verifies preferences were **successfully extracted** from the conversation

### 7.4 Test Results

```
============================== 61 passed in 0.86s ==============================
```

All 61 unit tests pass consistently. The conversation flow test passes with the live LLM API.

---

## 8. Bugs Discovered and Fixed

### 8.1 Bug Summary

```mermaid
flowchart TD
    B1["Bug 1: Repeating Questions"] --> F1["Fix: Track AI responses<br/>in conversation_history"]
    B2["Bug 2: Combined Genre Values<br/>(Action/Sci-Fi)"] --> F2["Fix: Normalization layer<br/>splits and loads both files"]
    B3["Bug 3: Hardcoded min_rounds"] --> F3["Fix: Read from config.json"]
    B4["Bug 4: Identical Greetings"] --> F4["Fix: Separate conversation LLM<br/>+ random topic seeds"]
    B5["Bug 5: API Key Exposed"] --> F5["Fix: .gitignore + config.example.json"]

    style B1 fill:#f99
    style B2 fill:#f99
    style B3 fill:#f99
    style B4 fill:#f99
    style B5 fill:#f99
    style F1 fill:#9f9
    style F2 fill:#9f9
    style F3 fill:#9f9
    style F4 fill:#9f9
    style F5 fill:#9f9
```

### 8.2 Detailed Bug Analysis

#### Bug 1: Repeating Questions

**Symptom**: The system asked the same question after every user input.

**Root Cause**: The conversation history only stored user messages and the LLM's internal reasoning. The LLM's actual spoken responses (greeting + follow-ups) were never recorded. Without seeing what it already said, the LLM regenerated the same question.

**Fix**: Store each AI response in `conversation_history` as `\nAssistant: {response}` immediately after generation.

**Impact**: Critical -- the core conversation loop was non-functional without this fix.

#### Bug 2: Combined Genre Values

**Symptom**: LLM returned `Action/Sci-Fi` but only `genre_action.json` was loaded (or neither, depending on exact matching).

**Root Cause**: The extraction LLM (qwen-turbo) frequently returns combined values like `Action/Sci-Fi` instead of choosing one. The matching code expected exact single values.

**Fix**: Added `_normalize_values()` method that splits combined values by `/`, `,`, `and`, `or`, validates each part against known options, and returns all valid matches as a comma-separated string. Updated `get_recommendations_semantic()` to split comma-separated values and load a file for each.

**Impact**: Medium -- recommendations were less diverse without multi-genre loading (56 vs 68 candidates).

#### Bug 3: Hardcoded min_rounds

**Symptom**: The "want to see recommendations?" prompt always appeared after round 3, ignoring the `min_rounds` config setting.

**Root Cause**: The condition `self.round_num >= 3` was hardcoded instead of reading `self.config.min_rounds`.

**Fix**: Added `min_rounds` property to Config class, replaced hardcoded `3` with `self.config.min_rounds`.

**Impact**: Low -- config was not respected, but the default value happened to be correct.

#### Bug 4: Identical Greetings

**Symptom**: Every session started with the exact same greeting: "Hey there! What's something you've been really into lately that makes you feel excited?"

**Root Cause**: The conversation LLM used the same low temperature (0.3) as the extraction LLM. Combined with identical empty inputs for the initial greeting, the deterministic low-temperature output was always the same.

**Fix**: Created a separate `conversation_llm` with temperature=0.8 for varied responses. Added 12 random topic seeds that inject variety into the prompt even when conversation history is empty.

**Impact**: Medium -- while functional, identical greetings gave the impression of a hardcoded system rather than an AI.

#### Bug 5: API Key Exposure

**Symptom**: `config.json` containing a real API key was not in `.gitignore`.

**Fix**: Added `config.json` and `_state/` to `.gitignore`. Created `config.example.json` as a template without real credentials.

**Impact**: Security -- API key could be accidentally committed to version control.

---

## 9. Implementation Statistics

### 9.1 System Metrics

| Metric | Value |
|--------|-------|
| Total movies indexed | 1,000 |
| Total users analyzed | 4,308 |
| Recommendation files generated | 41 |
| User segments | 9 |
| Genre categories | 18 |
| Mood categories | 5 |
| Era categories | 7 |
| Unit tests | 61 (100% passing) |
| Integration tests | 1 (LLM conversation flow) |

### 9.2 Code Metrics

| File | Lines | Purpose |
|------|-------|---------|
| `interactive_recommender.py` | 753 | LLM-powered interactive system |
| `process_recommendations.py` | 721 | Offline processing pipeline |
| `test_recommendations.py` | ~400 | Unit test suite |
| `main.py` | 188 | CLI entry point |
| `demo_recommendations.py` | 129 | Demo scenarios |
| `test_conversation_flow.py` | 118 | Conversation integration test |
| `prime/*.py` | ~150 | Collaborative filtering module |

### 9.3 Demo Results

| Scenario | Parameters | Files Matched | Candidates |
|----------|-----------|--------------|------------|
| Cold Start | (none) | fallback_popular | 20 |
| Gamer + Action + Exciting | segment=gamer, genre=Action, mood=exciting | 3 | 54 |
| Student + Thriller + Exciting | segment=student, genre=Thriller, mood=exciting | 3 | 51 |
| Parent + Comedy + Relaxing | segment=parent, genre=Comedy, mood=relaxing | 3 | 53 |
| 90s Nostalgia | era=90s | 1 | 8 |
| Philosophical (query) | query="deep philosophical" | 3 | 42 |
| Horror + Intense | genre=Horror, mood=intense | 2 | 40 |
| Sci-Fi Adventure (query) | query="sci-fi adventure space" | 3 | 42 |
| Romantic + Emotional | mood=emotional, genre=Romance | 2 | 40 |
| Classic Era | era=Classic | 1 | 20 |

---

## 10. Design Decisions and Trade-offs

### 10.1 Why Pre-computed Files Instead of Real-time Computation

| Approach | Pros | Cons |
|----------|------|------|
| **Pre-computed (chosen)** | O(1) lookup, no context overflow, works with any LLM | Stale data, fixed granularity |
| Real-time vector search | Always fresh, flexible queries | Requires embedding model, high latency |
| Direct LLM prompting | Simple implementation | Context overflow, expensive, non-deterministic |

### 10.2 Why Emotional Inference Instead of Direct Questions

Traditional recommendation systems ask explicit questions: "What genre do you prefer?" This feels mechanical and yields shallow preferences. Emotional inference through casual conversation detects deeper signals:

| User Says | Direct Match | Emotional Inference |
|-----------|-------------|-------------------|
| "Just finished grinding ranked matches" | No match | Segment: gamer, Mood: intense |
| "I'm so tired, just want to chill" | No match | Mood: relaxing |
| "No cap, that was fire" | No match | Segment: gen_z |
| "Back in my day, movies were better" | Era: Classic | Segment: boomer, Era: Classic |

### 10.3 Why Dual-Temperature LLM

A single LLM temperature creates a trade-off between conversation variety and extraction precision. The dual-temperature approach eliminates this:

| Task | Temperature | Rationale |
|------|------------|-----------|
| Conversation generation | 0.8 | Needs creativity, variety, natural flow |
| Preference extraction | 0.3 | Needs consistency, valid JSON output, correct categories |

---

## 11. Challenges and Limitations

### 11.1 LLM Model Quality

The system uses qwen-turbo via nano-gpt.com, a lightweight model. This introduces several limitations:

- **Combined value responses**: Returns `Action/Sci-Fi` instead of choosing one (mitigated by normalization layer)
- **Occasional movie references**: Despite prompt instructions to "never ask about movies directly," the LLM sometimes does
- **Limited emotional inference depth**: Subtle personality cues may be missed by the smaller model

### 11.2 Synthetic Data

The movie dataset contains synthetic entries (e.g., "TheGreat King", "AShawshank Betrayal") generated for testing. The recommendation logic is sound but movie titles are not real-world films (except a few like "Schindler's List" and "Forrest Gump").

### 11.3 Cold Start Granularity

The fallback files provide reasonable recommendations for unknown users, but the system cannot learn from a new user's first session. There is no feedback loop to update recommendation files based on user interactions.

---

## 12. Future Work

- **User feedback loop**: Learn from user ratings on recommendations to improve future suggestions
- **Hybrid approach**: Combine collaborative filtering signals with semantic matching for users who have rating history
- **Memory persistence**: Remember users across sessions for personalized experience
- **Real-time popularity**: Update recommendation files based on trending data
- **REST API**: Expose the system as a web service for integration with front-end applications
- **A/B testing framework**: Test different conversation strategies to optimize preference extraction accuracy

---

## 13. Conclusion

This project demonstrates that the LLM context window limitation can be effectively addressed through a two-stage architecture. By separating offline data processing from online semantic matching, the system handles 1,000 movies and 4,308 users while maintaining fast, explainable recommendations. The emotional inference approach through casual conversation provides a more natural user experience than explicit preference questionnaires, and the multi-value normalization layer ensures robust handling of LLM output variability. With 61 passing unit tests and validated LLM integration, the system is production-ready for deployment.

---

## References

Rothman, D. (2020). *Artificial intelligence by example* (2nd ed.). Packt Publishing.

Artasanchez, A., & Joshi, P. (2020). *Artificial intelligence with Python: Your complete guide to building intelligent apps using Python 3.x and TensorFlow 2* (2nd ed.). Packt Publishing.

LangChain Documentation. (2025). *LangChain: Building applications with LLMs through composability.* https://python.langchain.com/

Anthropic. (2025). *Claude Code Skills documentation.* https://docs.anthropic.com/

Resnick, P., et al. (1994). *GroupLens: An open architecture for collaborative filtering of netnews.* Proceedings of ACM CSCW.
