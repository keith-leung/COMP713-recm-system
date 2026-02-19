# Movie Recommendation System - Presentation Slides

**Dawen Liang**
COMP713: Advanced Artificial Intelligence
Dr. Jimmy Tsai
February 2026

---

## Slide 1: Title

### Movie Recommendation System using Anthropic Skills

**Two-Stage Architecture: Offline Processing + Online LLM Interaction**

Dawen Liang | COMP713: Advanced AI | Dr. Jimmy Tsai | February 2026

---

## Slide 2: The Problem

### LLM Context Window Challenge

**Direct LLM Recommendation is Impractical:**
- E-commerce with 1M products
- Token requirement: ~250M tokens per query
- Cost: ~$125 per recommendation
- Latency: 30-60 seconds

**Traditional Methods Have Issues:**
- Collaborative filtering: Cold start problem, no explainability
- Matrix factorization: Requires extensive training data

---

## Slide 3: Key Innovation

### Anthropic Skills (Late 2025)

**What is Skills?**
- Structured tool definitions for LLM agents
- State management across operations
- Incremental processing without context overflow

**This Project:**
- First academic exploration of Skills for recommendation systems
- Cutting-edge technology too new for most textbooks

---

## Slide 4: System Architecture

### Two-Stage Design

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE STAGE                         │
│              (One-time, via Skills)                      │
│  Raw Data → LLM Agent → 41 JSON Recommendation Files    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     ONLINE STAGE                         │
│                    (Real-time)                           │
│  User Chat → Lightweight LLM → Match Files → Results    │
└─────────────────────────────────────────────────────────┘
```

**Key Insight:** Separate heavy data processing from user interaction

---

## Slide 5: Multi-Dimensional Taxonomy

### 4 Dimensions, 41 Pre-Computed Files

| Dimension | Categories | Example Files |
|-----------|------------|---------------|
| **Segment** | 9 types | segment_gamer.json, segment_student.json |
| **Mood** | 5 types | mood_exciting.json, mood_relaxing.json |
| **Genre** | 18 types | genre_action.json, genre_sci_fi.json |
| **Era** | 7 types | era_classic.json, era_90s.json |

**Enables combinatorial personalization** – match users across multiple dimensions simultaneously

---

## Slide 6: Emotional Inference

### Conversational Preference Extraction

| User Statement | Traditional Extraction | LLM Emotional Inference |
|----------------|----------------------|------------------------|
| "Just finished grinding ranked matches" | No match | segment=gamer, mood=intense |
| "I'm so tired, just want to chill" | No match | mood=relaxing |
| "No cap, that was fire" | No match | segment=gen_z |
| "Back in my day, movies were better" | Era: Classic | segment=boomer, era=Classic |

**Advantage:** Natural conversation vs. mechanical questionnaires

---

## Slide 7: Test Results

### 61 Tests, 100% Passing

**Test Coverage:**
- Index & file loading (10 tests)
- Keyword matching (4 tests)
- Cold start handling (2 tests)
- Single/multi-feature matching (19 tests)
- Free-text queries (6 tests)
- Prime collaborative filtering (5 tests)
- Real-world scenarios (6 tests)
- LLM parser & conversation flow (4 tests)

**Recommendation Quality:**
| Scenario | Input | Candidates | Explanation |
|----------|-------|------------|-------------|
| Cold Start | (none) | 20 | "Popular: 4.2/5.0" |
| Gamer + Action | segment=gamer, genre=Action | 54 | "18% of gamers rated 4+" |
| 90s Nostalgia | era=90s | 8 | "Top-rated 90s: 4.1/5.0" |

---

## Slide 8: Comparative Analysis

### Skills vs. Traditional Approaches

| Criterion | Collaborative Filtering | Matrix Factorization | LLM Skills |
|-----------|------------------------|---------------------|------------|
| **Cold Start Handling** | ❌ Fails | ❌ Fails | ✅ Graceful fallback |
| **Explainability** | Low | Low | **High (why_recommended)** |
| **User Interaction** | None | None | **Conversational** |
| **Data Requirements** | Large matrix | Large training set | Moderate user data |
| **Scalability** | O(n×m) per query | O(1) after training | **O(1) file lookup** |
| **Real-time Adaptability** | No | No | **Yes (conversation)** |

---

## Slide 9: Limitations (Key Finding)

### Scalability Boundary

**Fundamental Constraint of LLM Skills:**

| Domain | Catalog Size | Cost per Query | Latency | Viable? |
|--------|-------------|----------------|---------|---------|
| **Movies** | ~1,000 | ~$0.00025 | <2 sec | ✅ Yes |
| **Books** | ~10,000 | ~$0.0025 | ~3 sec | ✅ Yes |
| **E-commerce** | ~10,000,000 | ~$25+ | 30-60 sec | ❌ No |

**Conclusion:**
- Skills work for **bounded, static catalogs** (movies, books)
- Traditional methods win for **massive, high-velocity domains** (Amazon)

---

## Slide 10: Conclusion

### Project Outcomes

**What We Built:**
- ✅ Handles 1,000 movies and 4,308 users efficiently
- ✅ Solves cold start problem via fallback files
- ✅ Achieves sub-2-second response at $0.00025/query
- ✅ 61 passing unit tests validate all functionality

**Skills = Valuable Addition to Recommendation Toolkit**

| Ideal For | Not Ideal For |
|-----------|---------------|
| Conversational interfaces | Large-scale e-commerce |
| Emotional inference | Rapid item turnover |
| Explainable recommendations | Cost-critical applications |

**The movie recommendation system is production-ready.**

---

## Slide 11: Q&A

### Questions?

---

## Demo Appendix (3 minutes if time permits)

### Live Demo Options

```bash
# CLI with known preferences
python main.py --segment gamer --mood exciting --top 3

# Interactive conversation with LLM
python interactive_recommender.py

# 10 pre-built recommendation scenarios
python demo_recommendations.py
```

---

**Total Presentation Time: 10 minutes**

- Slides 1-4: 3 min (context, problem, solution)
- Slides 5-7: 4 min (implementation, results)
- Slides 8-10: 3 min (analysis, limitations, conclusion)
- Demo: 3 min (optional)
