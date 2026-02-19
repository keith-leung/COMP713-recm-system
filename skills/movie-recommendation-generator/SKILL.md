# Movie Recommendation Generator Skill

---
name: movie-recommendation-generator
version: 2.0.0
description: |
  Offline recommendation generator that processes chunked data files 
  INCREMENTALLY (one file at a time) to avoid context limit overflow.
  Generates tagged recommendation files for online LLM semantic matching.
author: Keith Liang
---

## Overview

This skill processes large movie/user datasets split into chunks (`movies_001.json` ~ `movies_nnn.json`, `user_ratings_001.json` ~ `user_ratings_nnn.json`). Since loading all data at once would exceed LLM context limits, we process **one chunk at a time**, maintaining intermediate state files.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INCREMENTAL PROCESSING PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘

  [movies_001.json] ──┐                    ┌──> [segment_gamer.json]
  [movies_002.json] ──┼──> Analyze ──┐     ├──> [segment_student.json]
  [movies_nnn.json] ──┘    Movies    │     ├──> [mood_exciting.json]
                              │      │     ├──> [genre_action.json]
                              ▼      │     ├──> [era_classic.json]
                     [_state/movies_index.json]    
                              │      │     ├──> [fallback_popular.json]
                              │      │     └──> [index.json]
  [user_ratings_001.json] ──┐ │      │
  [user_ratings_002.json] ──┼─┼──> Analyze ──> Aggregate ──> Generate
  [user_ratings_nnn.json] ──┘ │    Users         Stats        Files
                              ▼
                     [_state/user_stats.json]
```

---

## Data Structures

### Movie Item (in movies_XXX.json)
```json
{
  "item_id": "M001",
  "title": "Vertigo",
  "year": 1958,
  "content": {
    "description": "A retired detective suffering from acrophobia...",
    "director": "Alfred Hitchcock",
    "cast": ["James Stewart", "Kim Novak", "Tom Helmore"]
  },
  "tags": {
    "genre": ["Thriller", "Mystery", "Romance"],
    "mood": ["Suspenseful", "Psychological"],
    "era": "Classic"
  }
}
```

### User Rating (in user_ratings_XXX.json)
```json
{
  "id0001": {
    "tags": ["gamer"],
    "scores": [
      {"item_id": "M007", "title": "The Godfather", "score": 4.5, "comment": "Great!"}
    ]
  }
}
```

---

## Processing Steps

### Phase 1: Process Movie Chunks (One at a Time)

For each `movies_XXX.json` file:

```python
# Pseudocode for Claude Code
def process_movie_chunk(chunk_file):
    """Process ONE movie chunk file"""
    
    # 1. Load this chunk only
    movies = json.load(chunk_file)
    
    # 2. Load existing state (or create empty)
    state = load_state("_state/movies_index.json") or {
        "all_genres": {},      # genre -> count
        "all_moods": {},       # mood -> count
        "all_eras": {},        # era -> count
        "movies_by_genre": {}, # genre -> [item_ids]
        "movies_by_mood": {},  # mood -> [item_ids]
        "movies_by_era": {},   # era -> [item_ids]
        "movie_lookup": {},    # item_id -> {title, year, genre, mood, era}
        "processed_chunks": []
    }
    
    # 3. Process each movie in this chunk
    for movie in movies:
        item_id = movie["item_id"]
        
        # Index by genre
        for genre in movie["tags"]["genre"]:
            state["all_genres"][genre] = state["all_genres"].get(genre, 0) + 1
            state["movies_by_genre"].setdefault(genre, []).append(item_id)
        
        # Index by mood
        for mood in movie["tags"]["mood"]:
            state["all_moods"][mood] = state["all_moods"].get(mood, 0) + 1
            state["movies_by_mood"].setdefault(mood, []).append(item_id)
        
        # Index by era
        era = movie["tags"]["era"]
        state["all_eras"][era] = state["all_eras"].get(era, 0) + 1
        state["movies_by_era"].setdefault(era, []).append(item_id)
        
        # Store lookup info
        state["movie_lookup"][item_id] = {
            "title": movie["title"],
            "year": movie["year"],
            "description": movie["content"]["description"][:100],
            "director": movie["content"]["director"],
            "genre": movie["tags"]["genre"],
            "mood": movie["tags"]["mood"],
            "era": era
        }
    
    # 4. Mark this chunk as processed
    state["processed_chunks"].append(chunk_file)
    
    # 5. Save state
    save_state("_state/movies_index.json", state)
    
    print(f"✓ Processed {chunk_file}: {len(movies)} movies")
```

**Run for each chunk:**
```bash
# Claude Code executes these sequentially
process_movie_chunk("data/movies_001.json")
process_movie_chunk("data/movies_002.json")
# ... continue until all chunks processed
```

---

### Phase 2: Process User Rating Chunks (One at a Time)

For each `user_ratings_XXX.json` file:

```python
def process_user_chunk(chunk_file):
    """Process ONE user ratings chunk file"""
    
    # 1. Load this chunk only
    users = json.load(chunk_file)
    
    # 2. Load existing state
    state = load_state("_state/user_stats.json") or {
        "user_segments": {},        # segment_tag -> {users: [], high_rated_movies: []}
        "movie_ratings": {},        # item_id -> {total_score, count, high_count}
        "segment_preferences": {},  # segment_tag -> {genres: {}, moods: {}, eras: {}}
        "processed_chunks": []
    }
    
    # 3. Load movie index (from Phase 1)
    movie_index = load_state("_state/movies_index.json")
    
    # 4. Process each user in this chunk
    for user_id, user_data in users.items():
        user_tags = user_data.get("tags") or []
        
        # Track user by segment tags
        for tag in user_tags:
            if tag not in state["user_segments"]:
                state["user_segments"][tag] = {"users": [], "high_rated_movies": []}
            state["user_segments"][tag]["users"].append(user_id)
        
        # If no tags, track as "general"
        if not user_tags:
            state["user_segments"].setdefault("general", {"users": [], "high_rated_movies": []})
            state["user_segments"]["general"]["users"].append(user_id)
            user_tags = ["general"]
        
        # Process ratings
        for rating in user_data.get("scores", []):
            item_id = rating["item_id"]
            score = rating["score"]
            
            # Aggregate movie ratings
            if item_id not in state["movie_ratings"]:
                state["movie_ratings"][item_id] = {"total_score": 0, "count": 0, "high_count": 0}
            state["movie_ratings"][item_id]["total_score"] += score
            state["movie_ratings"][item_id]["count"] += 1
            
            # Track high ratings (score >= 4.0)
            if score >= 4.0:
                state["movie_ratings"][item_id]["high_count"] += 1
                
                # Track which movies this segment likes
                for tag in user_tags:
                    state["user_segments"][tag]["high_rated_movies"].append(item_id)
                
                # Track segment preferences (genre/mood/era)
                if item_id in movie_index["movie_lookup"]:
                    movie_info = movie_index["movie_lookup"][item_id]
                    for tag in user_tags:
                        prefs = state["segment_preferences"].setdefault(tag, {
                            "genres": {}, "moods": {}, "eras": {}
                        })
                        for g in movie_info["genre"]:
                            prefs["genres"][g] = prefs["genres"].get(g, 0) + 1
                        for m in movie_info["mood"]:
                            prefs["moods"][m] = prefs["moods"].get(m, 0) + 1
                        era = movie_info["era"]
                        prefs["eras"][era] = prefs["eras"].get(era, 0) + 1
    
    # 5. Mark processed
    state["processed_chunks"].append(chunk_file)
    save_state("_state/user_stats.json", state)
    
    print(f"✓ Processed {chunk_file}: {len(users)} users")
```

---

### Phase 3: Generate Recommendation Files

After ALL chunks are processed, generate the final recommendation files:

```python
def generate_all_recommendations():
    """Generate all recommendation files from aggregated state"""
    
    # Load final states
    movie_index = load_state("_state/movies_index.json")
    user_stats = load_state("_state/user_stats.json")
    
    output_dir = "shared_recommendations"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # 1. Generate SEGMENT-based recommendations
    for segment_tag, segment_data in user_stats["user_segments"].items():
        if len(segment_data["users"]) < 3:  # Skip tiny segments
            continue
        
        filename = f"segment_{segment_tag.replace(' ', '_').lower()}.json"
        reco = generate_segment_recommendations(
            segment_tag, 
            segment_data, 
            user_stats["segment_preferences"].get(segment_tag, {}),
            movie_index,
            user_stats["movie_ratings"]
        )
        save_json(f"{output_dir}/{filename}", reco)
        generated_files.append({"filename": filename, "type": "segment", "tag": segment_tag})
    
    # 2. Generate MOOD-based recommendations
    mood_groups = {
        "exciting": ["Exciting", "Thrilling", "Action-packed", "Revolutionary"],
        "relaxing": ["Charming", "Romantic", "Heartwarming", "Witty"],
        "intense": ["Intense", "Suspenseful", "Psychological", "Serious"],
        "thoughtful": ["Philosophical", "Mind-bending", "Powerful"],
        "emotional": ["Emotional", "Hopeful", "Bittersweet", "Somber"]
    }
    for mood_name, mood_tags in mood_groups.items():
        filename = f"mood_{mood_name}.json"
        reco = generate_mood_recommendations(mood_name, mood_tags, movie_index, user_stats)
        save_json(f"{output_dir}/{filename}", reco)
        generated_files.append({"filename": filename, "type": "mood", "tag": mood_name})
    
    # 3. Generate GENRE-based recommendations
    for genre in movie_index["all_genres"].keys():
        if movie_index["all_genres"][genre] < 5:  # Skip rare genres
            continue
        filename = f"genre_{genre.lower().replace('-', '_')}.json"
        reco = generate_genre_recommendations(genre, movie_index, user_stats)
        save_json(f"{output_dir}/{filename}", reco)
        generated_files.append({"filename": filename, "type": "genre", "tag": genre})
    
    # 4. Generate ERA-based recommendations
    for era in movie_index["all_eras"].keys():
        filename = f"era_{era.lower().replace('-', '_').replace(' ', '_')}.json"
        reco = generate_era_recommendations(era, movie_index, user_stats)
        save_json(f"{output_dir}/{filename}", reco)
        generated_files.append({"filename": filename, "type": "era", "tag": era})
    
    # 5. Generate FALLBACK recommendations (REQUIRED)
    # Fallback: Popular (by average rating)
    reco = generate_fallback_popular(movie_index, user_stats)
    save_json(f"{output_dir}/fallback_popular.json", reco)
    generated_files.append({"filename": "fallback_popular.json", "type": "fallback", "is_fallback": True})
    
    # Fallback: Acclaimed (by high rating count)
    reco = generate_fallback_acclaimed(movie_index, user_stats)
    save_json(f"{output_dir}/fallback_acclaimed.json", reco)
    generated_files.append({"filename": "fallback_acclaimed.json", "type": "fallback", "is_fallback": True})
    
    # 6. Generate master INDEX
    generate_index(generated_files, movie_index, user_stats, output_dir)
    
    print(f"✓ Generated {len(generated_files)} recommendation files")
```

---

## Output File Format

### Recommendation File Structure

```json
{
  "meta": {
    "tag": "gamer",
    "type": "segment",
    "description": "Movies loved by gamers - typically action-packed, visually stunning, with game-like mechanics",
    "match_keywords": ["gamer", "gaming", "video games", "play games", "esports", "streamer"],
    "generated_at": "2025-01-25T10:00:00Z",
    "user_count_in_segment": 45,
    "candidate_movies": 120
  },
  "discovery_questions": [
    {
      "question": "Do you enjoy playing video games?",
      "positive_signals": ["yes", "love gaming", "play a lot", "gamer"]
    },
    {
      "question": "Do you prefer fast-paced action or slower character-driven stories?",
      "positive_signals": ["fast", "action", "exciting", "adrenaline"]
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "item_id": "M014",
      "title": "The Matrix",
      "year": 1999,
      "director": "Lana Wachowski, Lilly Wachowski",
      "genre": ["Action", "Sci-Fi"],
      "mood": ["Revolutionary", "Action-packed"],
      "era": "90s",
      "description_brief": "A computer hacker learns the true nature of reality...",
      "why_recommended": "Gamers love this for its video-game-inspired action and reality-bending concept. 87% of gamers rated it 4+ stars.",
      "stats": {
        "avg_rating_in_segment": 4.6,
        "rating_count_in_segment": 38,
        "overall_avg_rating": 4.2
      }
    }
  ]
}
```

### Index File (index.json)

```json
{
  "generated_at": "2025-01-25T10:00:00Z",
  "total_movies_indexed": 1000,
  "total_users_analyzed": 500,
  "total_recommendation_files": 25,
  
  "segments_found": ["gamer", "student", "boomer", "gen_z", "millennial", "parent", "male", "female", "general"],
  "genres_found": ["Action", "Crime", "Drama", "Romance", "Sci-Fi", "Comedy", "Thriller"],
  "moods_found": ["Exciting", "Intense", "Romantic", "Philosophical", "Heartwarming"],
  "eras_found": ["Classic", "80s", "90s", "2000s", "Modern"],
  
  "files": [
    {
      "filename": "segment_gamer.json",
      "type": "segment",
      "tag": "gamer",
      "description": "Action-packed movies loved by gamers",
      "match_keywords": ["gamer", "gaming", "video games"],
      "item_count": 20,
      "is_fallback": false
    },
    {
      "filename": "fallback_popular.json",
      "type": "fallback",
      "tag": "popular",
      "description": "Most popular movies - use when no specific match",
      "match_keywords": [],
      "item_count": 20,
      "is_fallback": true
    }
  ]
}
```

---

## Execution Checklist for Claude Code

### Before Running
- [ ] Verify all data chunks exist in `data/` directory
- [ ] Create `_state/` directory for intermediate files
- [ ] Create `shared_recommendations/` directory for output

### Phase 1: Movie Processing
```
For each movies_XXX.json:
  [ ] Load chunk
  [ ] Update movies_index.json state
  [ ] Log progress
```

### Phase 2: User Processing
```
For each user_ratings_XXX.json:
  [ ] Load chunk
  [ ] Update user_stats.json state
  [ ] Log progress
```

### Phase 3: Generation
```
  [ ] Generate segment files (segment_*.json)
  [ ] Generate mood files (mood_*.json)
  [ ] Generate genre files (genre_*.json)
  [ ] Generate era files (era_*.json)
  [ ] Generate fallback files (fallback_*.json) - REQUIRED
  [ ] Generate index.json
```

### Validation
- [ ] All JSON files are valid
- [ ] index.json lists all generated files
- [ ] Fallback files exist and contain recommendations

---

## Directory Structure

```
project/
├── data/
│   ├── movies_001.json
│   ├── movies_002.json
│   ├── user_ratings_001.json
│   └── user_ratings_002.json
│
├── _state/                          # Intermediate state (can be deleted after)
│   ├── movies_index.json           # Aggregated movie index
│   └── user_stats.json             # Aggregated user statistics
│
├── shared_recommendations/          # Final output
│   ├── index.json                  # Master index
│   │
│   ├── segment_gamer.json          # User segment files
│   ├── segment_student.json
│   ├── segment_boomer.json
│   ├── segment_gen_z.json
│   ├── segment_millennial.json
│   ├── segment_parent.json
│   ├── segment_male.json
│   ├── segment_female.json
│   ├── segment_general.json        # Users with no tags
│   │
│   ├── mood_exciting.json          # Mood-based files
│   ├── mood_relaxing.json
│   ├── mood_intense.json
│   ├── mood_thoughtful.json
│   ├── mood_emotional.json
│   │
│   ├── genre_action.json           # Genre-based files
│   ├── genre_crime.json
│   ├── genre_drama.json
│   ├── genre_romance.json
│   ├── genre_sci_fi.json
│   ├── genre_comedy.json
│   ├── genre_thriller.json
│   │
│   ├── era_classic.json            # Era-based files
│   ├── era_80s.json
│   ├── era_90s.json
│   ├── era_modern.json
│   │
│   ├── fallback_popular.json       # REQUIRED: Popular movies
│   └── fallback_acclaimed.json     # REQUIRED: Highly rated movies
│
└── skills/
    └── movie-recommendation-generator/
        └── SKILL.md                # This file
```

---

## Key Design Principles

1. **Incremental Processing**: Never load all data at once. Process one chunk, save state, continue.

2. **Stateful Accumulation**: Use `_state/` directory to store intermediate aggregations across chunks.

3. **Rich Tag System**: Generate multiple recommendation dimensions (segment, mood, genre, era).

4. **Discovery Questions**: Each recommendation file includes questions to help online LLM identify matching users.

5. **Semantic Matching**: Online LLM uses natural language understanding, not exact keyword matching.

6. **Graceful Fallback**: Always have `fallback_*.json` for users who don't match any specific profile.

7. **Explainable Recommendations**: Each recommendation includes `why_recommended` for transparency.