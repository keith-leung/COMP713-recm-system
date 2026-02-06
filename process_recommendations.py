#!/usr/bin/env python3
"""
Movie Recommendation Generator - Incremental Processing
Processes chunked movie and user data to generate tagged recommendation files
"""

import json
import os
import glob
from datetime import datetime
from collections import Counter

# State file paths
MOVIES_INDEX_FILE = "_state/movies_index.json"
USER_STATS_FILE = "_state/user_stats.json"
OUTPUT_DIR = "shared_recommendations"

def load_state(filepath):
    """Load state from JSON file, return None if doesn't exist"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_state(filepath, data):
    """Save state to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_json(filepath, data):
    """Save JSON to file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# ============================================================
# PHASE 1: Process Movie Chunks
# ============================================================

def process_movie_chunk(chunk_file):
    """Process ONE movie chunk file"""
    print(f"Processing {chunk_file}...")

    # 1. Load this chunk only
    with open(chunk_file, 'r') as f:
        movies = json.load(f)

    # 2. Load existing state (or create empty)
    state = load_state(MOVIES_INDEX_FILE) or {
        "all_genres": {},
        "all_moods": {},
        "all_eras": {},
        "movies_by_genre": {},
        "movies_by_mood": {},
        "movies_by_era": {},
        "movie_lookup": {},
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
            "description": movie["content"]["description"][:200] if len(movie["content"]["description"]) > 200 else movie["content"]["description"],
            "director": movie["content"]["director"],
            "cast": movie["content"].get("cast", [])[:3],  # First 3 cast members
            "genre": movie["tags"]["genre"],
            "mood": movie["tags"]["mood"],
            "era": era
        }

    # 4. Mark this chunk as processed
    state["processed_chunks"].append(chunk_file)

    # 5. Save state
    save_state(MOVIES_INDEX_FILE, state)

    print(f"  ✓ Processed {chunk_file}: {len(movies)} movies")

# ============================================================
# PHASE 2: Process User Rating Chunks
# ============================================================

def process_user_chunk(chunk_file):
    """Process ONE user ratings chunk file"""
    print(f"Processing {chunk_file}...")

    # 1. Load this chunk only
    with open(chunk_file, 'r') as f:
        users = json.load(f)

    # 2. Load existing state
    state = load_state(USER_STATS_FILE) or {
        "user_segments": {},
        "movie_ratings": {},
        "segment_preferences": {},
        "processed_chunks": []
    }

    # 3. Load movie index (from Phase 1)
    movie_index = load_state(MOVIES_INDEX_FILE)
    if not movie_index:
        raise Exception("Movie index not found! Run Phase 1 first.")

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
    save_state(USER_STATS_FILE, state)

    print(f"  ✓ Processed {chunk_file}: {len(users)} users")

# ============================================================
# PHASE 3: Generate Recommendation Files
# ============================================================

def generate_segment_recommendations(segment_tag, segment_data, segment_prefs, movie_index, movie_ratings):
    """Generate recommendations for a user segment"""

    # Get most liked movies in this segment
    movie_counts = Counter(segment_data["high_rated_movies"])
    top_movies = movie_counts.most_common(20)

    recommendations = []
    for rank, (item_id, high_rating_count) in enumerate(top_movies, 1):
        if item_id not in movie_index["movie_lookup"]:
            continue

        movie_info = movie_index["movie_lookup"][item_id]
        rating_info = movie_ratings.get(item_id, {"total_score": 0, "count": 1, "high_count": 0})

        avg_rating = rating_info["total_score"] / rating_info["count"] if rating_info["count"] > 0 else 0
        pct_high_rating = (high_rating_count / len(segment_data["users"])) * 100

        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "director": movie_info["director"],
            "genre": movie_info["genre"],
            "mood": movie_info["mood"],
            "era": movie_info["era"],
            "description_brief": movie_info["description"],
            "why_recommended": f"{int(pct_high_rating)}% of {segment_tag} users rated this 4+ stars. " +
                              f"Average rating: {avg_rating:.1f}/5.0",
            "stats": {
                "avg_rating_in_segment": round(avg_rating, 2),
                "rating_count_in_segment": high_rating_count,
                "overall_avg_rating": round(avg_rating, 2),
                "total_ratings": rating_info["count"]
            }
        })

    # Get top preferences for match keywords
    top_genres = sorted(segment_prefs.get("genres", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    top_moods = sorted(segment_prefs.get("moods", {}).items(), key=lambda x: x[1], reverse=True)[:3]

    match_keywords = [segment_tag, segment_tag.replace("_", " ")]
    match_keywords.extend([g[0].lower() for g in top_genres])
    match_keywords.extend([m[0].lower() for m in top_moods])

    return {
        "meta": {
            "tag": segment_tag,
            "type": "segment",
            "description": f"Movies highly rated by {segment_tag} users",
            "match_keywords": list(set(match_keywords)),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "user_count_in_segment": len(segment_data["users"]),
            "candidate_movies": len(movie_counts)
        },
        "discovery_questions": [
            {
                "question": f"Are you interested in movies that {segment_tag} users typically enjoy?",
                "positive_signals": [segment_tag, "yes", "interested"]
            }
        ],
        "recommendations": recommendations
    }

def generate_mood_recommendations(mood_name, mood_tags, movie_index, user_stats):
    """Generate recommendations based on mood"""

    # Find movies with these mood tags
    candidates = set()
    for mood_tag in mood_tags:
        if mood_tag in movie_index["movies_by_mood"]:
            candidates.update(movie_index["movies_by_mood"][mood_tag])

    # Score and rank
    scored_movies = []
    for item_id in candidates:
        if item_id in user_stats["movie_ratings"]:
            rating_info = user_stats["movie_ratings"][item_id]
            avg_rating = rating_info["total_score"] / rating_info["count"]
            scored_movies.append((item_id, avg_rating, rating_info["high_count"], rating_info["count"]))

    # Sort by high_count, then avg_rating
    scored_movies.sort(key=lambda x: (x[2], x[1]), reverse=True)
    top_movies = scored_movies[:20]

    recommendations = []
    for rank, (item_id, avg_rating, high_count, total_count) in enumerate(top_movies, 1):
        movie_info = movie_index["movie_lookup"][item_id]

        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "director": movie_info["director"],
            "genre": movie_info["genre"],
            "mood": movie_info["mood"],
            "era": movie_info["era"],
            "description_brief": movie_info["description"],
            "why_recommended": f"Perfect for a {mood_name} mood. Rated {avg_rating:.1f}/5.0 by {total_count} users.",
            "stats": {
                "avg_rating": round(avg_rating, 2),
                "high_rating_count": high_count,
                "total_ratings": total_count
            }
        })

    return {
        "meta": {
            "tag": mood_name,
            "type": "mood",
            "description": f"Movies perfect for a {mood_name} mood",
            "match_keywords": [mood_name] + mood_tags,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "candidate_movies": len(candidates)
        },
        "discovery_questions": [
            {
                "question": f"Are you in the mood for something {mood_name}?",
                "positive_signals": [mood_name, "yes"] + [t.lower() for t in mood_tags]
            }
        ],
        "recommendations": recommendations
    }

def generate_genre_recommendations(genre, movie_index, user_stats):
    """Generate recommendations for a genre"""

    candidates = movie_index["movies_by_genre"].get(genre, [])

    # Score and rank
    scored_movies = []
    for item_id in candidates:
        if item_id in user_stats["movie_ratings"]:
            rating_info = user_stats["movie_ratings"][item_id]
            avg_rating = rating_info["total_score"] / rating_info["count"]
            scored_movies.append((item_id, avg_rating, rating_info["high_count"], rating_info["count"]))

    scored_movies.sort(key=lambda x: (x[2], x[1]), reverse=True)
    top_movies = scored_movies[:20]

    recommendations = []
    for rank, (item_id, avg_rating, high_count, total_count) in enumerate(top_movies, 1):
        movie_info = movie_index["movie_lookup"][item_id]

        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "director": movie_info["director"],
            "genre": movie_info["genre"],
            "mood": movie_info["mood"],
            "era": movie_info["era"],
            "description_brief": movie_info["description"],
            "why_recommended": f"Top-rated {genre} movie. Average rating: {avg_rating:.1f}/5.0",
            "stats": {
                "avg_rating": round(avg_rating, 2),
                "high_rating_count": high_count,
                "total_ratings": total_count
            }
        })

    return {
        "meta": {
            "tag": genre,
            "type": "genre",
            "description": f"Top {genre} movies",
            "match_keywords": [genre.lower(), genre],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "candidate_movies": len(candidates)
        },
        "discovery_questions": [
            {
                "question": f"Do you enjoy {genre} movies?",
                "positive_signals": [genre.lower(), "yes", "love", genre]
            }
        ],
        "recommendations": recommendations
    }

def generate_era_recommendations(era, movie_index, user_stats):
    """Generate recommendations for an era"""

    candidates = movie_index["movies_by_era"].get(era, [])

    # Score and rank
    scored_movies = []
    for item_id in candidates:
        if item_id in user_stats["movie_ratings"]:
            rating_info = user_stats["movie_ratings"][item_id]
            avg_rating = rating_info["total_score"] / rating_info["count"]
            scored_movies.append((item_id, avg_rating, rating_info["high_count"], rating_info["count"]))

    scored_movies.sort(key=lambda x: (x[2], x[1]), reverse=True)
    top_movies = scored_movies[:20]

    recommendations = []
    for rank, (item_id, avg_rating, high_count, total_count) in enumerate(top_movies, 1):
        movie_info = movie_index["movie_lookup"][item_id]

        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "director": movie_info["director"],
            "genre": movie_info["genre"],
            "mood": movie_info["mood"],
            "era": movie_info["era"],
            "description_brief": movie_info["description"],
            "why_recommended": f"Top-rated {era} movie. Average rating: {avg_rating:.1f}/5.0",
            "stats": {
                "avg_rating": round(avg_rating, 2),
                "high_rating_count": high_count,
                "total_ratings": total_count
            }
        })

    return {
        "meta": {
            "tag": era,
            "type": "era",
            "description": f"Top movies from the {era} era",
            "match_keywords": [era.lower(), era],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "candidate_movies": len(candidates)
        },
        "discovery_questions": [
            {
                "question": f"Do you enjoy movies from the {era} era?",
                "positive_signals": [era.lower(), "yes", "love", era]
            }
        ],
        "recommendations": recommendations
    }

def generate_fallback_popular(movie_index, user_stats):
    """Generate fallback recommendations: most popular by average rating"""

    # Score all movies by average rating
    scored_movies = []
    for item_id, rating_info in user_stats["movie_ratings"].items():
        if rating_info["count"] >= 3:  # Require at least 3 ratings
            avg_rating = rating_info["total_score"] / rating_info["count"]
            scored_movies.append((item_id, avg_rating, rating_info["count"]))

    scored_movies.sort(key=lambda x: x[1], reverse=True)
    top_movies = scored_movies[:20]

    recommendations = []
    for rank, (item_id, avg_rating, total_count) in enumerate(top_movies, 1):
        movie_info = movie_index["movie_lookup"][item_id]

        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "director": movie_info["director"],
            "genre": movie_info["genre"],
            "mood": movie_info["mood"],
            "era": movie_info["era"],
            "description_brief": movie_info["description"],
            "why_recommended": f"Highly rated movie with {avg_rating:.1f}/5.0 average from {total_count} users.",
            "stats": {
                "avg_rating": round(avg_rating, 2),
                "total_ratings": total_count
            }
        })

    return {
        "meta": {
            "tag": "popular",
            "type": "fallback",
            "description": "Most popular movies - use when no specific match found",
            "match_keywords": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "is_fallback": True,
            "candidate_movies": len(scored_movies)
        },
        "recommendations": recommendations
    }

def generate_fallback_acclaimed(movie_index, user_stats):
    """Generate fallback recommendations: most acclaimed by high rating count"""

    # Score all movies by number of high ratings
    scored_movies = []
    for item_id, rating_info in user_stats["movie_ratings"].items():
        if rating_info["high_count"] >= 2:  # At least 2 high ratings
            avg_rating = rating_info["total_score"] / rating_info["count"]
            scored_movies.append((item_id, rating_info["high_count"], avg_rating, rating_info["count"]))

    scored_movies.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_movies = scored_movies[:20]

    recommendations = []
    for rank, (item_id, high_count, avg_rating, total_count) in enumerate(top_movies, 1):
        movie_info = movie_index["movie_lookup"][item_id]

        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "director": movie_info["director"],
            "genre": movie_info["genre"],
            "mood": movie_info["mood"],
            "era": movie_info["era"],
            "description_brief": movie_info["description"],
            "why_recommended": f"Critically acclaimed with {high_count} high ratings. Average: {avg_rating:.1f}/5.0",
            "stats": {
                "avg_rating": round(avg_rating, 2),
                "high_rating_count": high_count,
                "total_ratings": total_count
            }
        })

    return {
        "meta": {
            "tag": "acclaimed",
            "type": "fallback",
            "description": "Most acclaimed movies - use when no specific match found",
            "match_keywords": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "is_fallback": True,
            "candidate_movies": len(scored_movies)
        },
        "recommendations": recommendations
    }

def generate_index(generated_files, movie_index, user_stats, output_dir):
    """Generate master index.json"""

    total_users = sum(len(seg["users"]) for seg in user_stats["user_segments"].values())

    index = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_movies_indexed": len(movie_index["movie_lookup"]),
        "total_users_analyzed": total_users,
        "total_recommendation_files": len(generated_files),

        "segments_found": list(user_stats["user_segments"].keys()),
        "genres_found": list(movie_index["all_genres"].keys()),
        "moods_found": list(movie_index["all_moods"].keys()),
        "eras_found": list(movie_index["all_eras"].keys()),

        "files": generated_files
    }

    save_json(f"{output_dir}/index.json", index)
    print(f"  ✓ Generated index.json")

def generate_all_recommendations():
    """Generate all recommendation files from aggregated state"""
    print("\n=== Phase 3: Generating Recommendation Files ===")

    # Load final states
    movie_index = load_state(MOVIES_INDEX_FILE)
    user_stats = load_state(USER_STATS_FILE)

    if not movie_index or not user_stats:
        raise Exception("State files not found! Run Phases 1 and 2 first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated_files = []

    # 1. Generate SEGMENT-based recommendations
    print("\nGenerating segment-based recommendations...")
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
        save_json(f"{OUTPUT_DIR}/{filename}", reco)
        generated_files.append({
            "filename": filename,
            "type": "segment",
            "tag": segment_tag,
            "description": reco["meta"]["description"],
            "match_keywords": reco["meta"]["match_keywords"],
            "item_count": len(reco["recommendations"]),
            "is_fallback": False
        })
        print(f"  ✓ {filename}")

    # 2. Generate MOOD-based recommendations
    print("\nGenerating mood-based recommendations...")
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
        if len(reco["recommendations"]) > 0:
            save_json(f"{OUTPUT_DIR}/{filename}", reco)
            generated_files.append({
                "filename": filename,
                "type": "mood",
                "tag": mood_name,
                "description": reco["meta"]["description"],
                "match_keywords": reco["meta"]["match_keywords"],
                "item_count": len(reco["recommendations"]),
                "is_fallback": False
            })
            print(f"  ✓ {filename}")

    # 3. Generate GENRE-based recommendations
    print("\nGenerating genre-based recommendations...")
    for genre in movie_index["all_genres"].keys():
        if movie_index["all_genres"][genre] < 5:  # Skip rare genres
            continue
        filename = f"genre_{genre.lower().replace('-', '_').replace(' ', '_')}.json"
        reco = generate_genre_recommendations(genre, movie_index, user_stats)
        if len(reco["recommendations"]) > 0:
            save_json(f"{OUTPUT_DIR}/{filename}", reco)
            generated_files.append({
                "filename": filename,
                "type": "genre",
                "tag": genre,
                "description": reco["meta"]["description"],
                "match_keywords": reco["meta"]["match_keywords"],
                "item_count": len(reco["recommendations"]),
                "is_fallback": False
            })
            print(f"  ✓ {filename}")

    # 4. Generate ERA-based recommendations
    print("\nGenerating era-based recommendations...")
    for era in movie_index["all_eras"].keys():
        filename = f"era_{era.lower().replace('-', '_').replace(' ', '_')}.json"
        reco = generate_era_recommendations(era, movie_index, user_stats)
        if len(reco["recommendations"]) > 0:
            save_json(f"{OUTPUT_DIR}/{filename}", reco)
            generated_files.append({
                "filename": filename,
                "type": "era",
                "tag": era,
                "description": reco["meta"]["description"],
                "match_keywords": reco["meta"]["match_keywords"],
                "item_count": len(reco["recommendations"]),
                "is_fallback": False
            })
            print(f"  ✓ {filename}")

    # 5. Generate FALLBACK recommendations
    print("\nGenerating fallback recommendations...")
    reco = generate_fallback_popular(movie_index, user_stats)
    save_json(f"{OUTPUT_DIR}/fallback_popular.json", reco)
    generated_files.append({
        "filename": "fallback_popular.json",
        "type": "fallback",
        "tag": "popular",
        "description": reco["meta"]["description"],
        "match_keywords": [],
        "item_count": len(reco["recommendations"]),
        "is_fallback": True
    })
    print(f"  ✓ fallback_popular.json")

    reco = generate_fallback_acclaimed(movie_index, user_stats)
    save_json(f"{OUTPUT_DIR}/fallback_acclaimed.json", reco)
    generated_files.append({
        "filename": "fallback_acclaimed.json",
        "type": "fallback",
        "tag": "acclaimed",
        "description": reco["meta"]["description"],
        "match_keywords": [],
        "item_count": len(reco["recommendations"]),
        "is_fallback": True
    })
    print(f"  ✓ fallback_acclaimed.json")

    # 6. Generate master INDEX
    print("\nGenerating master index...")
    generate_index(generated_files, movie_index, user_stats, OUTPUT_DIR)

    print(f"\n✓ Successfully generated {len(generated_files)} recommendation files")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("Movie Recommendation Generator - Incremental Processing")
    print("=" * 60)

    # Phase 1: Process all movie chunks
    print("\n=== Phase 1: Processing Movie Chunks ===")
    movie_files = sorted(glob.glob("data/movies_*.json"))
    print(f"Found {len(movie_files)} movie chunk files")

    for movie_file in movie_files:
        process_movie_chunk(movie_file)

    print(f"\n✓ Phase 1 Complete: Processed {len(movie_files)} movie chunks")

    # Phase 2: Process all user rating chunks
    print("\n=== Phase 2: Processing User Rating Chunks ===")
    user_files = sorted(glob.glob("data/user_ratings_*.json"))
    print(f"Found {len(user_files)} user rating chunk files")

    for user_file in user_files:
        process_user_chunk(user_file)

    print(f"\n✓ Phase 2 Complete: Processed {len(user_files)} user rating chunks")

    # Phase 3: Generate all recommendation files
    generate_all_recommendations()

    print("\n" + "=" * 60)
    print("✓ All processing complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"  - index.json (master index)")
    print(f"  - segment_*.json (user segment recommendations)")
    print(f"  - mood_*.json (mood-based recommendations)")
    print(f"  - genre_*.json (genre-based recommendations)")
    print(f"  - era_*.json (era-based recommendations)")
    print(f"  - fallback_*.json (fallback recommendations)")

if __name__ == "__main__":
    main()
