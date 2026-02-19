import argparse
import json
import os

RECOMMENDATIONS_DIR = 'shared_recommendations'


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--user', dest='user', default='anonymous',
                        help='Input user for recommendations')
    parser.add_argument('--approach', dest='approach', default='semantic',
                        choices=['prime', 'semantic'],
                        help='Recommendation approach to use (default: semantic)')
    # Semantic approach parameters (known features)
    parser.add_argument('--segment', dest='segment', default=None,
                        help='User segment (e.g., gamer, student, parent, boomer, millennial, gen_z)')
    parser.add_argument('--mood', dest='mood', default=None,
                        help='Desired mood (e.g., exciting, relaxing, intense, thoughtful, emotional)')
    parser.add_argument('--genre', dest='genre', default=None,
                        help='Preferred genre (e.g., Action, Comedy, Thriller, Sci-Fi, Drama)')
    parser.add_argument('--era', dest='era', default=None,
                        help='Preferred era (e.g., Classic, 80s, 90s, Modern, 2000s)')
    parser.add_argument('--query', dest='query', default=None,
                        help='Free-text query for semantic keyword matching (e.g., "something deep and philosophical")')
    parser.add_argument('--top', dest='top', type=int, default=5,
                        help='Number of recommendations to return (default: 5)')
    return parser


def get_recommendations_prime(data, user):
    """Get recommendations using the prime (collaborative filtering) approach."""
    from prime import get_recommendations
    return get_recommendations(data, user)


def load_index():
    """Load the master index of available recommendation files."""
    index_path = os.path.join(RECOMMENDATIONS_DIR, 'index.json')
    with open(index_path, 'r') as f:
        return json.load(f)


def load_recommendation_file(filename):
    """Load a specific recommendation JSON file."""
    filepath = os.path.join(RECOMMENDATIONS_DIR, filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def match_by_keywords(index, query_terms):
    """Match query terms against file match_keywords in the index.

    Returns a list of (filename, score) tuples sorted by match score descending.
    Score is the number of query terms that matched keywords in the file.
    """
    scored = []
    for file_entry in index['files']:
        if file_entry.get('is_fallback'):
            continue
        keywords = [kw.lower() for kw in file_entry.get('match_keywords', [])]
        score = sum(1 for term in query_terms if term in keywords)
        if score > 0:
            scored.append((file_entry['filename'], score, file_entry))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def get_recommendations_semantic(args):
    """Get recommendations using the semantic (pre-computed) approach.

    Supports known features (segment, mood, genre, era), free-text query,
    and cold start (no features -- falls back to popular).
    """
    index = load_index()
    matched_files = []

    # Direct parameter matching: look up the exact file by type and tag
    # Supports comma-separated values (e.g. "Action,Sci-Fi") for multi-match
    def _match_param(param_value, file_type, tag_transform=None):
        if not param_value:
            return
        values = [v.strip() for v in param_value.split(',')]
        for val in values:
            if tag_transform:
                val = tag_transform(val)
            for entry in index['files']:
                if entry['type'] == file_type and entry['tag'].lower() == val.lower():
                    if entry['filename'] not in matched_files:
                        matched_files.append(entry['filename'])
                    break

    _match_param(args.segment, 'segment', lambda v: v.replace('_', ' '))
    _match_param(args.mood, 'mood')
    _match_param(args.genre, 'genre')
    _match_param(args.era, 'era')

    # Free-text query: match against keywords across all files
    if args.query:
        query_terms = [t.strip().lower() for t in args.query.split() if t.strip()]
        keyword_matches = match_by_keywords(index, query_terms)
        for filename, score, entry in keyword_matches:
            if filename not in matched_files:
                matched_files.append(filename)
            if len(matched_files) >= 3:
                break

    # Cold start: no features provided, fall back to popular
    if not matched_files:
        matched_files.append('fallback_popular.json')

    # Collect recommendations from all matched files, deduplicating by item_id
    seen_ids = set()
    results = []
    sources = []

    for filename in matched_files:
        rec_data = load_recommendation_file(filename)
        source_desc = rec_data.get('meta', {}).get('description', filename)
        sources.append(source_desc)
        for rec in rec_data.get('recommendations', []):
            item_id = rec['item_id']
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                rec['_source'] = source_desc
                results.append(rec)

    return results, sources


def print_semantic_results(results, sources, user, top_n):
    """Print semantic recommendation results with explanations."""
    has_features = len(sources) > 0 and sources != ['Most popular movies - use when no specific match found']

    if has_features:
        print(f"\nMovie recommendations for {user} (using semantic approach)")
        print(f"Matched categories: {', '.join(sources)}")
    else:
        print(f"\nMovie recommendations for {user} (cold start - popular movies)")

    print(f"Showing top {min(top_n, len(results))} of {len(results)} candidates:\n")

    for i, rec in enumerate(results[:top_n]):
        print(f"  {i+1}. {rec['title']} ({rec['year']})")
        print(f"     Director: {rec.get('director', 'Unknown')}")
        print(f"     Genre: {', '.join(rec.get('genre', []))}")
        print(f"     Mood: {', '.join(rec.get('mood', []))}")
        print(f"     Era: {rec.get('era', 'Unknown')}")
        print(f"     Why: {rec.get('why_recommended', 'N/A')}")
        if rec.get('_source'):
            print(f"     Source: {rec['_source']}")
        print()


def main():
    args = build_arg_parser().parse_args()
    user = args.user
    approach = args.approach

    if approach == 'prime':
        # Load the movie ratings data for collaborative filtering
        ratings_file = 'data/ratings.json'
        with open(ratings_file, 'r') as f:
            data = json.load(f)

        movies = get_recommendations_prime(data, user)
        print(f"\nMovie recommendations for {user} (using {approach} approach):")
        for i, movie in enumerate(movies):
            print(f"  {i+1}. {movie}")

    elif approach == 'semantic':
        results, sources = get_recommendations_semantic(args)
        print_semantic_results(results, sources, user, args.top)

    else:
        print(f"Unknown approach: {approach}")
        return


if __name__ == '__main__':
    main()
