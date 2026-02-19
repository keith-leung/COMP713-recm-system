"""
Demo script showing different recommendation scenarios.

This demonstrates various user personas and their personalized movie recommendations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import get_recommendations_semantic, print_semantic_results
from unittest.mock import Mock


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_scenario(name, segment=None, mood=None, genre=None, era=None, query=None):
    """Run a single demo scenario."""
    print_header(name)

    args = Mock()
    args.segment = segment
    args.mood = mood
    args.genre = genre
    args.era = era
    args.query = query

    # Show what we're looking for
    print("\n[Searching for:]", end=" ")
    parts = []
    if segment:
        parts.append(f"Segment: {segment}")
    if mood:
        parts.append(f"Mood: {mood}")
    if genre:
        parts.append(f"Genre: {genre}")
    if era:
        parts.append(f"Era: {era}")
    if query:
        parts.append(f"Query: \"{query}\"")
    print(", ".join(parts) if parts else "Popular movies (cold start)")

    results, sources = get_recommendations_semantic(args)

    print(f"\n[Found {len(results)} candidates from {len(sources)} source(s)]")
    print(f"[Sources: {', '.join(sources)}]")

    print_semantic_results(results, sources, "Demo User", 3)

    return results


def main():
    """Run all demo scenarios."""
    print_header("COMP713 Movie Recommendation System - Demo Scenarios")

    scenarios = [
        {
            "name": "Scenario 1: Cold Start - New User",
            "description": "User with no prior preferences",
            "params": {}
        },
        {
            "name": "Scenario 2: Gamer wants Action",
            "description": "Gamer segment looking for exciting action movies",
            "params": {"segment": "gamer", "genre": "Action", "mood": "exciting"}
        },
        {
            "name": "Scenario 3: Student wants Thrillers",
            "description": "Student looking for thrilling content",
            "params": {"segment": "student", "genre": "Thriller", "mood": "exciting"}
        },
        {
            "name": "Scenario 4: Parent wants Family Comedy",
            "description": "Parent looking for relaxing comedy",
            "params": {"segment": "parent", "genre": "Comedy", "mood": "relaxing"}
        },
        {
            "name": "Scenario 5: 90s Nostalgia",
            "description": "User wants movies from the 90s",
            "params": {"era": "90s"}
        },
        {
            "name": "Scenario 6: Deep & Philosophical",
            "description": "Free-text query for thought-provoking content",
            "params": {"query": "deep philosophical mind-bending movies"}
        },
        {
            "name": "Scenario 7: Horror Fan",
            "description": "User wants intense horror movies",
            "params": {"genre": "Horror", "mood": "intense"}
        },
        {
            "name": "Scenario 8: Sci-Fi Adventure",
            "description": "Free-text query for sci-fi adventure",
            "params": {"query": "sci-fi adventure space"}
        },
        {
            "name": "Scenario 9: Romantic & Emotional",
            "description": "User wants emotional romantic content",
            "params": {"mood": "emotional", "genre": "Romance"}
        },
        {
            "name": "Scenario 10: Classic Films",
            "description": "User wants classic era films",
            "params": {"era": "Classic"}
        },
    ]

    for scenario in scenarios:
        demo_scenario(
            scenario["name"],
            **scenario["params"]
        )

    print_header("Demo Complete")
    print("All scenarios demonstrated!")
    print("Run with: python demo_recommendations.py")


if __name__ == '__main__':
    main()
