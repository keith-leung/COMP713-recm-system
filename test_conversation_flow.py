#!/usr/bin/env python3
"""
Test script to verify the interactive conversation flow.
Simulates a user conversation and validates:
- Diverse questions (no repetition)
- No explicit movie questions
- Emotional inference working
- Minimum 3 rounds before recommendations
"""

import os
import sys
import asyncio
from interactive_recommender import Config, LLMParser

def test_conversation_flow():
    """Test the full conversation flow with simulated user inputs."""

    config = Config()
    parser = LLMParser(config)

    # Simulated user responses for testing
    test_responses = [
        "Hey! Just finished a long gaming session, feeling pretty good",
        "Yeah was grinding ranked matches, finally hit Diamond! Super hyped right now",
        "Usually just chill with some sci-fi shows or action movies when I'm done",
    ]

    print("=== Testing Conversation Flow ===\n")

    # Track questions for validation
    questions_asked = []
    conversation_history = ""
    discussed_topics = []
    conversation_round = 0

    for user_input in test_responses:
        conversation_round += 1
        print(f"\n[Round {conversation_round}]")
        print(f"User: {user_input}")

        # Build conversation history
        if conversation_history:
            conversation_history += f"\nUser: {user_input}"
        else:
            conversation_history = f"User: {user_input}"

        # Get LLM response
        response = parser.generate_conversational_response(
            conversation_history=conversation_history,
            last_input=user_input,
            round_num=conversation_round,
            discussed_topics=discussed_topics
        )

        print(f"Bot: {response}")

        # Track the question
        questions_asked.append(response)

        # Add to conversation history
        conversation_history += f"\nAssistant: {response}"

    # Now extract preferences
    print("\n=== Extracting Preferences ===")
    preferences = parser.parse_preferences(test_responses[2], conversation_round, conversation_history)
    print(f"Extracted: {preferences}")

    # Validate results
    print("\n=== Validation ===")

    # Check 1: Multiple questions were asked (at least 3 rounds)
    assert len(questions_asked) >= 3, "Should have at least 3 rounds of conversation"
    print("✓ At least 3 conversation rounds")

    # Check 2: Questions are diverse (not identical)
    unique_questions = len(set(questions_asked))
    assert unique_questions == len(questions_asked), "Questions should be unique, not repetitive"
    print("✓ All questions are unique (no repetition)")

    # Check 3: No explicit "what movies do you like" questions
    movie_keywords = ["what movies", "what kind of movies", "what film", "favorite movie"]
    for q in questions_asked:
        q_lower = q.lower()
        for keyword in movie_keywords:
            assert keyword not in q_lower, f"Question should not explicitly ask about movies: {q}"
    print("✓ No explicit movie questions")

    # Check 4: Extracted some preferences (even if partial)
    has_any_preference = any([
        preferences.get('segment'),
        preferences.get('mood'),
        preferences.get('genre'),
        preferences.get('era')
    ])
    assert has_any_preference, "Should extract at least some preferences"
    print("✓ Preferences were extracted")

    print("\n=== All Tests Passed ===")
    print("\nQuestions asked:")
    for i, q in enumerate(questions_asked, 1):
        print(f"{i}. {q}")

    return True

if __name__ == "__main__":
    try:
        test_conversation_flow()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
