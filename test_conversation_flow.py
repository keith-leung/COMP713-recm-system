#!/usr/bin/env python3
"""
Test script to verify the interactive conversation flow with freeform recommendations.
Simulates a user conversation and validates:
- Diverse questions (no repetition)
- No explicit movie questions
- Natural conversation flow
- Freeform recommendations generated
"""

import os
import sys
from unittest.mock import patch, MagicMock
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

    # Now generate freeform recommendations
    print("\n=== Generating Freeform Recommendations ===")
    recommendations = parser.generate_recommendations(conversation_history)
    print(f"\nRecommendations:\n{recommendations}\n")

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

    # Check 4: Recommendations were generated
    assert recommendations, "Should generate freeform recommendations"
    print("✓ Freeform recommendations were generated")

    # Check 5: Recommendations mention movies or film content
    rec_lower = recommendations.lower()
    movie_indicators = ["movie", "film", "watch", "recommend"]
    has_movie_content = any(indicator in rec_lower for indicator in movie_indicators)
    assert has_movie_content, "Recommendations should mention movies or films"
    print("✓ Recommendations contain movie-related content")

    print("\n=== All Tests Passed ===")
    print("\nQuestions asked:")
    for i, q in enumerate(questions_asked, 1):
        print(f"{i}. {q}")

    return True

def test_conversation_flow_with_mocks():
    """Test conversation flow using mocks to avoid actual API calls."""

    print("\n=== Testing Conversation Flow (With Mocks) ===\n")

    config = Config()

    # Mock the chains to avoid actual API calls
    with patch.object(LLMParser, '_create_conversation_chain') as mock_conv, \
         patch.object(LLMParser, '_create_recommendation_chain') as mock_rec:

        # Setup mock responses for conversation
        mock_responses = [
            "Nice! What do you like to do for fun?",
            "That's awesome! Do you have any other hobbies?",
            "Cool! What kind of entertainment do you enjoy?"
        ]

        # Setup mock recommendation
        mock_rec_text = ("Based on our conversation, I think you'd enjoy:\n\n"
                        "1. The Matrix (1999) - Perfect for someone who loves gaming and exciting action!\n"
                        "2. Edge of Tomorrow (2014) - Action-packed with a gaming-like time loop mechanic.\n"
                        "3. Ready Player One (2018) - A gamer's paradise with tons of references!")

        # Configure mock chain
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            MagicMock(content=mock_responses[0]),
            MagicMock(content=mock_responses[1]),
            MagicMock(content=mock_responses[2]),
            MagicMock(content=mock_rec_text)
        ]

        mock_conv.return_value = mock_chain
        mock_rec.return_value = mock_chain

        parser = LLMParser(config)
        parser.conversation_chain = mock_chain
        parser.recommendation_chain = mock_chain

        # Simulated conversation
        conversation_history = ""
        discussed_topics = []
        questions_asked = []

        for i, user_input in enumerate(["I like gaming", "Playing competitive matches", "Action movies are cool"]):
            conversation_history += f"\nUser: {user_input}"

            response = parser.generate_conversational_response(
                conversation_history=conversation_history,
                last_input=user_input,
                round_num=i+1,
                discussed_topics=discussed_topics
            )

            print(f"Round {i+1} - Bot: {response}")
            questions_asked.append(response)
            conversation_history += f"\nAssistant: {response}"

        # Generate recommendations
        recommendations = parser.generate_recommendations(conversation_history)
        print(f"\nRecommendations:\n{recommendations}\n")

        # Validation
        assert len(questions_asked) == 3, "Should have 3 conversation rounds"
        assert len(set(questions_asked)) == 3, "All questions should be unique"
        assert recommendations, "Should generate recommendations"

        print("✓ All mock tests passed")

        return True

if __name__ == "__main__":
    try:
        # Test with mocks first (faster, no API calls)
        test_conversation_flow_with_mocks()

        # Then test with real LLM (if available)
        print("\n" + "="*60)
        print("Testing with real LLM (requires Ollama or fallback)...")
        print("="*60)
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
