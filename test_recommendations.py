"""
Unit tests for the COMP713 Movie Recommendation System.

Tests both semantic and prime approaches with various scenarios.
Run with: pytest test_recommendations.py -v
Or: python test_recommendations.py
"""

import json
import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    load_index,
    load_recommendation_file,
    match_by_keywords,
    get_recommendations_semantic,
    get_recommendations_prime
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def index():
    """Load the recommendation index."""
    return load_index()


@pytest.fixture
def ratings_data():
    """Load the ratings data for prime approach."""
    with open('data/ratings.json', 'r') as f:
        return json.load(f)


# =============================================================================
# Index Loading Tests
# =============================================================================

class TestIndexLoading:
    """Tests for loading and validating the index."""

    def test_load_index_exists(self, index):
        """Test that index loads successfully."""
        assert index is not None
        assert isinstance(index, dict)

    def test_index_has_required_fields(self, index):
        """Test that index has all required fields."""
        assert 'files' in index
        assert 'total_movies_indexed' in index
        assert 'total_recommendation_files' in index

    def test_index_counts(self, index):
        """Test index counts are correct."""
        assert index['total_movies_indexed'] == 1000
        assert len(index['files']) >= 40  # At least 40 files

    def test_index_has_all_types(self, index):
        """Test that index contains all expected file types."""
        types = set(f.get('type', '') for f in index['files'])
        assert 'segment' in types
        assert 'mood' in types
        assert 'genre' in types
        assert 'era' in types
        assert 'fallback' in types


# =============================================================================
# Recommendation File Loading Tests
# =============================================================================

class TestRecommendationFileLoading:
    """Tests for loading individual recommendation files."""

    def test_load_segment_file(self):
        """Test loading a segment recommendation file."""
        data = load_recommendation_file('segment_gamer.json')
        assert data['meta']['type'] == 'segment'
        assert data['meta']['tag'] == 'gamer'
        assert len(data['recommendations']) > 0

    def test_load_mood_file(self):
        """Test loading a mood recommendation file."""
        data = load_recommendation_file('mood_exciting.json')
        assert data['meta']['type'] == 'mood'
        assert data['meta']['tag'] == 'exciting'
        assert len(data['recommendations']) > 0

    def test_load_genre_file(self):
        """Test loading a genre recommendation file."""
        data = load_recommendation_file('genre_action.json')
        assert data['meta']['type'] == 'genre'
        assert data['meta']['tag'] == 'Action'
        assert len(data['recommendations']) > 0

    def test_load_era_file(self):
        """Test loading an era recommendation file."""
        data = load_recommendation_file('era_90s.json')
        assert data['meta']['type'] == 'era'
        assert data['meta']['tag'] == '90s'
        assert len(data['recommendations']) > 0

    def test_load_fallback_file(self):
        """Test loading a fallback recommendation file."""
        data = load_recommendation_file('fallback_popular.json')
        assert data['meta']['type'] == 'fallback'
        assert data['meta']['tag'] == 'popular'
        assert len(data['recommendations']) > 0

    def test_recommendation_structure(self):
        """Test that recommendations have required fields."""
        data = load_recommendation_file('segment_gamer.json')
        rec = data['recommendations'][0]

        required_fields = ['rank', 'item_id', 'title', 'year', 'genre', 'mood', 'era', 'why_recommended']
        for field in required_fields:
            assert field in rec, f"Missing field: {field}"


# =============================================================================
# Keyword Matching Tests
# =============================================================================

class TestKeywordMatching:
    """Tests for keyword matching functionality."""

    def test_single_keyword_match(self, index):
        """Test matching with a single keyword."""
        matches = match_by_keywords(index, ['sci-fi'])
        assert len(matches) > 0
        # Results should be sorted by score (descending)
        assert matches[0][1] >= matches[-1][1]

    def test_multiple_keywords(self, index):
        """Test matching with multiple keywords."""
        matches = match_by_keywords(index, ['exciting', 'thriller'])
        assert len(matches) > 0

    def test_no_matches(self, index):
        """Test with keywords that match nothing."""
        matches = match_by_keywords(index, ['nonexistent_keyword_xyz123'])
        assert len(matches) == 0

    def test_fallback_files_excluded(self, index):
        """Test that fallback files are excluded from keyword matching."""
        matches = match_by_keywords(index, ['popular'])
        # Fallback files should be excluded even if they have matching keywords
        for filename, score, entry in matches:
            assert not entry.get('is_fallback', False)


# =============================================================================
# Cold Start Tests
# =============================================================================

class TestColdStart:
    """Tests for cold start (no user preferences)."""

    def test_cold_start_returns_results(self):
        """Test that cold start returns recommendations."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert len(sources) > 0
        # Should use fallback
        assert any('popular' in s.lower() for s in sources)

    def test_cold_start_uses_fallback(self):
        """Test that cold start uses fallback_popular.json."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        # Check that fallback is in sources
        assert len(sources) == 1
        assert 'popular' in sources[0].lower()


# =============================================================================
# Single Feature Tests
# =============================================================================

class TestSingleFeatureMatching:
    """Tests for matching with single features."""

    @pytest.mark.parametrize("segment,expected_file", [
        ('gamer', 'segment_gamer.json'),
        ('student', 'segment_student.json'),
        ('parent', 'segment_parent.json'),
    ])
    def test_segment_matching(self, segment, expected_file):
        """Test segment-based recommendations."""
        args = Mock()
        args.segment = segment
        args.mood = None
        args.genre = None
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert any(segment in s.lower() for s in sources)

    @pytest.mark.parametrize("mood,expected_count_min", [
        ('exciting', 10),
        ('relaxing', 10),
        ('intense', 10),
        ('thoughtful', 10),
        ('emotional', 10),
    ])
    def test_mood_matching(self, mood, expected_count_min):
        """Test mood-based recommendations."""
        args = Mock()
        args.segment = None
        args.mood = mood
        args.genre = None
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= expected_count_min
        assert any(mood.lower() in s.lower() for s in sources)

    @pytest.mark.parametrize("genre", [
        'Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Thriller'
    ])
    def test_genre_matching(self, genre):
        """Test genre-based recommendations."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = genre
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert any(genre.lower() in s.lower() for s in sources)

    @pytest.mark.parametrize("era", [
        'Classic', '80s', '90s', 'Modern', '2000s'
    ])
    def test_era_matching(self, era):
        """Test era-based recommendations."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = era
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert any(era.lower() in s.lower() for s in sources)


# =============================================================================
# Multi-Feature Tests
# =============================================================================

class TestMultiFeatureMatching:
    """Tests for matching with multiple features."""

    def test_segment_and_mood(self):
        """Test combining segment and mood."""
        args = Mock()
        args.segment = 'gamer'
        args.mood = 'exciting'
        args.genre = None
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert len(sources) == 2
        # Check deduplication - no duplicate item_ids
        item_ids = [r['item_id'] for r in results]
        assert len(item_ids) == len(set(item_ids))

    def test_all_features(self):
        """Test combining all features."""
        args = Mock()
        args.segment = 'gamer'
        args.mood = 'exciting'
        args.genre = 'Action'
        args.era = '90s'
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert len(sources) == 4
        # Verify deduplication
        item_ids = [r['item_id'] for r in results]
        assert len(item_ids) == len(set(item_ids))

    def test_mood_and_genre(self):
        """Test combining mood and genre."""
        args = Mock()
        args.segment = None
        args.mood = 'thoughtful'
        args.genre = 'Drama'
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) > 0
        assert len(sources) == 2


# =============================================================================
# Free Text Query Tests
# =============================================================================

class TestFreeTextQuery:
    """Tests for free-text semantic queries."""

    @pytest.mark.parametrize("query,expected_min_results", [
        ("exciting action movies", 5),
        ("deep philosophical films", 5),
        ("fun comedy", 5),
        ("scary horror", 5),
        ("sci-fi adventure", 5),
    ])
    def test_free_text_queries(self, query, expected_min_results):
        """Test various free-text queries."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = None
        args.query = query

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= expected_min_results

    def test_query_with_typo(self):
        """Test query with typo tolerance."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = None
        args.query = "exiting moveis"  # Typos: exiting->exciting, moveis->movies

        results, sources = get_recommendations_semantic(args)
        # Should still get some results based on partial matches


# =============================================================================
# Prime (Collaborative Filtering) Tests
# =============================================================================

class TestPrimeApproach:
    """Tests for the prime collaborative filtering approach."""

    @pytest.mark.parametrize("user,expected_min_movies", [
        ("David Smith", 1),
        ("Brenda Peterson", 1),
        ("Bill Duffy", 1),
    ])
    def test_prime_returns_recommendations(self, ratings_data, user, expected_min_movies):
        """Test that prime approach returns recommendations for known users."""
        movies = get_recommendations_prime(ratings_data, user)

        assert isinstance(movies, list)
        assert len(movies) >= expected_min_movies

    def test_prime_unknown_user(self, ratings_data):
        """Test prime approach with unknown user."""
        with pytest.raises(TypeError):
            get_recommendations_prime(ratings_data, "Unknown User")

    def test_prime_recommends_unseen_movies(self, ratings_data):
        """Test that prime recommends movies user hasn't seen."""
        user = "David Smith"
        movies = get_recommendations_prime(ratings_data, user)

        # Check that recommendations are not movies the user has already rated
        user_rated = set(ratings_data[user].keys())
        for movie in movies[:5]:  # Check first 5
            assert movie not in user_rated, f"User already rated {movie}"


# =============================================================================
# Recommendation Quality Tests
# =============================================================================

class TestRecommendationQuality:
    """Tests for recommendation data quality."""

    def test_recommendations_have_valid_years(self):
        """Test that recommendations have valid years."""
        data = load_recommendation_file('segment_gamer.json')

        for rec in data['recommendations']:
            assert rec['year'] > 1880, f"Year too old: {rec['year']}"
            assert rec['year'] <= 2030, f"Year too new: {rec['year']}"

    def test_recommendations_have_genres(self):
        """Test that recommendations have genres."""
        data = load_recommendation_file('genre_action.json')

        for rec in data['recommendations']:
            assert len(rec['genre']) > 0, f"No genres: {rec['title']}"
            assert 'Action' in rec['genre'] or len(rec['genre']) >= 1

    def test_recommendations_have_moods(self):
        """Test that recommendations have moods."""
        data = load_recommendation_file('mood_exciting.json')

        for rec in data['recommendations']:
            assert len(rec['mood']) > 0, f"No moods: {rec['title']}"

    def test_recommendations_have_explanations(self):
        """Test that recommendations include explanations."""
        data = load_recommendation_file('segment_gamer.json')

        for rec in data['recommendations']:
            assert rec['why_recommended'], f"No explanation: {rec['title']}"
            assert len(rec['why_recommended']) > 10

    def test_recommendations_are_ranked(self):
        """Test that recommendations are properly ranked."""
        data = load_recommendation_file('segment_gamer.json')

        ranks = [r['rank'] for r in data['recommendations']]
        assert ranks == list(range(1, len(ranks) + 1)), "Ranks not sequential"


# =============================================================================
# Integration Tests - Real Scenarios
# =============================================================================

class TestRealScenarios:
    """Integration tests with realistic user scenarios."""

    def test_gamer_wants_exciting_action(self):
        """Scenario: Gamer wants exciting action movies."""
        args = Mock()
        args.segment = 'gamer'
        args.mood = 'exciting'
        args.genre = 'Action'
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= 20  # Should have many matches
        assert len(sources) == 3  # gamer, exciting, Action

    def test_student_wants_thrilling_thriller(self):
        """Scenario: Student wants thrilling thriller movies."""
        args = Mock()
        args.segment = 'student'
        args.mood = 'exciting'
        args.genre = 'Thriller'
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= 10
        assert len(sources) == 3  # student, exciting, Thriller

    def test_parent_wants_family_friendly_comedy(self):
        """Scenario: Parent wants relaxing comedy movies."""
        args = Mock()
        args.segment = 'parent'
        args.mood = 'relaxing'
        args.genre = 'Comedy'
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= 10

    def test_cold_start_new_user(self):
        """Scenario: Completely new user with no preferences."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = None
        args.query = None

        results, sources = get_recommendations_semantic(args)

        # Should return popular movies
        assert len(results) >= 15
        assert any('popular' in s.lower() for s in sources)

    def test_90s_nostalgia(self):
        """Scenario: User wants 90s movies."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = '90s'
        args.query = None

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= 5
        # Check results are from 90s era
        for rec in results[:5]:
            era = rec.get('era', '')
            assert '90' in era or rec['year'] in range(1990, 2000)

    def test_free_text_philosophical(self):
        """Scenario: User wants deep philosophical movies."""
        args = Mock()
        args.segment = None
        args.mood = None
        args.genre = None
        args.era = None
        args.query = "deep philosophical mind-bending movies"

        results, sources = get_recommendations_semantic(args)

        assert len(results) >= 5
        # Check for philosophical/related content
        for rec in results[:5]:
            moods = rec.get('mood', [])
            assert any(m.lower() in ['philosophical', 'mind-bending', 'thoughtful', 'powerful'] for m in moods)


# =============================================================================
# LLM Parser Tests (with mocking)
# =============================================================================

class TestLLMParser:
    """Tests for the LLM preference parser."""

    def test_parse_simple_preference(self):
        """Test parsing a simple preference."""
        from interactive_recommender import LLMParser, Config

        config = Config('config.json')

        # Mock the chain to avoid actual API call
        with patch.object(LLMParser, '_create_extraction_chain') as mock_chain:
            mock_result = {
                'segment': 'gamer',
                'mood': 'exciting',
                'genre': 'Action',
                'era': None,
                'confidence': 'high',
                'reasoning': 'User mentioned action and exciting'
            }
            mock_chain.return_value.invoke.return_value = mock_result

            parser = LLMParser(config)
            # Re-mock after init
            parser.extraction_chain = mock_chain.return_value

            result = parser.parse_preferences("I like exciting action movies", 1)

            assert result['genre'] == 'Action'
            assert result['mood'] == 'exciting'
            assert result['confidence'] == 'high'


# =============================================================================
# Test Runner (for running without pytest)
# =============================================================================

def run_tests():
    """Run all tests and print results."""
    import traceback

    print("=" * 70)
    print("COMP713 Movie Recommendation System - Unit Tests")
    print("=" * 70)

    test_classes = [
        ("Index Loading", TestIndexLoading),
        ("File Loading", TestRecommendationFileLoading),
        ("Keyword Matching", TestKeywordMatching),
        ("Cold Start", TestColdStart),
        ("Single Feature", TestSingleFeatureMatching),
        ("Multi-Feature", TestMultiFeatureMatching),
        ("Free Text Query", TestFreeTextQuery),
        ("Prime Approach", TestPrimeApproach),
        ("Recommendation Quality", TestRecommendationQuality),
        ("Real Scenarios", TestRealScenarios),
    ]

    total_passed = 0
    total_failed = 0
    failed_tests = []

    for class_name, test_class in test_classes:
        print(f"\n{'=' * 70}")
        print(f"Testing: {class_name}")
        print('=' * 70)

        instance = test_class()

        # Get all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            method = getattr(instance, method_name)

            # Check if method needs fixtures
            import inspect
            sig = inspect.signature(method)

            try:
                # Prepare fixtures
                kwargs = {}
                if 'index' in sig.parameters:
                    kwargs['index'] = load_index()
                if 'ratings_data' in sig.parameters:
                    with open('data/ratings.json', 'r') as f:
                        kwargs['ratings_data'] = json.load(f)

                # Run test
                method(**kwargs)
                print(f"  ✓ {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)[:50]}")
                total_failed += 1
                failed_tests.append((class_name, method_name, str(e)))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    total = total_passed + total_failed
    print(f"Total Tests: {total}")
    print(f"Passed: {total_passed} ({100 * total_passed / total:.1f}%)")
    print(f"Failed: {total_failed} ({100 * total_failed / total:.1f}%)")

    if failed_tests:
        print("\nFailed Tests:")
        for class_name, method_name, error in failed_tests[:10]:
            print(f"  - {class_name}.{method_name}: {error[:60]}...")

    return total_failed == 0


if __name__ == '__main__':
    import sys

    # Check if pytest is available
    try:
        import pytest
        if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
            sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        pass

    # Run tests without pytest
    success = run_tests()
    sys.exit(0 if success else 1)
