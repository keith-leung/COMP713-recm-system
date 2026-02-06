"""
Interactive Movie Recommendation System

Uses LangChain with an LLM to parse user preferences through conversational questions,
then recommends movies from the pre-computed shared_recommendations/ folder.
"""

import json
import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime

# Setup logging - ONLY to file, NOT to console
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"recommender_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create file handler only (no console output)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
# Don't propagate to root logger (prevents console output)
logger.propagate = False

# Reduce noise from external libraries - set to WARNING to suppress DEBUG/INFO logs
for name in ["httpx", "httpcore", "openai", "langchain", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)
    # Also set all child loggers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith(name + ".") or logger_name.startswith(name):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    load_index,
    load_recommendation_file,
    get_recommendations_semantic,
    print_semantic_results
)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration manager for the recommendation system."""

    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found at {self.config_path}")
            print("Creating default config file...")
            self.create_default_config()
            return self.load_config()

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "llm": {
                "provider": "openai_compatible",
                "api_base": "http://localhost:11434/v1",
                "api_key": "your-api-key-here",
                "model": "gpt-3.5-turbo",
                "temperature": 0.3
            },
            "recommendation": {
                "max_rounds": 3,
                "fallback_on_insufficient_info": True,
                "default_top_n": 5
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config at {self.config_path}")
        print("Please edit the file with your API settings before running again.")

    @property
    def api_base(self) -> str:
        return self.config.get('llm', {}).get('api_base', 'http://localhost:11434/v1')

    @property
    def api_key(self) -> str:
        return self.config.get('llm', {}).get('api_key', '')

    @property
    def model(self) -> str:
        return self.config.get('llm', {}).get('model', 'gpt-3.5-turbo')

    @property
    def temperature(self) -> float:
        return self.config.get('llm', {}).get('temperature', 0.3)

    @property
    def min_rounds(self) -> int:
        return self.config.get('recommendation', {}).get('min_rounds', 3)

    @property
    def max_rounds(self) -> int:
        return self.config.get('recommendation', {}).get('max_rounds', 10)

    @property
    def default_top_n(self) -> int:
        return self.config.get('recommendation', {}).get('default_top_n', 5)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class MoviePreferences(BaseModel):
    """Structured output for movie preferences."""

    segment: Optional[str] = Field(
        default=None,
        description=f"User segment. Must be one of: general, gamer, parent, student, boomer, female, gen z, male, millennial"
    )
    mood: Optional[str] = Field(
        default=None,
        description="Desired mood. Must be one of: exciting, relaxing, intense, thoughtful, emotional"
    )
    genre: Optional[str] = Field(
        default=None,
        description=f"Preferred genre. Must be one of: Thriller, Mystery, Romance, Crime, Drama, Biography, Sport, Comedy, History, Action, Adventure, Sci-Fi, War, Music, Western, Horror, Fantasy, Animation"
    )
    era: Optional[str] = Field(
        default=None,
        description="Preferred era. Must be one of: Classic, 80s, 90s, Modern, 2000s, 60s-70s, 80s-90s"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the extracted preferences"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these choices were made"
    )


# =============================================================================
# LLM Parser using LangChain
# =============================================================================

class LLMParser:
    """Uses LangChain with an LLM to parse user preferences."""

    # Available options from the index
    AVAILABLE_SEGMENTS = ['general', 'gamer', 'parent', 'student', 'boomer',
                          'female', 'gen z', 'male', 'millennial']
    AVAILABLE_MOODS = ['exciting', 'relaxing', 'intense', 'thoughtful', 'emotional']
    AVAILABLE_GENRES = ['Thriller', 'Mystery', 'Romance', 'Crime', 'Drama',
                        'Biography', 'Sport', 'Comedy', 'History', 'Action',
                        'Adventure', 'Sci-Fi', 'War', 'Music', 'Western',
                        'Horror', 'Fantasy', 'Animation']
    AVAILABLE_ERAS = ['Classic', '80s', '90s', 'Modern', '2000s', '60s-70s', '80s-90s']

    def __init__(self, config: Config):
        self.config = config
        self.llm = self._create_llm(temperature=self.config.temperature)
        self.conversation_llm = self._create_llm(temperature=0.8)
        self.extraction_chain = self._create_extraction_chain()
        self.question_chain = self._create_question_chain()

    def _create_llm(self, temperature: float = 0.3):
        """Create a LangChain LLM instance with specified temperature."""
        return ChatOpenAI(
            model=self.config.model,
            base_url=self.config.api_base,
            api_key=self.config.api_key,
            temperature=temperature,
        )

    def _create_extraction_chain(self):
        """Create the LangChain chain for preference extraction through emotional inference."""

        # Build the available options lists
        segments_list = ', '.join(self.AVAILABLE_SEGMENTS)
        moods_list = ', '.join(self.AVAILABLE_MOODS)
        genres_list = ', '.join(self.AVAILABLE_GENRES)
        eras_list = ', '.join(self.AVAILABLE_ERAS)

        system_prompt = """You are an empathetic observer analyzing casual conversation to infer the user's vibe and personality.

YOUR TASK:
Read between the lines of casual conversation to understand WHO this person is and HOW they're feeling.

EMOTIONAL INFERENCE GUIDE:

**Segment Detection (from personality and lifestyle clues):**
- gamer: mentions gaming, late nights, "grinding", competitive attitude, tech-savvy talk
- student: mentions studying, exams, broke/low budget, campus life, procrastination
- parent: mentions kids, family time, "when kids are asleep", busy schedule
- gen z: uses slang (no cap, bet, slay, rizz), tiktok references, casual tone
- millennial: mentions nostalgia, 90s/2000s, work-life balance, adulting
- boomer: more formal tone, mentions "back in my day", traditional values
- female: often mentions emotional content, relationships, style preferences
- male: often mentions action, competition, technical details
- general: neutral, can't tell specific demographic

**Mood Detection (from emotional state and tone):**
- exciting: high energy, exclamation marks, enthusiasm, "let's go!", hyped
- relaxing: chill vibes, "just want to unwind", low energy but positive, peaceful
- intense: stressed, terse responses, frustration, need for release, edge
- thoughtful: reflective tone, asking deep questions, philosophical, contemplative
- emotional: sad, going through something, need comfort, vulnerable

**Genre Preference (from personality signals):**
- Action/Sci-Fi: competitive, high energy, likes challenges, gamer vibes
- Comedy: lighthearted, joking, wants to laugh, stressed and needs relief
- Drama: serious, empathetic, emotionally intelligent, thoughtful
- Horror/Thriller: mentions adrenaline, excitement, edge, boredom with normal
- Romance: emotional, vulnerable, mentions relationships, comfort-seeking

**Era Preference (from cultural references):**
- Classic: mentions old films, traditional, "they don't make em like they used to"
- 80s-90s: nostalgic for these eras, mentions "grew up watching", retro vibes
- Modern: current references, new releases, streaming services
- 2000s: early internet nostalgia, Y2K mentions

CRITICAL RULES:
1. INFER from emotional signals, tone, and context - NOT explicit keywords
2. Low confidence is OK - better to admit uncertainty than guess wrong
3. If user gives nothing to work with, return null for everything
4. Consider their current emotional state, not just general preferences
5. Short/curt answers = likely tired/stressed (intense or relaxing mood)
6. Enthusiastic answers = exciting mood
7. No preferences detected = all null (cold start)

Return JSON: segment, mood, genre, era, confidence (high/medium/low), reasoning (explain your inference)
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Full conversation:\n{full_conversation}\n\nLatest input: {input}\n\nRound: {round_num}/{max_rounds}")
        ])

        json_parser = JsonOutputParser()

        return prompt | self.llm | json_parser

    def _create_question_chain(self):
        """Create the LangChain chain for generating conversational responses."""

        system_prompt = """You are a friendly assistant having a casual conversation to understand the user's vibe and personality.

CRITICAL - EVERY response must be unique and explore different aspects:
- Don't repeat topics
- Don't stay on the same theme
- Probe different areas of their life naturally
- Be genuinely curious about them

CONVERSATION STRATEGY:
- Round 1: Start with something warm and open, try to make the conversation engaging and interesting to gather more informative sense
- Round 2+: Shift to a completely different topic (hobbies, weekend plans, friends, food, music, entertainment preferences, dreams, memories)
- If they mentioned work/tired, explore OTHER areas (what they do for fun, what makes them happy, weekend activities)
- Keep discovering new facets of their personality

GOOD conversation paths to explore:
- Free time activities and hobbies
- Music and entertainment preferences
- Social life and friends
- Weekend plans or recent adventures
- Food preferences (can indicate personality)
- Technology/gaming usage
- Reading/learning interests
- Travel or dream destinations
- Nostalgia and childhood memories
- Stress relief methods
- What makes them laugh or cry

EMOTIONAL SIGNALS TO NOTICE:
- Short/curt answers + low energy = might be tired, stressed, or bad mood
- High energy + enthusiasm = exciting mood
- Mention of specific activities = segment clues
- Slang and tone = generation clues
- Emotional openness = mood indicators

IMPORTANT:
- Maximum 2 sentences per response
- Never ask about movies directly
- Be friendly and empathetic
- If they seem down on one topic, SHIFT to something uplifting
- Keep discovering NEW things about them

IMPORTANT - DIVERSITY:
- If talked about work/school in last turn, ask about hobbies/fun next
- If talked about tiredness, ask about what energizes them
- If talked about one topic, explore a completely different area next
- Keep the conversation moving in new directions

Return ONLY your conversational response (casual, friendly, 1-2 sentences max).
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Conversation so far:\n{conversation_history}\n\nRound: {round_num}\n\nTopics already discussed: {discussed_topics}\n\nTopic seed: {topic_seed}\n\nGenerate a unique response exploring a NEW direction:")
        ])

        return prompt | self.conversation_llm

    def _normalize_values(self, value: Optional[str], valid_options: List[str]) -> List[str]:
        """Extract all valid values from a raw LLM output string.

        Handles combined values like 'Action/Sci-Fi' by splitting and returning
        ALL valid matches. Also handles case mismatches.
        """
        if not value:
            return []

        matched = []

        # Try exact match first (case-insensitive)
        for opt in valid_options:
            if value.lower() == opt.lower():
                return [opt]

        # Try splitting combined values (e.g. "Action/Sci-Fi", "Action, Comedy")
        parts = [value]
        for sep in ['/', ',', ' and ', ' or ']:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        parts = [p.strip() for p in parts if p.strip()]

        for part in parts:
            for opt in valid_options:
                if part.lower() == opt.lower() and opt not in matched:
                    matched.append(opt)

        # If no matches from splitting, try substring match as last resort
        if not matched:
            for opt in valid_options:
                if opt.lower() in value.lower() and opt not in matched:
                    matched.append(opt)

        if not matched:
            logger.warning(f"Could not normalize '{value}' to any of {valid_options}")

        return matched

    def _normalize_preferences(self, parsed: Dict) -> Dict:
        """Validate and normalize all extracted preferences against known valid values.

        Returns comma-separated strings when multiple values match (e.g. 'Action,Sci-Fi').
        """
        for key, options in [
            ('segment', self.AVAILABLE_SEGMENTS),
            ('mood', self.AVAILABLE_MOODS),
            ('genre', self.AVAILABLE_GENRES),
            ('era', self.AVAILABLE_ERAS),
        ]:
            values = self._normalize_values(parsed.get(key), options)
            parsed[key] = ','.join(values) if values else None
        return parsed

    def parse_preferences(self, user_input: str, round_num: int, full_conversation: str = "") -> Dict:
        """Parse user input to infer preferences from emotional signals.

        Args:
            user_input: The user's natural language response
            round_num: Current question round number
            full_conversation: Full conversation history for context

        Returns:
            Dictionary with inferred preferences
        """
        try:
            logger.info(f"Parsing preferences for round {round_num}: {user_input[:100]}")

            result = self.extraction_chain.invoke({
                "input": user_input,
                "full_conversation": full_conversation or f"User: {user_input}",
                "round_num": round_num,
                "max_rounds": self.config.max_rounds
            })

            logger.debug(f"Extraction result: {result}")

            # result is a dict from JsonOutputParser
            parsed = {
                "segment": result.get("segment"),
                "mood": result.get("mood"),
                "genre": result.get("genre"),
                "era": result.get("era"),
                "confidence": result.get("confidence", "medium"),
                "reasoning": result.get("reasoning", "")
            }

            # Normalize values against known valid options
            parsed = self._normalize_preferences(parsed)

            logger.info(f"Parsed preferences: segment={parsed['segment']}, mood={parsed['mood']}, genre={parsed['genre']}, era={parsed['era']}")
            return parsed

        except Exception as e:
            logger.error(f"Error parsing preferences: {e}", exc_info=True)
            # Return null preferences - don't show error to user
            return {
                "segment": None, "mood": None, "genre": None, "era": None,
                "confidence": "low", "reasoning": f"Parse error: {str(e)}"
            }

    # Topic seeds for initial greeting variety
    TOPIC_SEEDS = [
        "what they do for fun on weekends",
        "a hobby they recently picked up",
        "something that made them smile today",
        "what they'd do with a free afternoon",
        "their favorite way to unwind after a long day",
        "something they're looking forward to this week",
        "a skill they wish they had",
        "what kind of adventures they enjoy",
        "their go-to comfort activity",
        "something they've been curious about lately",
        "what gets them fired up and energized",
        "their ideal lazy day",
    ]

    def generate_conversational_response(self, conversation_history: str, last_input: str, round_num: int, discussed_topics: list) -> str:
        """Generate a casual conversational response.

        Uses the conversation LLM (higher temperature) for natural, varied responses.
        """
        try:
            import random
            logger.info(f"Generating conversational response for round {round_num}")
            logger.debug(f"Discussed topics: {discussed_topics}")

            # Pick a random topic seed so the LLM varies its output
            topic_seed = random.choice(self.TOPIC_SEEDS)

            result = self.question_chain.invoke({
                "conversation_history": conversation_history[-800:] if conversation_history else "Just starting",
                "last_input": last_input,
                "round_num": round_num,
                "discussed_topics": ", ".join(discussed_topics) if discussed_topics else "nothing yet",
                "topic_seed": topic_seed,
            })

            logger.debug(f"LLM response: {result}")

            # Extract the response - handle different response formats
            response = ""
            if hasattr(result, 'content'):
                response = result.content
            elif isinstance(result, str):
                response = result
            elif isinstance(result, dict):
                response = result.get('content', str(result))

            response = str(response).strip()
            # Remove any quotes and artifacts
            response = response.strip('"').strip("'")
            # Remove common artifacts
            for artifact in ["content='", "additional_kwargs=", "response_metadata=", "true", "null"]:
                response = response.replace(artifact, "")

            logger.info(f"Cleaned response: {response}")
            return response if response else "I see, tell me more."

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}", exc_info=True)

            # Diverse fallback responses based on round
            fallback_responses = [
                ["Interesting! Tell me more.", "No way, really?", "I hear you."],
                ["Gotcha. What else is on your mind?", "That's cool.", "Nice."],
                ["Oh really?", "Go on...", "And then what?"],
            ]
            import random
            round_fallbacks = fallback_responses[min(round_num - 1, len(fallback_responses) - 1)]
            fallback = random.choice(round_fallbacks)
            logger.warning(f"Using fallback response: {fallback}")
            return fallback


# =============================================================================
# Discovery Questions Flow
# =============================================================================

class DiscoveryQuestions:
    """Manages casual conversation to infer user preferences emotionally."""

    def __init__(self, parser: LLMParser, config: Config):
        self.parser = parser
        self.config = config
        self.round_num = 0
        self.preferences = {}
        self.conversation_history = ""
        self.discussed_topics = []

    def ask_questions(self) -> Dict:
        """Run casual conversation to infer preferences.

        Returns:
            Final inferred preferences
        """
        print("=" * 60)
        print("Movie Recommendation System")
        print("(Just having a casual chat...)")
        print("=" * 60)

        self.preferences = {
            'segment': None, 'mood': None, 'genre': None, 'era': None
        }
        self.conversation_history = ""
        self.discussed_topics = []

        # Generate initial greeting via LLM
        initial_greeting = self.parser.generate_conversational_response(
            "",
            "",
            1,
            []
        )

        # Clean up initial greeting
        if hasattr(initial_greeting, 'content'):
            initial_greeting = initial_greeting.content
        initial_greeting = str(initial_greeting).strip().strip('"').strip("'")
        for artifact in ["content='", "additional_kwargs=", "response_metadata="]:
            initial_greeting = initial_greeting.replace(artifact, "")

        print(f"\n{initial_greeting}")

        # Track the initial greeting in conversation history so LLM knows what it already said
        self.conversation_history += f"\nAssistant: {initial_greeting}"

        while True:
            self.round_num += 1

            # Get user input
            print(f"\n")
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['done', 'skip', 'that\'s it', 'bye', 'goodbye', 'quit', 'exit', 'stop']:
                print("\nAlright, let me find some recommendations for you...")
                break

            # Add to conversation history
            self.conversation_history += f"\nYou: {user_input}"

            # Track topics discussed
            self._track_discussed_topics(user_input)

            # Parse the preferences (silently)
            parsed = self.parser.parse_preferences(
                user_input,
                self.round_num,
                self.conversation_history
            )

            # Merge with existing preferences
            for key in ['segment', 'mood', 'genre', 'era']:
                if parsed.get(key):
                    self.preferences[key] = parsed[key]

            # Generate casual response
            response = self.parser.generate_conversational_response(
                self.conversation_history,
                user_input,
                self.round_num,
                self.discussed_topics
            )

            # Clean up response
            if hasattr(response, 'content'):
                response = response.content
            response = str(response).strip().strip('"').strip("'")
            for artifact in ["content='", "additional_kwargs=", "response_metadata="]:
                response = response.replace(artifact, "")

            # Track the AI's actual response in conversation history
            self.conversation_history += f"\nAssistant: {response}"

            print(f"\n{response}")

            # After min_rounds, check if user wants recommendations
            if self.round_num >= self.config.min_rounds:
                has_any = any(v is not None for v in self.preferences.values())

                if has_any:
                    try:
                        more = input("\nI'm getting a sense of your vibe. Want to see some recommendations now, or keep chatting? ( recommendations / chat ): ").strip().lower()
                        if more in ['recommendations', 'rec', 'yes', 'y', 'show']:
                            break
                        elif more in ['chat', 'continue', 'more', 'c']:
                            print("\n")
                        else:
                            break
                    except (EOFError, KeyboardInterrupt):
                        break
                else:
                    try:
                        more = input("\nWant to keep chatting a bit more? (Y/n): ").strip().lower()
                        if more == 'n':
                            break
                    except (EOFError, KeyboardInterrupt):
                        break

        return self.preferences

    def _track_discussed_topics(self, user_input: str):
        """Track what topics have been mentioned to avoid repetition."""
        topic_keywords = {
            'work': ['work', 'job', 'office', 'boss', 'colleague', 'meeting', 'deadline'],
            'sleep': ['sleep', 'tired', 'exhausted', 'awake', 'bed', 'nap', 'rest'],
            'gaming': ['game', 'gaming', 'play', 'ranked', 'match', 'level', 'console', 'pc'],
            'family': ['family', 'kids', 'children', 'parent', 'wife', 'husband', 'mom', 'dad'],
            'school': ['school', 'class', 'homework', 'exam', 'study', 'college', 'university'],
            'food': ['food', 'eat', 'dinner', 'lunch', 'breakfast', 'cooking', 'restaurant'],
            'music': ['music', 'song', 'band', 'concert', 'spotify', 'listen'],
            'sports': ['sport', 'gym', 'workout', 'exercise', 'run', 'fitness'],
            'travel': ['travel', 'trip', 'vacation', 'beach', 'holiday'],
            'social': ['friend', 'party', 'hangout', 'social', 'meet'],
            'weekend': ['weekend', 'free time', 'day off', 'holiday'],
            'stress': ['stress', 'stressed', 'anxious', 'worried', 'overwhelmed'],
        }

        user_lower = user_input.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                if topic not in self.discussed_topics:
                    self.discussed_topics.append(topic)

    def _show_understanding(self, parsed: Dict):
        """Show what was inferred (for debugging)."""
        print("\n[Inferred from this response:]")

        parts = []
        for key in ['segment', 'mood', 'genre', 'era']:
            if parsed.get(key):
                parts.append(f"{key}: {parsed[key]}")

        if parts:
            print("  " + ", ".join(parts))

        if parsed.get('reasoning'):
            print(f"  Reasoning: {parsed['reasoning']}")

        # Show accumulated preferences
        accumulated = [f"{k}: {v}" for k, v in self.preferences.items() if v]
        if accumulated:
            print(f"\n[Profile so far: {', '.join(accumulated)}]")

    def _has_sufficient_info(self) -> bool:
        """Check if we have enough information."""
        filled = sum(1 for v in self.preferences.values() if v is not None)
        return filled >= 1


# =============================================================================
# Main Recommender
# =============================================================================

class MockArgs:
    """Mock args object for get_recommendations_semantic."""
    def __init__(self, segment=None, mood=None, genre=None, era=None, query=None):
        self.segment = segment
        self.mood = mood
        self.genre = genre
        self.era = era
        self.query = query


class InteractiveRecommender:
    """Main interactive recommendation system using LangChain."""

    def __init__(self, config_path: str = 'config.json'):
        self.config = Config(config_path)
        self.parser = LLMParser(self.config)
        self.index = load_index()

    def run(self):
        """Run the interactive recommendation session."""
        print_header("Interactive Movie Recommendation System")
        print(f"Connected to LLM at: {self.config.api_base}")
        print(f"Using model: {self.config.model}")
        print(f"Movies available: {self.index['total_movies_indexed']}")
        print(f"Recommendation files: {self.index['total_recommendation_files']}")

        try:
            # Run discovery questions
            discovery = DiscoveryQuestions(self.parser, self.config)
            preferences = discovery.ask_questions()

            # Generate recommendations
            self.generate_recommendations(preferences)

        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using the Movie Recommendation System.")
        except Exception as e:
            print(f"\n\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

    def generate_recommendations(self, preferences: Dict):
        """Generate and display recommendations based on preferences."""
        print_header("Your Movie Recommendations")

        has_prefs = any(preferences.values())

        if not has_prefs:
            print("\nNo specific preferences detected - showing popular movies!\n")
            args = MockArgs()
        else:
            args = MockArgs(
                segment=preferences.get('segment'),
                mood=preferences.get('mood'),
                genre=preferences.get('genre'),
                era=preferences.get('era')
            )

        results, sources = get_recommendations_semantic(args)
        print_semantic_results(results, sources, "You", self.config.default_top_n)

        if has_prefs:
            print("\n[Matched based on your preferences:]")
            for key, value in preferences.items():
                if value:
                    display_value = value.replace(',', ', ') if isinstance(value, str) else value
                    print(f"  - {key}: {display_value}")

        if len(results) > self.config.default_top_n:
            try:
                more = input(f"\nShow more results? ({len(results)} total available) (Y/n): ").strip().lower()
                if more != 'n':
                    print_semantic_results(results, sources, "You", len(results))
            except EOFError:
                # Non-interactive mode, skip the prompt
                pass


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactive Movie Recommendation System (powered by LangChain)'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to config file (default: config.json)'
    )
    parser.add_argument(
        '--test-config',
        action='store_true',
        help='Test the config connection and exit'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a demo with predefined preferences'
    )

    args = parser.parse_args()

    recommender = InteractiveRecommender(args.config)

    if args.test_config:
        print("Testing configuration...")
        print(f"API Base: {recommender.config.api_base}")
        print(f"Model: {recommender.config.model}")

        try:
            test_parser = LLMParser(recommender.config)
            result = test_parser.parse_preferences("I like action movies and exciting content", 1)
            print(f"\nAPI Test Result: {json.dumps(result, indent=2)}")
            print("\n✓ Configuration is working!")
        except Exception as e:
            print(f"\n✗ Configuration test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.demo:
        # Demo mode with predefined preferences
        print_header("Demo Mode")
        demo_preferences = {
            'segment': 'gamer',
            'mood': 'exciting',
            'genre': 'Action',
            'era': None
        }
        print(f"Demo preferences: {demo_preferences}\n")
        recommender.generate_recommendations(demo_preferences)
    else:
        recommender.run()


if __name__ == '__main__':
    main()
