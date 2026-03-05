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

# Enable readline support for arrow keys and line editing
try:
    import readline
    # Set up readline for better input handling
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")
    # Enable history for user convenience
    readline.set_history_length(100)
    # Configure for better terminal handling
    readline.parse_and_bind("set bell-style none")  # No bell on errors
    readline.parse_and_bind("set show-all-if-ambiguous on")
except ImportError:
    # readline not available on Windows by default, but that's okay
    pass

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

# Also add console handler for LLM timing (for debugging)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
llm_logger = logging.getLogger('llm_timing')
llm_logger.setLevel(logging.INFO)
llm_logger.addHandler(console_handler)
llm_logger.propagate = False

# Reduce noise from external libraries - set to WARNING to suppress DEBUG/INFO logs
for name in ["httpx", "httpcore", "openai", "langchain", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)
    # Also set all child loggers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith(name + ".") or logger_name.startswith(name):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# No more semantic imports - using pure LLM freeform recommendations


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

    # Fallback configuration properties
    @property
    def fallback_enabled(self) -> bool:
        return self.config.get('llm', {}).get('fallback', {}).get('enabled', False)

    @property
    def fallback_api_base(self) -> str:
        return self.config.get('llm', {}).get('fallback', {}).get('api_base', '')

    @property
    def fallback_api_key(self) -> str:
        return self.config.get('llm', {}).get('fallback', {}).get('api_key', '')

    @property
    def fallback_model(self) -> str:
        return self.config.get('llm', {}).get('fallback', {}).get('model', 'qwen-turbo')

    @property
    def prompts_file(self) -> str:
        return self.config.get('llm', {}).get('prompts_file', 'prompts.json')

    def load_prompts(self) -> dict:
        """Load prompts from external file - REQUIRED, no fallbacks."""
        prompts_path = self.prompts_file
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"ERROR: Prompts file not found at {prompts_path}. Please create prompts.json file.")

        try:
            with open(prompts_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to load prompts from {prompts_path}: {e}")


# No more Pydantic models - LLM generates freeform recommendations directly


# =============================================================================
# LLM Parser using LangChain
# =============================================================================

class LLMParser:
    """Uses LangChain with an LLM for conversation and freeform recommendations."""

    # Class-level state to remember if primary failed
    _primary_failed = False

    def __init__(self, config: Config):
        self.config = config
        self._primary_llm = None
        self._fallback_llm = None
        self.llm = self._get_llm(temperature=self.config.temperature)
        self.conversation_llm = self._get_llm(temperature=0.8)
        self.conversation_chain = self._create_conversation_chain()
        self.recommendation_chain = self._create_recommendation_chain()

    def _get_llm(self, temperature: float = 0.3):
        """Get the appropriate LLM based on fallback state."""
        import time

        # If primary already failed, use fallback directly
        if self._primary_failed and self.config.fallback_enabled:
            logger.info(f"[LLM] Primary already failed, using fallback directly")
            return self._get_fallback_llm(temperature)

        # Create primary with fast timeout
        primary = ChatOpenAI(
            model=self.config.model,
            base_url=self.config.api_base,
            api_key=self.config.api_key,
            temperature=temperature,
            timeout=0.5,  # < 1 second timeout for local LLM
            max_retries=0,  # No retries, fail immediately
        )

        if not self.config.fallback_enabled:
            return primary

        # Create fallback
        fallback = self._get_fallback_llm(temperature)

        # Use LangChain's with_fallbacks with custom exception handler
        # that sets our flag after first failure
        from langchain_core.runnables import RunnableLambda

        def try_primary_then_fallback(input):
            start = time.time()
            if self._primary_failed:
                print(f"\n[LLM TIMING] Using fallback directly (primary failed before)")
                logger.info(f"[LLM] Using fallback directly (primary failed before)")
                result = fallback.invoke(input)
                elapsed = time.time() - start
                print(f"[LLM TIMING] Fallback took {elapsed:.2f}s\n")
                logger.info(f"[LLM] Fallback took {elapsed:.2f}s")
                return result
            try:
                print(f"\n[LLM TIMING] Trying primary (localhost:11434)")
                logger.info(f"[LLM] Trying primary (localhost:11434)")
                result = primary.invoke(input)
                elapsed = time.time() - start
                print(f"[LLM TIMING] Primary succeeded in {elapsed:.2f}s\n")
                logger.info(f"[LLM] Primary succeeded in {elapsed:.2f}s")
                return result
            except Exception as e:
                wait_time = time.time() - start
                print(f"[LLM TIMING] Primary failed after {wait_time:.2f}s - switching to fallback")
                logger.warning(f"[LLM] Primary failed after {wait_time:.2f}s: {e}")
                self._primary_failed = True
                fb_start = time.time()
                result = fallback.invoke(input)
                fb_elapsed = time.time() - fb_start
                print(f"[LLM TIMING] Fallback took {fb_elapsed:.2f}s (total: {wait_time + fb_elapsed:.2f}s)\n")
                logger.info(f"[LLM] Fallback took {fb_elapsed:.2f}s")
                return result

        return RunnableLambda(try_primary_then_fallback)

    def _get_fallback_llm(self, temperature: float):
        """Create the fallback LLM."""
        if self._fallback_llm is None:
            self._fallback_llm = ChatOpenAI(
                model=self.config.fallback_model,
                base_url=self.config.fallback_api_base,
                api_key=self.config.fallback_api_key,
                temperature=temperature,
                timeout=30.0,
            )
        return self._fallback_llm

    def _create_conversation_chain(self):
        """Create the LangChain chain for casual conversation.
        Prompts are loaded from external prompts.json file.
        """
        prompts = self.config.load_prompts()
        conversation_config = prompts.get('conversation_chain', {})
        system_prompt = conversation_config.get('system_prompt',
            "ERROR: system_prompt not found in prompts.json under conversation_chain key")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Conversation so far:\n{conversation_history}\n\nRound: {round_num}\n\nTopics already discussed: {discussed_topics}")
        ])

        return prompt | self.conversation_llm

    def _create_recommendation_chain(self):
        """Create the LangChain chain for freeform movie recommendations.
        LLM naturally generates recommendations based on conversation context.
        """
        prompts = self.config.load_prompts()
        recommendation_config = prompts.get('recommendation_chain', {})
        system_prompt = recommendation_config.get('system_prompt',
            "ERROR: system_prompt not found in prompts.json under recommendation_chain key")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here's our conversation:\n\n{conversation_history}")
        ])

        return prompt | self.llm

    # No more normalization methods - LLM handles freeform recommendations

    @property
    def topic_seeds(self) -> list:
        """Load topic seeds from external prompts.json."""
        prompts = self.config.load_prompts()
        conversation_config = prompts.get('conversation_chain', {})
        return conversation_config.get('topic_seeds',
            ["hobbies", "weekend plans", "friends", "food", "music", "entertainment"])

    def generate_conversational_response(self, conversation_history: str, last_input: str, round_num: int, discussed_topics: list) -> str:
        """Generate a casual conversational response.

        Uses the conversation LLM (higher temperature) for natural, varied responses.
        """
        try:
            logger.info(f"Generating conversational response for round {round_num}")

            result = self.conversation_chain.invoke({
                "conversation_history": conversation_history[-800:] if conversation_history else "Just starting",
                "round_num": round_num,
                "discussed_topics": ", ".join(discussed_topics) if discussed_topics else "nothing yet",
            })

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
            for artifact in ["content='", "additional_kwargs=", "response_metadata=", "true", "null"]:
                response = response.replace(artifact, "")

            logger.info(f"Cleaned response: {response}")
            return response if response else "I see, tell me more."

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}", exc_info=True)
            import random
            fallback_responses = [
                "Interesting! Tell me more.",
                "No way, really?",
                "That's cool.",
                "Gotcha. What else is on your mind?",
            ]
            return random.choice(fallback_responses)

    def generate_recommendations(self, conversation_history: str) -> str:
        """Generate freeform movie recommendations based on the conversation.

        LLM naturally reads the conversation and recommends movies.
        No structured extraction - just pure LLM vibe understanding.
        """
        try:
            logger.info("Generating freeform recommendations")

            result = self.recommendation_chain.invoke({
                "conversation_history": conversation_history,
            })

            # Extract the response
            response = ""
            if hasattr(result, 'content'):
                response = result.content
            elif isinstance(result, str):
                response = result
            elif isinstance(result, dict):
                response = result.get('content', str(result))

            return str(response).strip()

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            return "Sorry, I had trouble generating recommendations. Let's try again!"


# =============================================================================
# Discovery Questions Flow
# =============================================================================

class DiscoveryQuestions:
    """Manages casual conversation to infer user preferences emotionally."""

    def __init__(self, parser: LLMParser, config: Config):
        self.parser = parser
        self.config = config
        self.round_num = 0
        self.conversation_history = ""
        self.discussed_topics = []

    def ask_questions(self) -> str:
        """Run casual conversation, then generate freeform recommendations.

        Returns:
            The full conversation history for recommendation generation
        """
        print("=" * 60)
        print("Movie Recommendation System")
        print("(Let's have a casual chat first...)")
        print("=" * 60)

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

            if user_input.lower() in ['done', 'skip', 'that\'s it', 'bye', 'goodbye', 'quit', 'exit', 'stop', 'recommend', 'recommendations']:
                print("\nAlright, let me find some recommendations for you...")
                break

            # Add to conversation history
            self.conversation_history += f"\nYou: {user_input}"

            # Track topics discussed
            self._track_discussed_topics(user_input)

            # Generate conversational response
            response = self.parser.generate_conversational_response(
                self.conversation_history,
                user_input,
                self.round_num,
                self.discussed_topics
            )

            # Clean up response
            response = str(response).strip().strip('"').strip("'")
            for artifact in ["content=", "response=", "additional_kwargs=", "response_metadata="]:
                response = response.replace(artifact, "")

            # Track the AI's actual response in conversation history
            self.conversation_history += f"\nAssistant: {response}"

            print(f"\n{response}")

            # After min_rounds, check if user wants recommendations
            if self.round_num >= self.config.min_rounds:
                try:
                    more = input("\nWant to see some recommendations now, or keep chatting? ( recommendations / chat ): ").strip().lower()
                    if more in ['recommendations', 'rec', 'yes', 'y', 'show', 'done']:
                        break
                    elif more in ['chat', 'continue', 'more', 'c']:
                        print("\n")
                    else:
                        break
                except (EOFError, KeyboardInterrupt):
                    break

        return self.conversation_history

    def _track_discussed_topics(self, user_input: str):
        """Track what topics have been mentioned to avoid repetition.
        The LLM intelligently extracts topics from user input - NO hardcoded keyword mapping.
        """
        # For now, simply append the raw input to avoid repetition
        # The LLM handles topic diversity through conversation context
        if user_input and user_input not in self.discussed_topics:
            self.discussed_topics.append(user_input[:50])  # Store first 50 chars as topic identifier

    # No more _show_understanding or _has_sufficient_info - LLM handles everything


# =============================================================================
# Main Recommender
# =============================================================================

class InteractiveRecommender:
    """Main interactive recommendation system using LangChain with freeform recommendations."""

    def __init__(self, config_path: str = 'config.json'):
        self.config = Config(config_path)
        self.parser = LLMParser(self.config)

    def run(self):
        """Run the interactive recommendation session."""
        print_header("Interactive Movie Recommendation System")
        print(f"Connected to LLM at: {self.config.api_base}")
        print(f"Using model: {self.config.model}")
        print(f"(Freeform recommendations - LLM understands your vibe naturally)")

        try:
            # Run casual conversation
            discovery = DiscoveryQuestions(self.parser, self.config)
            conversation_history = discovery.ask_questions()

            # Generate freeform recommendations
            self.generate_recommendations(conversation_history)

        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using the Movie Recommendation System.")
        except Exception as e:
            print(f"\n\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

    def generate_recommendations(self, conversation_history: str):
        """Generate and display freeform movie recommendations based on conversation."""
        print_header("Your Movie Recommendations")

        # Let the LLM generate freeform recommendations
        recommendations = self.parser.generate_recommendations(conversation_history)

        print(f"\n{recommendations}\n")


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactive Movie Recommendation System (powered by LangChain with freeform LLM recommendations)'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to config file (default: config.json)'
    )

    args = parser.parse_args()
    recommender = InteractiveRecommender(args.config)
    recommender.run()


if __name__ == '__main__':
    main()
