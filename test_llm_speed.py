#!/usr/bin/env python3
"""
Mock LLM Speed Test
Tests pure API latency for different models - 3 rounds each.
Models that fail or are too slow in rounds 1-2 are disqualified.
"""

import requests
import time

NANOGPT_API_BASE = "https://nano-gpt.com/api/v1"
NANOGPT_API_KEY = "sk-nano-a4dfa5c1-157f-4a68-8e6a-5a255e5eabb1"

MODELS_TO_TEST = [
    # Smallest models first (1B-4B range)
    "gemma-3-1b",
    "gemma-3-4b",
    "llama-3.2-3b",
    "qwen3-1b",
    "qwen3-4b",
    "ministral-3b",
    "phi-4-mini",

    # 7B-8B range
    "mistral-7b",
    "ministral-8b",
    "qwen3-8b",

    # Small efficient models
    "yi-lightning",
    "mistral-nemo",
    "llama-3.1-8b",

    # For comparison
    "gemini-2.0-flash",
    "gpt-4o-mini",
]

# Speed threshold in seconds - if round 1+2 exceed this, model is disqualified
SPEED_THRESHOLD = 4.0

def test_model(model_name):
    """Test a single model with 3 rounds."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {model_name}")
    print('=' * 60)

    times = []

    for round_num in range(1, 4):
        start = time.time()

        try:
            response = requests.post(
                f"{NANOGPT_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {NANOGPT_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Say hello"}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.7
                },
                timeout=30
            )

            elapsed = time.time() - start
            times.append(elapsed)

            if response.status_code == 200:
                print(f"  Round {round_num}: {elapsed:.2f}s ✓")
            else:
                error = response.json().get('error', {}).get('message', 'Unknown error')
                print(f"  Round {round_num}: FAILED - {error}")
                return None

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            print(f"  Round {round_num}: TIMEOUT after {elapsed:.2f}s")
            return None
        except Exception as e:
            elapsed = time.time() - start
            print(f"  Round {round_num}: ERROR - {str(e)[:50]}")
            return None

        # Speed check: disqualify if rounds 1-2 are too slow
        if round_num <= 2:
            if elapsed > SPEED_THRESHOLD:
                print(f"  ❌ DISQUALIFIED - Round {round_num} exceeded {SPEED_THRESHOLD}s threshold")
                return None

    # Calculate stats
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n  Average: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s")

    return {
        'model': model_name,
        'times': times,
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
    }

def main():
    print("=" * 60)
    print("LLM SPEED TEST - 3 Rounds per Model")
    print(f"Speed Threshold: {SPEED_THRESHOLD}s for rounds 1-2")
    print("=" * 60)

    qualified = []
    disqualified = []

    for model in MODELS_TO_TEST:
        result = test_model(model)

        if result:
            qualified.append(result)
        else:
            disqualified.append(model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if qualified:
        print(f"\n✓ QUALIFIED ({len(qualified)}):")
        qualified.sort(key=lambda x: x['avg'])
        for r in qualified:
            print(f"  {r['model']:<25} Avg: {r['avg']:.2f}s  (Min: {r['min']:.2f}s, Max: {r['max']:.2f}s)")

    if disqualified:
        print(f"\n❌ DISQUALIFIED ({len(disqualified)}):")
        for m in disqualified:
            print(f"  - {m}")

    if qualified:
        winner = qualified[0]
        print(f"\n🏆 WINNER: {winner['model']} ({winner['avg']:.2f}s average)")

if __name__ == "__main__":
    main()
