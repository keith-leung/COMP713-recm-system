import argparse
import json

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--user', dest='user', required=True, 
            help='Input user for recommendations')
    parser.add_argument('--approach', dest='approach', default='prime',
            choices=['prime'],
            help='Recommendation approach to use (default: prime)')
    return parser

def get_recommendations_prime(data, user):
    """Get recommendations using the prime (collaborative filtering) approach."""
    from prime import get_recommendations
    return get_recommendations(data, user)

def main():
    args = build_arg_parser().parse_args()
    user = args.user
    approach = args.approach

    # Load the movie ratings data
    ratings_file = 'data/ratings.json'
    with open(ratings_file, 'r') as f: 
        data = json.loads(f.read())

    # Get recommendations based on the selected approach
    if approach == 'prime':
        movies = get_recommendations_prime(data, user)
    # Add more approaches here as they are implemented
    # elif approach == 'another_approach':
    #     movies = get_recommendations_another(data, user)
    else:
        print(f"Unknown approach: {approach}")
        return

    # Print the recommendations
    print(f"\nMovie recommendations for {user} (using {approach} approach):")
    for i, movie in enumerate(movies):
        print(f"{i+1}. {movie}")

if __name__ == '__main__':
    main()
