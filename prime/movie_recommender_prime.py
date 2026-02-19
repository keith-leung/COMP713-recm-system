import argparse
import json
import numpy as np
from .collaborative_filtering import find_similar_users

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True, 
            help='Input user')
    return parser

# Get movie recommendations for the input user
def get_recommendations(dataset, input_user, num_similar_users=None):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    # If num_similar_users is not specified, use all other users
    if num_similar_users is None:
        num_similar_users = len(dataset) - 1

    # Use find_similar_users to get similar users sorted by similarity score
    similar_users = find_similar_users(dataset, input_user, num_similar_users)

    # Computes overall scores based on user similarity
    for user_data in similar_users:
        user = user_data[0]
        similarity_score = float(user_data[1])

        if similarity_score <= 0:
            continue

        filtered_list = [x for x in dataset[user] if x not in \
                dataset[input_user] or dataset[input_user][x] == 0]

        for item in filtered_list:
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Generate movie ranks by normalization
    movie_scores = np.array([[score/similarity_scores[item], item] 
            for item, score in overall_scores.items()])

    # Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations
    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = '../data/ratings.json'
    with open(ratings_file, 'r') as f: 
        data = json.loads(f.read())

    print("\nMovie recommendations for " + user + ":")
    movies = get_recommendations(data, user) 
    for i, movie in enumerate(movies):
        print(str(i+1) + '. ' + movie)
