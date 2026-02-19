import numpy as np
from .compute_scores import pearson_score

def find_similar_users(dataset, user, num_users):
    """
    Find users in the dataset that are similar to the input user.
    
    Args:
        dataset: Dictionary containing user ratings
        user: Input user name
        num_users: Number of similar users to find
    
    Returns:
        List of similar users sorted by similarity score (descending)
    """
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')
    
    # Compute Pearson score between input user and all other users
    scores = np.array([[x, pearson_score(dataset, user, x)] 
                       for x in dataset if x != user])
    
    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    
    # Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]
    
    return scores[top_users]
