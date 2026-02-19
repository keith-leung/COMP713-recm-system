import numpy as np

def pearson_score(dataset, user1, user2):
    """
    Compute the Pearson correlation score between two users.
    
    Args:
        dataset: Dictionary containing user ratings
        user1: First user name
        user2: Second user name
    
    Returns:
        Pearson correlation score between -1 and 1
    """
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')
    
    # Get the list of movies rated by both users
    common_movies = {}
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    
    num_ratings = len(common_movies)
    
    # If there are no common movies, return 0
    if num_ratings == 0:
        return 0
    
    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])
    
    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])
    
    # Calculate the sum of products of ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])
    
    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
    if Sxx * Syy == 0:
        return 0
    
    return Sxy / np.sqrt(Sxx * Syy)
