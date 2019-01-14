import math
import random
from numpy.random import choice


# k represents the percentage of movies to be added to the user
k1 = 0.01
k5 = 0.05
k10 = 0.10

# Define function, assuming data is all available
def select_movies(user_movies, k_percentage, movies_list, strategy = 'random'):
    ''' Construct a obfuscated list of user movies

    user_movies: Set of movies the user has rated
    k_percentage: Percentage of movies to add to the user's list of rated movies e.g. 1.0 will double the size of the original user_movies list. 
    movies_list: List of tuples (Movie Name, Score) sorted by Score: [('Movie A', 3), ('Movie B', 2.7), ('Movie C', 0.5),...]
    strategy: 'random', 'sampled' or 'greedy'. Details are provided in the BlurMe research paper
    '''
    
    new_user_movies = set(user_movies)
    weight_total = sum(score for movie, score in movies_list)
    movies_list_distribution = [score/weight_total for movie, score in movies_list] # Create distribution

    # TODO: Ask if this should be rounded up to allow a minimum of 1 movie to be added as long as the user has rated a movie?
    num_add_movies = math.ceil(k_percentage * len(user_movies))
    num_added = 0
    greedy_index = 0

    # Repeat until num_add_movies have been added to the original set of user rated movies
    while num_added < num_add_movies:
                                                                                                                 # Select a movie using the given strategy
        if (strategy == 'random'):
            selected_movie = random.choice(movies_list)
        elif (strategy == 'sampled'):
            selected_movie = choice(movies_list, 1, movies_list_distribution)
        elif (strategy == 'greedy'):
            selected_movie = movies_list[greedy_index]
            greedy_index += 1
        else:
            print("Please use 'random', 'sampled', or 'greedy' for the strategy.")

        # Check if it isn't in S already
        if selected_movie not in new_user_movies:
            # If it isn't, then add it to the new set
            new_user_movies.add(selected_movie)
            num_added += 1

    return new_user_movies

# Use the select_movies implemented function above
user_movies = []
user_gender = 0 # 0 is male, 1 is female
L_M = [] # List of male movies sorted in decreasing order by scoring function
L_F = [] # List of female movies sorted in decreasing order by scoring function
movies_list = L_M if user_gender == 1 else L_F # Use the male list if the user is female. Otherwise use the female list if the user is male

# TODO: Retrieve list of movies rated by user
# TODO: L_M and L_F based on NN
select_movies(user_movies=user_movies, k_percentage=k5, movies_list=movies_list, strategy = 'random')