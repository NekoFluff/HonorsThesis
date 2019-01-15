import math
import random
from numpy.random import choice
from data_train_test import dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np

from rating_predictor import get_rating_predictor_using_training_data

# k represents the percentage of movies to be added to the user
k1 = 0.01
k5 = 0.05
k10 = 0.10
k15 = 0.15
NUM_ITEMS = 1682
TEST_PERCENTAGE = 0.2
L_M = [] # List of male movies sorted in decreasing order by scoring function
L_F = [] # List of female movies sorted in decreasing order by scoring function

(train_ratings, train_labels), (test_ratings,
                                test_labels), (training_user_ids, testing_user_ids) = dataset.split_training_testing_for_NN(dataset.MF_training_x, dataset.MF_training_y, TEST_PERCENTAGE)



# Define function, assuming data is all available
def select_movies(user_movies, k_percentage, movies_list, strategy = 'random', print_selected_movie=False):
    ''' Construct a obfuscated list of user movies

    user_movies: Set of movies the user has rated
    k_percentage: Percentage of movies to add to the user's list of rated movies e.g. 1.0 will double the size of the original user_movies list. 
    movies_list: List of tuples (Movie Index, Score) sorted by Score: [('Movie A Index', 3), ('Movie B Index', 2.7), ('Movie C Index', 0.5),...]
    strategy: 'random', 'sampled' or 'greedy'. Details are provided in the BlurMe research paper
    print_selected_movie: Prints every selected movie if enabled
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
            selected_movie = random.choice(movies_list)[0]
        elif (strategy == 'sampled'):
            selected_movie = choice(movies_list, 1, movies_list_distribution)[0]
        elif (strategy == 'greedy'):
            selected_movie = movies_list[greedy_index][0]
            greedy_index += 1
            if greedy_index >= len(movies_list): 
                break
        else:
            print("Please use 'random', 'sampled', or 'greedy' for the strategy.")

        if print_selected_movie:
            print("Selected Movie: ", selected_movie)

        # Check if it isn't in S already
        if selected_movie not in new_user_movies:
            # If it isn't, then add it to the new set
            new_user_movies.add(int(selected_movie))
            num_added += 1

    return new_user_movies


###########################################################
#  Load the NN model and retrieve the L_M and L_F movie lists
###########################################################
model_save_path = './models/gender_inference_NN.h5'
model = keras.models.load_model(model_save_path)
# model.summary()

# We need to recompile the model after loading
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Generate input for each movie to determine if it's male or female
singular_movie_input = np.eye(NUM_ITEMS)

# Perform prediction
predictions = model.predict(singular_movie_input)
print("Sample Movie Predictions:\n[Male, Female]")
for i in predictions[:5]:
    print(i)
print('-'*100)

movie_scores = [(movie_index, p[0] - p[1]) for movie_index, p in enumerate(predictions)] # Male - Female scores. Max 1.0 (Very male) and Min -1.0 (Very female)

# Sort the movie scores
sorted_movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)  # Sort based on timestamp
print("\nSample sorted predictions (First 5):")
for i in sorted_movie_scores[:5]:
    print(i)

print("\nSample sorted predictions (Last 5):")
for i in sorted_movie_scores[-5:]:
    print(i)
print('-'*100)

# Split the movie scores into L_M and L_F 
for index, score_pair in enumerate(sorted_movie_scores):
    if score_pair[1] <= 0:
        print("Split Scores at Index:", index)
        L_M = sorted_movie_scores[:index]
        L_F = sorted_movie_scores[index:]
        L_F = [(i[0], abs(i[1])) for i in L_F]
        L_F.reverse()
        print("L_M (Male Movies List [First 5])", L_M[:5])

        print("L_F (Female Movies List [First 5])", L_F[:5])
        break


mf = get_rating_predictor_using_training_data()

#####################################################################
# Test the select_movies implemented function above on a single user
#####################################################################
# Retrieve user 
user1 = dataset.user_item_matrix[1]
user_gender = dataset.user_info[1]['gender'] # 0 is male, 1 is female
movies_list = L_M if user_gender == 1 else L_F # Use the male list if the user is female. Otherwise use the female list if the user is male

# Retrieve set of movies rated by user
user_movies = set([movie_index for movie_index, rating in enumerate(user1) if rating > 0])


new_user_movies = select_movies(user_movies=user_movies, k_percentage=k10, movies_list=movies_list, strategy = 'greedy')

print("Known User Gender: ", user_gender)
print("Original User Movie Length: ", len(user_movies))
print("New User Movie Length: ", len(new_user_movies))
print("[Added {} movies]".format(len(new_user_movies) - len(user_movies)))


print("\n[[Male Probability | Female Probability]]")
print("Original Predicted User Gender: ", model.predict([[user1]]))
user1_new = [0] * NUM_ITEMS
for movie_index in new_user_movies:
    #new_user_vector[movie_index] = 1
    user1_new[movie_index] = mf.get_rating(1, movie_index)
    #print("User ID:", user_id, "Movie Index:", movie_index, "Rating:", new_user_vector[movie_index])


print("New Predicted User Gender: ", model.predict([[user1_new]]))

print("\n","#"*100,"\n")

################################################################
# Use the select_movies implemented function above on all users
################################################################
success_original = 0
success_obfuscated = 0
total_num_users = 0

modified_user_item_matrix = []

# print("Testing Set:\n", testing_user_ids)
# print("Training Set:\n", training_user_ids)
for user_index, user_vector in enumerate(test_ratings):
    user_id = testing_user_ids[user_index]
    total_num_users += 1
    user_gender = dataset.user_info[user_id]['gender'] # 0 is male, 1 is female
    movies_list = L_M if user_gender == 1 else L_F # Use the male list if the user is female. Otherwise use the female list if the user is male

    # Retrieve set of movies rated by user
    user_movies = set([movie_index for movie_index, rating in enumerate(user_vector) if rating > 0])
    new_user_movies = select_movies(user_movies=user_movies, k_percentage=k15, movies_list=movies_list, strategy = 'greedy')

    # print("Known User Gender: ", user_gender)
    # print("Original User Movie Length: ", len(user_movies))
    # print("New User Movie Length: ", len(new_user_movies))
    # print("[Added {} movies]".format(len(new_user_movies) - len(user_movies)))

    # print("\n[[Male Probability | Female Probability]]")
    old_prediction = model.predict([[user_vector]])
    # print("Original Predicted User Gender: ", old_prediction)

    #print("Old:", np.argmax(old_prediction[0]))
    if user_gender == np.argmax(old_prediction[0]):
        success_original += 1


    new_user_vector = [0] * NUM_ITEMS
    for movie_index in new_user_movies:
        # TODO: Important! Use average/predicted rating instead of 1 if ratings are not binary
        #new_user_vector[movie_index] = 1
        new_user_vector[movie_index] = mf.get_rating(user_id, movie_index)
        #print("User ID:", user_id, "Movie Index:", movie_index, "Rating:", new_user_vector[movie_index])

    modified_user_item_matrix.append((user_id, new_user_vector))
    new_prediction = model.predict([[new_user_vector]])
    #print("New:", np.argmax(new_prediction[0]))

    #print("New Predicted User Gender: ", new_prediction)
    if user_gender == np.argmax(new_prediction[0]):
        success_obfuscated += 1
        
print("\n", "#"*100, "\n")
print("NN Gender Inference Accuracy of test set before obfuscation:", success_original/total_num_users)
# The below two lines are a faster way of performing the some of the above lines of code in the for loop
# loss, acc = model.evaluate(dataset.testing_x, dataset.testing_y)
# print("Restored model accuracy: {:5.2f}%".format(100*acc))

print("NN Gender Inference Accuracy of test set after obfuscation:", success_obfuscated/total_num_users)
print("\n", "#"*100, "\n")


