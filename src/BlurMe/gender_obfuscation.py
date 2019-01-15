import math
import random
import os 
import tensorflow as tf
from tensorflow import keras
import numpy as np

from data_train_test import dataset
from rating_predictor import get_rating_predictor_using_training_data
from gender_inference_NN import split_data, TEST_PERCENTAGES, get_NN_model_location

# k represents the percentage of movies to be added to the user
k_values = [0.01, 0.05, 0.10, 0.15, 0.20]
chosen_strategy = 'greedy' # Or 'sampled', or 'random'
NUM_ITEMS = 1682
L_M = [] # List of male movies sorted in decreasing order by scoring function
L_F = [] # List of female movies sorted in decreasing order by scoring function

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

    if strategy == 'sampled':
        movie_ids = [movie for movie, score in movies_list]

    # Repeat until num_add_movies have been added to the original set of user rated movies
    while num_added < num_add_movies:
                                                                                                                 # Select a movie using the given strategy
        if (strategy == 'random'):
            selected_movie = random.choice(movies_list)[0]
        elif (strategy == 'sampled'):
            selected_movie = np.random.choice(movie_ids, 1, movies_list_distribution)[0]
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


def load_NN_and_movie_lists(model_save_path):
    ###########################################################
    #  Load the NN model and retrieve the L_M and L_F movie lists
    ###########################################################
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
    # print("Sample Movie Predictions:\n[Male, Female]")
    # for i in predictions[:5]:
    #     print(i)
    # print('-'*100)

    movie_scores = [(movie_index, p[0] - p[1]) for movie_index, p in enumerate(predictions)] # Male - Female scores. Max 1.0 (Very male) and Min -1.0 (Very female)

    # Sort the movie scores
    sorted_movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)  # Sort based on timestamp
    # print("\nSample sorted predictions (First 5):")
    # for i in sorted_movie_scores[:5]:
    #     print(i)

    # print("\nSample sorted predictions (Last 5):")
    # for i in sorted_movie_scores[-5:]:
    #     print(i)
    # print('-'*100)

    # Split the movie scores into L_M and L_F 
    for index, score_pair in enumerate(sorted_movie_scores):
        if score_pair[1] <= 0:
            print("Split Scores at Index:", index)
            L_M = sorted_movie_scores[:index]
            L_F = sorted_movie_scores[index:]
            L_F = [(i[0], abs(i[1])) for i in L_F]
            L_F.reverse()
            print("L_M (First 3 in Male Movies List [ID, SCORE])", L_M[:3])

            print("L_F (First 3 in Female Movies List [ID, SCORE])", L_F[:3])
            break

    return model, L_M, L_F

def test_NN_with_user(model, user_vector, user_gender, L_M, L_F, chosen_k):
    mf = get_rating_predictor_using_training_data()

    '''
    k: The percentage of movies to add (obfuscation)
    '''
    #####################################################################
    # Test the select_movies implemented function above on a single user
    #####################################################################
    movies_list = L_M if user_gender == 1 else L_F # Use the male list if the user is female. Otherwise use the female list if the user is male

    # Retrieve set of movies rated by user
    user_movies = set([movie_index for movie_index, rating in enumerate(user_vector) if rating > 0])


    new_user_movies = select_movies(user_movies=user_movies, k_percentage=chosen_k, movies_list=movies_list, strategy=chosen_strategy)

    print("Known User Gender: ", user_gender)
    print("Original User Movie Length: ", len(user_movies))
    print("New User Movie Length: ", len(new_user_movies))
    print("[Added {} movies]".format(len(new_user_movies) - len(user_movies)))


    print("\n[[Male Probability | Female Probability]]")
    print("Original Predicted User Gender: ", model.predict([[user_vector]]))
    user_vector_new = [0] * NUM_ITEMS
    for movie_index in new_user_movies:
        #new_user_vector[movie_index] = 1
        user_vector_new[movie_index] = mf.get_rating(1, movie_index)
        #print("User ID:", user_id, "Movie Index:", movie_index, "Rating:", new_user_vector[movie_index])


    print("New Predicted User Gender: ", model.predict([[user_vector_new]]))

    print("\n","#"*100,"\n")

# print("Testing Set:\n", test_user_ids)
def test_NN(model, test_ratings, test_user_ids, L_M, L_F, chosen_k):
    ''' Evaluate on test set without obfuscation and with obfuscation

    k: The percentage of movies to add (obfuscation)
    '''
    ################################################################
    # Use the select_movies implemented function above on all users
    ################################################################
    success_original = 0
    success_obfuscated = 0
    total_num_users = 0 
    modified_user_item_matrix = [] # Includes user ids
    test_labels = [dataset.user_info[user_id]['gender'] for user_id in test_user_ids]
    test_ratings_obfuscated = [] # Does not include user ids 
    mf = get_rating_predictor_using_training_data()

    for user_index, user_vector in enumerate(test_ratings):
        user_id = test_user_ids[user_index]
        total_num_users += 1
        user_gender = test_labels[user_index]# 0 is male, 1 is female
        movies_list = L_M if user_gender == 1 else L_F # Use the male list if the user is female. Otherwise use the female list if the user is male

        # Retrieve set of movies rated by user
        user_movies = set([movie_index for movie_index, rating in enumerate(user_vector) if rating > 0])
        new_user_movies = select_movies(user_movies=user_movies, k_percentage=chosen_k, movies_list=movies_list, strategy=chosen_strategy)

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

        test_ratings_obfuscated.append(new_user_vector)
        modified_user_item_matrix.append((user_id, new_user_vector))
        new_prediction = model.predict([[new_user_vector]])
        #print("New:", np.argmax(new_prediction[0]))

        #print("New Predicted User Gender: ", new_prediction)
        if user_gender == np.argmax(new_prediction[0]):
            success_obfuscated += 1
            
    print("\n", "#"*100, "\n")
    non_obfuscated_result = success_original/total_num_users
    print("NN Gender Inference Accuracy of test set before obfuscation:", non_obfuscated_result)

    # The below two lines are a faster way of performing the some of the above lines of code in the for loop
    non_obfuscated_loss, non_obfuscated_acc = model.evaluate(test_ratings, test_labels)
    print("Non-Obfuscated Accuracy (using evaluate function):", non_obfuscated_acc)
    print("Non-Obfuscated Loss (using evaluate function):", non_obfuscated_loss)


    obfuscated_result = success_obfuscated/total_num_users
    print("NN Gender Inference Accuracy of test set after obfuscation:", obfuscated_result)
    print("\n", "#"*100, "\n")

    # The below two lines are a faster way of performing the some of the above lines of code in the for loop
    obfuscated_loss, obfuscated_acc = model.evaluate(np.array(test_ratings_obfuscated), test_labels)
    print("Obfuscated Accuracy (using evaluate function):", obfuscated_acc)
    print("Obfuscated Loss (using evaluate function):", obfuscated_loss)

    return non_obfuscated_loss, non_obfuscated_acc, obfuscated_loss, obfuscated_acc, modified_user_item_matrix #(non_obfuscated_result, obfuscated_result)

if __name__ == "__main__":

    # Retrieve sample user 
    user1 = dataset.user_item_matrix[1]
    user_gender = dataset.user_info[1]['gender'] # 0 is male, 1 is female

    # Results across all NNs
    for k in k_values:
        non_obfuscated_losses = []
        non_obfuscated_accuracies = []
        obfuscated_losses = []
        obfuscated_accuracies = []
        for test_percentage in TEST_PERCENTAGES:
            # Load model for every dataset
            (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids) = split_data(test_percentage)

            saved_model_location = get_NN_model_location(test_percentage)
            model, L_M, L_F = load_NN_and_movie_lists(saved_model_location)

            # Uncomment if you want to test on an individual user
            # test_NN_with_user(model, user1, user_gender, L_M, L_F, k)

            non_obfuscated_loss, non_obfuscated_acc, obfuscated_loss, obfuscated_acc, modified_user_item_matrix = test_NN(model, test_ratings, test_user_ids, L_M, L_F, k)
            non_obfuscated_losses.append(non_obfuscated_loss)
            non_obfuscated_accuracies.append(non_obfuscated_acc)
            obfuscated_losses.append(obfuscated_loss)
            obfuscated_accuracies.append(obfuscated_acc)

            
        # Evaluated on test set without obfuscation and with k% obfuscation
        results = np.array([TEST_PERCENTAGES, non_obfuscated_losses, obfuscated_losses, non_obfuscated_accuracies, obfuscated_accuracies])

        if not os.path.exists('./results/'):
            os.makedirs('./results/')
        #np.save("./results/gender_inference_NN_{}k_obfuscation.npy".format(k), results)  
        print('-' * 100)
        print("[First row is test percentages]")
        print("[Second row is non_obfuscated_losses]")
        print("[Third row is obfuscated_losses]")
        print("[Fourth row is non_obfuscated_accuracies]")
        print("[Fifth row is obfuscated_accuracies]")

        print(results)
        save_location = "./results/gender_inference_NN_{:.2f}k_obfuscation.out".format(k)
        print("Saved at " + save_location)
        np.savetxt(save_location, results)

