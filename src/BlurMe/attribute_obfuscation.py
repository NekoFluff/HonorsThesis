import math
import random
import os 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import options

from data_train_test import load_dataset
from rating_predictor import get_rating_predictor_using_training_data
from attribute_inference_NN import get_NN_model_location, auc

# Define function, assuming data is all available
def select_movies(user_movies, k, movies_list, strategy = 'random', print_selected_movie=False):
    ''' Construct a obfuscated list of user movies

    user_movies: Set of movies the user has rated
    k: Percentage (or flat value if k >= 1) of movies to add to the user's list of rated movies e.g. 1.0 will double the size of the original user_movies list. 
    movies_list: List of tuples (Movie Index, Class, Score) sorted by Score: [('Movie A Index', 'Class 1', 3), ('Movie B Index', 'Class 1', 2.7), ('Movie C Index', 'Class 2', 0.5),...]
    strategy: 'random', 'sampled' or 'greedy'. Details are provided in the BlurMe research paper
    print_selected_movie: Prints every selected movie if enabled
    '''
    
    new_user_movies = set(user_movies)
    weight_total = sum(score for movie, category, score in movies_list)
    movies_list_distribution = [score/weight_total for movie, category, score in movies_list] # Create distribution

    # TODO: Ask if this should be rounded up to allow a minimum of 1 movie to be added as long as the user has rated a movie?
    if k < 1:
        num_add_movies = math.ceil(k * len(user_movies))
    else:
        num_add_movies = k

    num_added = 0
    greedy_index = 0

    if strategy == 'sampled':
        movie_ids = [movie for movie, attribute, score in movies_list]

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
    #  Load the NN model and retrieve the job movie lists
    ###########################################################
    model = keras.models.load_model(model_save_path)
    # model.summary()

    # We need to recompile the model after loading
    model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Generate input for each movie to determine if it's male or female
    singular_movie_input = np.eye(options.NUM_ITEMS)
    singular_movie_input *= 5
    print(singular_movie_input)
    # Perform prediction
    predictions = model.predict(singular_movie_input)
    # print("Sample Movie Predictions:\n[Male, Female]")
    # for i in predictions[:5]:
    #     print(i)
    # print('-'*100)

    if options.inference_target == 'gender':
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
                L_M = [(i[0], 0, i[1]) for i in L_M]
                L_F = sorted_movie_scores[index:]
                L_F = [(i[0], 1, abs(i[1])) for i in L_F]
                L_F.reverse()
                print("L_M (First 3 in Male Movies List [ID, SCORE])", L_M[:3])

                print("L_F (First 3 in Female Movies List [ID, SCORE])", L_F[:3])
                break

        return model, [L_M, L_F]

    elif options.inference_target == 'job':
        
        movie_scores = [(movie_index, np.argmax(p), p[np.argmax(p)]) for movie_index, p in enumerate(predictions)] # categorized by argmax, scored by softmax
    elif options.inference_target == 'age':
        movie_scores = [(movie_index, np.argmax(p), p[np.argmax(p)]) for movie_index, p in enumerate(predictions)] # categorized by argmax, scored by softmax
    else:
        print("Please use 'age', 'gender', or 'job' as the inference target. Set the inference target in options.py file.")
    
    # Sort the movie scores ('age' or job')
    sorted_movie_scores = sorted(movie_scores, key=lambda x: x[1]) # Sort based on what category it was placed in

    # Split into different arrays
    categorical_movies = [] # First index is category 0 (Older)
    for i in range(options.NUM_CLASSES):
        categorical_movies.append([f for f in sorted_movie_scores if f[1] == i])

    for i in range(options.NUM_CLASSES):
        print("Movies in category #{}: {}".format(i, len(categorical_movies[i]))) # [Over 45, Below or 35, Over or Equal 35 & Below or Equal 45]

    print("\nSample sorted predictions (First 5):")
    for i in sorted_movie_scores[:5]:
        print(i)

    print("\nSample sorted predictions (Last 5):")
    for i in sorted_movie_scores[-5:]:
        print(i)
    print('-'*100)

    return model, categorical_movies

def test_NN_with_user(model, user_vector, user_attribute, categorical_movies, chosen_k):
    dataset = load_dataset(0, chosen_k)
    mf = get_rating_predictor_using_training_data(dataset.MF_training)

    '''
    k: The percentage of movies to add (obfuscation)
    '''
    #####################################################################
    # Test the select_movies implemented function above on a single user
    #####################################################################

    choices = set(range(3)) # One for each job
    choices.remove(user_attribute)
    movies_list = categorical_movies[random.choice(list(choices))[0]]# Use anything besides what the user is defined as

    # Retrieve set of movies rated by user
    user_movies = set([movie_index for movie_index, rating in enumerate(user_vector) if rating > 0])


    new_user_movies = select_movies(user_movies=user_movies, k=chosen_k, movies_list=movies_list, strategy=options.chosen_strategy)

    print("Known User Attribute: ", user_attribute)
    print("Original User Movie Length: ", len(user_movies))
    print("New User Movie Length: ", len(new_user_movies))
    print("[Added {} movies]".format(len(new_user_movies) - len(user_movies)))


    print("Original Predicted User Attribute: ", model.predict([[user_vector]]))
    user_vector_new = [0] * options.NUM_ITEMS
    for movie_index in new_user_movies:
        #new_user_vector[movie_index] = 1
        user_vector_new[movie_index] = mf.get_rating(1, movie_index)
        #print("User ID:", user_id, "Movie Index:", movie_index, "Rating:", new_user_vector[movie_index])


    print("New Predicted User Attribute: ", model.predict([[user_vector_new]]))

    print("\n","#"*100,"\n")

def test_NN(model, test_ratings, test_labels, test_user_ids, categorical_movies, test_percentage, chosen_k):
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
    dataset = load_dataset(0, chosen_k)

    # if options.inference_target == 'gender':
    #     test_labels = [dataset.user_info[user_id]['gender'] for user_id in test_user_ids]
    # elif options.inference_target == 'age':
    #     test_labels = [dataset.user_ages[user_id] for user_id in test_user_ids]
    # elif options.inference_target == 'job':
    #     test_labels = [dataset.user_jobs[user_id] for user_id in test_user_ids]
    # else:
    #     print("Please use 'gender', 'age', or 'job' and the inference target.")
    #     return
        
    test_ratings_obfuscated = [] # Does not include user ids 
    mf = get_rating_predictor_using_training_data(dataset.MF_training)

    for user_index, user_vector in enumerate(test_ratings):
        user_id = test_user_ids[user_index]
        total_num_users += 1
        user_attribute = test_labels[user_index] # 0 is male, 1 is female in the case of 'gender'

        choices = set(range(options.NUM_CLASSES)) # One for each job
        choices.remove(user_attribute)
        for i, v in enumerate(categorical_movies):
            if len(v) == 0 and i != user_attribute:
                choices.remove(i)
        
        movies_list = categorical_movies[random.choice(list(choices))]# Use anything besides what the user is defined as


        # Retrieve set of movies rated by user
        user_movies = set([movie_index for movie_index, rating in enumerate(user_vector) if rating > 0])
        new_user_movies = select_movies(user_movies=user_movies, k=chosen_k, movies_list=movies_list, strategy=options.chosen_strategy)

        # print("Known User job: ", user_attribute)
        # print("Original User Movie Length: ", len(user_movies))
        # print("New User Movie Length: ", len(new_user_movies))
        # print("[Added {} movies]".format(len(new_user_movies) - len(user_movies)))

        # print("\n[[Male Probability | Female Probability]]")
        old_prediction = model.predict([[user_vector]])
        # print("Original Predicted User job: ", old_prediction)

        #print("Old:", np.argmax(old_prediction[0]))
        

        #print("{} == {}? -> {}".format(user_attribute, np.argmax(old_prediction[0]), user_attribute == np.argmax(old_prediction[0])))
        if user_attribute == np.argmax(old_prediction[0]):
            success_original += 1


        # Assign movie rating to added movie (average or predicted)
        new_user_vector = [0] * options.NUM_ITEMS
        for movie_index in new_user_movies:
            if options.average_or_predicted_ratings == 'average':
                new_user_vector[movie_index] = dataset.user_item_averages[movie_index]
            elif options.average_or_predicted_ratings == 'predicted':
                new_user_vector[movie_index] = mf.get_rating(user_id, movie_index)
                
            #print("User ID:", user_id, "Movie Index:", movie_index, "Rating:", new_user_vector[movie_index])


        test_ratings_obfuscated.append(new_user_vector)
        modified_user_item_matrix.append((user_id, new_user_vector))
        new_prediction = model.predict([[new_user_vector]])
        #print("New:", np.argmax(new_prediction[0]))

        #print("New Predicted User job: ", new_prediction)
        if user_attribute == np.argmax(new_prediction[0]):
            success_obfuscated += 1
            
    print("\n", "#"*100, "\n")
    non_obfuscated_result = success_original/total_num_users
    print("NN job Inference Accuracy of test set before obfuscation:", non_obfuscated_result)

    # The below two lines are a faster way of performing the some of the above lines of code in the for loop
    #print("test_labels", test_labels)
    non_obfuscated_loss, non_obfuscated_acc = model.evaluate(test_ratings, test_labels)
    print("Non-Obfuscated Accuracy (using evaluate function):", non_obfuscated_acc)
    print("Non-Obfuscated Loss (using evaluate function):", non_obfuscated_loss)


    obfuscated_result = success_obfuscated/total_num_users
    print("NN job Inference Accuracy of test set after obfuscation:", obfuscated_result)
    print("\n", "#"*100, "\n")

    # The below two lines are a faster way of performing the some of the above lines of code in the for loop
    obfuscated_loss, obfuscated_acc = model.evaluate(np.array(test_ratings_obfuscated), test_labels)
    print("Obfuscated Accuracy (using evaluate function):", obfuscated_acc)
    print("Obfuscated Loss (using evaluate function):", obfuscated_loss)


    non_obfuscated_predicted_attributes = model.predict(test_ratings)
    non_obfuscated_auc_micro, non_obfuscated_auc_macro = auc(test_labels, non_obfuscated_predicted_attributes, test_percentage, chosen_k)

    obfuscated_predicted_attributes = model.predict(np.array(test_ratings_obfuscated))
    obfuscated_auc_micro, obfuscated_auc_macro = auc(test_labels, obfuscated_predicted_attributes, test_percentage, chosen_k)

    return non_obfuscated_loss, non_obfuscated_acc, obfuscated_loss, obfuscated_acc, non_obfuscated_auc_micro, non_obfuscated_auc_macro, obfuscated_auc_micro, obfuscated_auc_macro, modified_user_item_matrix #(non_obfuscated_result, obfuscated_result)

if __name__ == "__main__":

    # Results across all NNs
    for k in options.k_values:
        dataset = load_dataset(0, k)

        non_obfuscated_losses = []
        non_obfuscated_accuracies = []
        obfuscated_losses = []
        obfuscated_accuracies = []

        non_obfuscated_auc_micros = []
        non_obfuscated_auc_macros = []
        obfuscated_auc_micros = []
        obfuscated_auc_macros = []

        for test_percentage in options.TEST_PERCENTAGES:
            # Load model for every dataset
            dataset = load_dataset(test_percentage, k)
            (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids) = dataset.get_training_testing_for_NN()

            saved_model_location = get_NN_model_location(test_percentage)
            model, categorical_movies = load_NN_and_movie_lists(saved_model_location)

            non_obfuscated_loss, non_obfuscated_acc, obfuscated_loss, obfuscated_acc, non_obfuscated_auc_micro, non_obfuscated_auc_macro, obfuscated_auc_micro, obfuscated_auc_macro, modified_user_item_matrix = test_NN(model, test_ratings, test_labels, test_user_ids, categorical_movies, test_percentage, k)

            
            del model, categorical_movies
            tf.keras.backend.clear_session()

            non_obfuscated_losses.append(non_obfuscated_loss)
            non_obfuscated_accuracies.append(non_obfuscated_acc)
            obfuscated_losses.append(obfuscated_loss)
            obfuscated_accuracies.append(obfuscated_acc)
            
            non_obfuscated_auc_micros.append(non_obfuscated_auc_micro)
            non_obfuscated_auc_macros.append(non_obfuscated_auc_macro)
            obfuscated_auc_micros.append(obfuscated_auc_micro)
            obfuscated_auc_macros.append(obfuscated_auc_macro)
            
        # Evaluated on test set without obfuscation and with k% obfuscation
        results = np.array([options.TEST_PERCENTAGES, non_obfuscated_losses, obfuscated_losses, non_obfuscated_accuracies, obfuscated_accuracies, non_obfuscated_auc_micros, non_obfuscated_auc_macros, obfuscated_auc_micros, obfuscated_auc_macros])

        print('-' * 100)
        print("[First row is test percentages]")
        print("[Second row is non_obfuscated_losses]")
        print("[Third row is obfuscated_losses]")
        print("[Fourth row is non_obfuscated_accuracies]")
        print("[Fifth row is obfuscated_accuracies]")

        print("[Sixth row is non_obfuscated_auc_micros]")
        print("[Seventh row is non_obfuscated_auc_macros]")
        print("[Eigth row is obfuscated_auc_micros]")
        print("[Ninth row is obfuscated_auc_macros]")

        print(results)
        if k < 1:
            save_location = options.results_folder + "/{}_inference_NN_{:.2f}%k_obfuscation.out".format(options.inference_target, k)

        else:
            save_location = options.results_folder + "/{}_inference_NN_{:.2f}k_obfuscation.out".format(options.inference_target, k)
        print("Saved at " + save_location)
        np.savetxt(save_location, results)

