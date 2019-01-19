import os 
import numpy as np
from matrix_factorization import MF
from data_train_test import load_dataset
import options

def get_rating_predictor_using_training_data(training_data, training_enabled=False, reconstruct=False, skip_already_trained=False):
    # Train by holding out 10 ratings per user
    R = training_data
    mf = MF(R, K=options.latent_matrix_dimension, alpha=0.01, beta=0.01, iterations=options.MF_TRAINING_ITERATIONS)
    folder = "./models/matrix_factorization_[dim{}]/".format(options.latent_matrix_dimension)
    if not reconstruct and os.path.isdir(folder):
        mf.load(folder_location=folder)
        
    # Train if training is enabled
    if training_enabled and not (skip_already_trained and os.path.isdir(folder)):
        mf.train()
        if not os.path.isdir(folder):
            os.mkdir(folder)
        mf.save(folder_location=folder)

    return mf

def get_rating_predictor_using_obscured_data(training_data, modified_user_item_matrix, test_percentage, k_obfuscation, training_enabled=False, reconstruct=False, skip_already_trained=False):
    # Train by holding out same 10 ratings, but include the obfuscation ratings for 10% of the users
    R = training_data # Data with ratings held out
    for (user_id, new_user_vector) in modified_user_item_matrix:
        #print("Diff:", sum(new_user_vector-R[user_id])) # Make sure we are actually modifying the array
        R[user_id] = new_user_vector # Replace user history with obfuscated user history

    # TODO: 10 fold cross validation...?
    mf = MF(R, K=options.latent_matrix_dimension, alpha=0.01, beta=0.01, iterations=options.MF_TRAINING_ITERATIONS)
    MF_inference_folder = "./models/MF_{}_trained_with_{}_rating/".format(options.inference_target, options.average_or_predicted_ratings)
    folder = MF_inference_folder + "/matrix_factorization_with_obfuscation_[dim{}]_[test%{:.2f}]_[k{:.2f}]/".format(options.latent_matrix_dimension, test_percentage, k_obfuscation)
    if not reconstruct and os.path.isdir(folder):
        mf.load(folder_location=folder)
    
    # Train if training is enabled
    if training_enabled and not (skip_already_trained and os.path.isdir(folder)):
        mf.train()
        if not os.path.isdir(MF_inference_folder):
            os.mkdir(MF_inference_folder)
        if not os.path.isdir(folder):
            os.mkdir(folder)
    
        mf.save(folder_location=folder)

    return mf

def view_change_in_results():

    # Compute results with differing values of k
    for precision_recall_k in options.precision_at_k_values:

        dataset = load_dataset(0)

        mf1 = get_rating_predictor_using_training_data(training_data=dataset.MF_training, training_enabled=False)
        mf1_mse = mf1.mse(dataset.MF_testing)
        mf1_mae = mf1.mae(dataset.MF_testing)
        mf1_precisions, mf1_recalls, mf1_F1_list = mf1.precision_and_recall_at_k(dataset.MF_testing, k=precision_recall_k)
        mf1_avg_precision = sum(mf1_precisions)/len(mf1_precisions)
        mf1_avg_recall = sum(mf1_recalls)/len(mf1_recalls)
        mf1_avg_F1 = sum(mf1_F1_list)/len(mf1_F1_list)

        results = []
        results.append([0, 0, mf1_mse, mf1_mae, mf1_avg_precision, mf1_avg_recall, mf1_avg_F1])

        # Each NN will recommend different movies, which will affect how the Matrix Factorization is trained
        for test_percentage in options.TEST_PERCENTAGES: 
            
            dataset = load_dataset(test_percentage)
            (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids) = dataset.get_training_testing_for_NN()
            model, categorical_movies = load_NN_and_movie_lists(get_NN_model_location(test_percentage))

            # For every k value (obfuscation percentage), generate the rmse 
            for k in options.k_values:
                print("Retrieving MF for k: {} and test_percentage: {:.2f}".format(k, test_percentage))   
                _, _, _, _, _, _, _, _, modified_user_item_matrix = test_NN(model, test_ratings, test_labels, test_user_ids, categorical_movies, test_percentage, k)

                mf2 = get_rating_predictor_using_obscured_data(dataset.MF_training, modified_user_item_matrix=modified_user_item_matrix, test_percentage=test_percentage, k_obfuscation=k, training_enabled=False)
                mf2_mse = mf2.mse(dataset.MF_testing)
                mf2_mae = mf2.mae(dataset.MF_testing)

                mf2_precisions, mf2_recalls, mf2_F1_list = mf2.precision_and_recall_at_k(dataset.MF_testing, k=precision_recall_k)
                mf2_avg_precision = sum(mf2_precisions)/len(mf2_precisions)
                mf2_avg_recall = sum(mf2_recalls)/len(mf2_recalls)
                mf2_avg_F1 = sum(mf2_F1_list)/len(mf2_F1_list)

                print("MF2 RMSE - MF1 RMSE = {}".format(mf2_mse - mf1_mse))    
                print("MF2 MAE - MF1 MAE = {}".format(mf2_mae - mf1_mae))    
                
                results.append([test_percentage, k, mf2_mse, mf2_mae, mf2_avg_precision, mf2_avg_recall, mf2_avg_F1])
                np_results = np.array(results)

            # Delete the model after you're done with it
            del model

        print(np_results)

        save_location = options.MF_results_folder + "/matrix_factorization_recommender_results_k{}.out".format(precision_recall_k)

        print("Saved at Matrix Factorization Factorization Results at " + save_location)
        print("Data is to be read as:")
        print(['test_percentage', 'k (obfuscation percent)', 'mf2_mse', 'mf2_mae', 'mf2_avg_precision', 'mf2_avg_recall', 'mf2_avg_F1', 'mf2_auc_micro', 'mf2_auc_macro'])

        np.savetxt(save_location, np_results)



if __name__ == "__main__":
    # IMPORTANT: Set generate to True if you wish to generate the Matrix Factorizations, otherwise set to false to skip training
    generate = True

    from attribute_obfuscation import test_NN, load_NN_and_movie_lists
    from attribute_inference_NN import get_NN_model_location
    
    if generate:
        get_rating_predictor_using_training_data(load_dataset(0).MF_training, training_enabled=True, skip_already_trained=True)
        
        # Each NN will recommend different movies, which will affect how the Matrix Factorization is trained
        for test_percentage in options.TEST_PERCENTAGES: 
            dataset = load_dataset(test_percentage)

            (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids) = dataset.get_training_testing_for_NN()
            model, categorical_movies = load_NN_and_movie_lists(get_NN_model_location(test_percentage))

            # For every k value (obfuscation percentage), generate the rmse 
            for k in options.k_values:

                print("Retrieving Obfuscated User Item Matrix to train Matrix Factorization Recommender...")
                _, _, _, _, _, _, _, _, modified_user_item_matrix = test_NN(model, test_ratings, test_labels, test_user_ids, categorical_movies, test_percentage, k)
                
                print("\nTraining MF for k: {} and test_percentage: {:.2f}".format(k, test_percentage))
                get_rating_predictor_using_obscured_data(dataset.MF_training, modified_user_item_matrix=modified_user_item_matrix, test_percentage=test_percentage, k_obfuscation=k, training_enabled=True, skip_already_trained=True)


        view_change_in_results()