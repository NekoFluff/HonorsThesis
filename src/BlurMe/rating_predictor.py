import os 
import numpy as np
from matrix_factorization import MF
from data_train_test import dataset

k_values = [0.01, 0.05, 0.10, 0.15, 0.20] # IMPORTANT! These need to match with the ones in gender_obfuscation
# TODO: Move these into a different file

def get_rating_predictor_using_training_data(training_enabled=False, reconstruct=False):
    # Train by holding out 10 ratings per user
    R = dataset.MF_training_x
    K = 128 # Dimension of latent matrix
    mf = MF(R, K=K, alpha=0.01, beta=0.01, iterations=150)
    folder = "./models/matrix_factorization_[dim{}]/".format(K)
    if not reconstruct and os.path.isdir(folder):
        mf.load(folder_location=folder)
    elif not os.path.isdir(folder):
        os.mkdir(folder)
        

    # Train if training is enabled
    if training_enabled:
        mf.train()
        mf.save(folder_location=folder)

    print("Matrix Factorization RMSE (with withheld ratings):", mf.mse(R))

    # Test on 10 withheld ratings and calculate RMSE
    print("Matrix Factorization RMSE (including withheld ratings):", mf.mse(dataset.MF_testing_x)) 
    print("Predicted Rating for User 0, Item 0:", mf.get_rating(0,0))
    return mf

def get_rating_predictor_using_obscured_data(modified_user_item_matrix, test_percentage, k_obfuscation, training_enabled=False, reconstruct=False):
    # Train by holding out same 10 ratings, but include the obfuscation ratings for 10% of the users
    R = dataset.MF_training_x # Data with ratings held out
    for (user_id, new_user_vector) in modified_user_item_matrix:
        #print("Diff:", sum(new_user_vector-R[user_id])) # Make sure we are actually modifying the array
        R[user_id] = new_user_vector # Replace user history with obfuscated user history

    # TODO: 10 fold cross validation...?
    K = 128 # Dimension of latent matrix
    mf = MF(R, K=K, alpha=0.01, beta=0.01, iterations=150)
    folder = "./models/matrix_factorization_with_obfuscation_[dim{}]_[test%{:.2f}]_[k{:.2f}]/".format(K, test_percentage, k_obfuscation)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    elif not reconstruct and os.path.isdir(folder):
        mf.load(folder_location=folder)

    # Train if training is enabled
    if training_enabled:
        mf.train()
        mf.save(folder_location=folder)

    print("Matrix Factorization RMSE (with withheld ratings and obfuscation):", mf.mse(R))

    # Test on 10 withheld ratings and calculate RMSE
    print("Matrix Factorization RMSE (including withheld ratings and obfuscation):", mf.mse(dataset.MF_testing_x)) 
    print("Predicted Rating for User 0, Item 0:", mf.get_rating(0,0))
    return mf

def view_change_in_results():

    mf1 = get_rating_predictor_using_training_data(training_enabled=True)
    mf1_mse = mf1.mse(dataset.MF_testing_x)
    mf1_mae = mf1.mae(dataset.MF_testing_x)
    mf1_precisions, mf1_recalls, mf1_F1_list = mf1.precision_and_recall_at_k(dataset.MF_testing_x, k=5)

    # Each NN will recommend different movies, which will affect how the Matrix Factorization is trained
    for test_percentage in TEST_PERCENTAGES: 
        
        (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids) = split_data(test_percentage)
        model, L_M, L_F = load_NN_and_movie_lists(get_NN_model_location(test_percentage))

        # For every k value (obfuscation percentage), generate the rmse 
        for k in k_values:
            print("Retrieving MF for k: {} and test_percentage: {:.2f}".format(k, test_percentage))   
            _, _, _, _, modified_user_item_matrix = test_NN(model, test_ratings, test_user_ids, L_M, L_F, k)

            mf2 = get_rating_predictor_using_obscured_data(modified_user_item_matrix=modified_user_item_matrix, test_percentage=test_percentage, k_obfuscation=k, training_enabled=False)
            mf2_mse = mf2.mse(dataset.MF_testing_x)
            mf2_mae = mf2.mae(dataset.MF_testing_x)
            mf2_precisions, mf2_recalls, mf2_F1_list = mf2.precision_and_recall_at_k(dataset.MF_testing_x, k=5)

            print("MF2 MSE - MF1 MSE = {}".format(mf2_mse - mf1_mse))    
            print("MF2 MAE - MF1 MAE = {}".format(mf2_mae - mf1_mae))    

            
if __name__ == "__main__":
    # IMPORTANT: Set generate to True if you wish to generate the Matrix Factorizations

    generate = True
    
    from gender_obfuscation import test_NN, load_NN_and_movie_lists
    from gender_inference_NN import get_NN_model_location, TEST_PERCENTAGES, split_data

    if generate:
        #get_rating_predictor_using_training_data(training_enabled=True)
         
        # Each NN will recommend different movies, which will affect how the Matrix Factorization is trained
        for test_percentage in TEST_PERCENTAGES: 
            
            (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids) = split_data(test_percentage)
            model, L_M, L_F = load_NN_and_movie_lists(get_NN_model_location(test_percentage))

            # For every k value (obfuscation percentage), generate the rmse 
            for k in k_values:
                
                # TODO: Remove these two lines
                if (test_percentage == 0.05 and (k == 0.01 or k == 0.05)) or (test_percentage == 0.1 and k==0.01):
                    continue


                print("Retrieving Obfuscated User Item Matrix to train Matrix Factorization Recommender...")
                _, _, _, _, modified_user_item_matrix = test_NN(model, test_ratings, test_user_ids, L_M, L_F, k)
                
                print("Training MF for k: {} and test_percentage: {:.2f}".format(k, test_percentage))
                get_rating_predictor_using_obscured_data(modified_user_item_matrix=modified_user_item_matrix, test_percentage=test_percentage, k_obfuscation=k, training_enabled=True)


    view_change_in_results()