import os 
import numpy as np
from matrix_factorization import MF
from data_train_test import dataset

def get_rating_predictor_using_training_data(training_enabled=False, reconstruct=False):
    # Train by holding out 10 ratings per user
    R = dataset.MF_training_x
    K = 128 # Dimension of latent matrix
    mf = MF(R, K=K, alpha=0.01, beta=0.01, iterations=150)
    folder = "./models/matrix_factorization_K{}/".format(K)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    elif not reconstruct and os.path.isdir(folder):
        mf.load(folder_location=folder)

    # Train if training is enabled
    if training_enabled:
        mf.train()
        mf.save(folder_location=folder)

    print("Matrix Factorization RMSE (with withheld ratings):", mf.mse(R))

    # Test on 10 withheld ratings and calculate RMSE
    print("Matrix Factorization RMSE (including withheld ratings):", mf.mse(dataset.MF_testing_x)) 
    print("Predicted Rating for User 0, Item 0:", mf.get_rating(0,0))
    return mf

def get_rating_predictor_using_obscured_data(training_enabled=False, reconstruct=False):
    # Train by holding out same 10 ratings, but include the obfuscation ratings for 10% of the users
    from gender_obfuscation import modified_user_item_matrix
    R = dataset.MF_training_x # Data with ratings held out
    for (user_id, new_user_vector) in modified_user_item_matrix:
        #print("Diff:", sum(new_user_vector-R[user_id]))

        R[user_id] = new_user_vector # Replace user history with obfuscated user history
    # TODO: 10 fold cross validation...

    K = 128 # Dimension of latent matrix
    mf = MF(R, K=K, alpha=0.01, beta=0.01, iterations=150)
    folder = "./models/matrix_factorization_with_obfuscation_K{}/".format(K)
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

def view_change_in_rmse():
    mf1 = get_rating_predictor_using_training_data(training_enabled=False)
    mf2 = get_rating_predictor_using_obscured_data(training_enabled=False)
    print("-"*100)
    print("Final Result:")
    print("MF2 - MF1 = {}".format(mf2.mse(dataset.MF_testing_x) - mf1.mse(dataset.MF_testing_x)))



if __name__ == "__main__":
    # IMPORTANT: Set generate to True if you wish to generate the Matrix Factorizations

    generate = False
    
    if generate:
        get_rating_predictor_using_training_data(training_enabled=True)
        get_rating_predictor_using_obscured_data(training_enabled=True)

    view_change_in_rmse()