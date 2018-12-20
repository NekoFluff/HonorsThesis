
import os
import pickle
import numpy as np
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Logger import Logger, default_logger
from options.AllOptions import AllOptions
import pdb


class DataPreproccessor(object):

    def log(self, output):
        '''A simple logging function the pre-appends a [DataPreprocessor] tag to the beginning of any message passed in.
        '''
        default_logger.log("[DataPreproccessor]: " + output)

    def __init__(self, raw_folder_path, cache_folder_path, csv_file_name):
        default_logger.log_time()
        self.raw_folder_path = raw_folder_path
        self.cache_folder_path = cache_folder_path
        self.target_csv_file = csv_file_name
        self.log("Initialized preprocessor")

    def get_user_reviews(self):
        '''Get user reviews
        '''
        user_reviews = None

        user_reviews_pkl_file_path = self.cache_folder_path + self.target_csv_file + '.pkl'
        if os.path.exists(user_reviews_pkl_file_path):
            with open(user_reviews_pkl_file_path, 'rb') as user_reviews_file:
                self.log("Loaded stored user reviews data at: {}".format(
                    user_reviews_pkl_file_path))
                user_reviews = pickle.load(user_reviews_file)
        else:
            # Log the time of this function call
            default_logger.set_checkpoint()
            default_logger.log_time()
            user_reviews = pd.read_csv(
                self.raw_folder_path + self.target_csv_file + '.csv')

            # Get products
            product_ids = array(user_reviews.asin)
            self.log("First 10 products IDs: {}".format(product_ids[:10]))

            # Integer encode products
            product_label_encoder = LabelEncoder()
            integer_encoded_products = product_label_encoder.fit_transform(
                product_ids)
            self.log("First 10 ENCODED products IDs: {}".format(
                integer_encoded_products[:10]))

            # Reorganize column labels for products
            user_reviews['original_asin'] = user_reviews['asin']
            user_reviews['asin'] = integer_encoded_products.astype(int)

            # Get reviewers
            user_ids = array(user_reviews.reviewerID)
            self.log("First 10 reviewer IDs: {}".format(user_ids[:10]))

            # Integer encode reviewers
            reviewer_label_encoder = LabelEncoder()
            integer_encoded_reviewers = reviewer_label_encoder.fit_transform(
                user_ids)
            self.log("First 10 ENCODED user IDs: {}".format(
                integer_encoded_reviewers[:10]))

            # Reorganize column labls for users
            user_reviews['original_reviewerID'] = user_reviews['reviewerID']
            user_reviews['reviewerID'] = integer_encoded_reviewers.astype(int)

            # Save the user reviews dataframe to a file
            self.log("Saving preprocessed user reviews data...")
            with open(user_reviews_pkl_file_path, 'wb') as user_reviews_file:
                pickle.dump(user_reviews, user_reviews_file,
                            pickle.HIGHEST_PROTOCOL)
            self.log("Saved user reviews data at: {}".format(
                user_reviews_pkl_file_path))

            # Log the time this function call took
            default_logger.log_time()

        return user_reviews

    def get_baskets_reviews(self, prior_or_train, reconstruct=False, none_idx=10673):
        '''
        TAKEN FROM DREAM
        DEPRECIATED UNTIL I GO BACK AND READ THE PAPER AGAIN
        '''
        filepath = self.cache_dir + './reviews_' + prior_or_train + '.pkl'

        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                user_reviews = pickle.load(f)
                print(user_reviews)

        else:
            # Retrieve data and sort
            user_reviews = self.get_user_reviews()
            print("Original")
            print(user_reviews)
            user_reviews = user_reviews.sort_values(
                ['reviewerID', 'unixReviewTime', 'asin', 'overall'], ascending=True)

            print("Sorted")
            print(user_reviews)

            # Filter data (reviewer and product)
            rid_pid = user_reviews[['reviewerID', 'asin']].drop_duplicates()

            print("RID, PID")
            print(rid_pid)
            user_reviews = user_reviews.groupby(['reviewerID', 'original_asin'])[
                'asin'].apply(list).reset_index()
            print("User Reviews Array Form")
            print(user_reviews)

            user_reviews = user_reviews.groupby(
                ['reviewerID'])['asin'].apply(list).reset_index()
            user_reviews.columns = ['user_id', 'reviewed_products']

            print("User Reviews")
            print(user_reviews)

            with open(filepath, 'wb') as f:
                pickle.dump(user_reviews, f, pickle.HIGHEST_PROTOCOL)
        return user_reviews


class Dataset(object):
    '''A dataset object for storing information about the user reviews.
    '''

    def log(output):
        '''A simple logging function the pre-appends a [Dataset] tag to the beginning of any message passed in.
        '''
        default_logger.log("[Dataset]: " + output)

    def __init__(self, subset_percentage=1.0, training_percentage=0.8):
        '''Initalizes the dataset. 

        subset_fraction: A value between 0.0 and 1.0. Determines how much of the review data to use.

        Creates a preprocessor instance and retrieves the stored user reviews which is stored in the 'user_reviews' attribute.
        The initializer will also populate these attributes:
        
        training: Training subset of data
        '''
        self.preprocessor = DataPreproccessor(raw_folder_path=AllOptions.DataOptions.raw_folder_path,
                                              cache_folder_path=AllOptions.DataOptions.cache_folder_path,
                                              csv_file_name='Modified_Video_Games.csv')
        self.user_reviews = self.preprocessor.get_user_reviews()
        self.log("First 10 Dataset Reviews: {}".format(self.user_reviews[:10]))
        self.log("# Unique Users:", len(
            self.user_reviews['reviewerID'].unique().tolist()))
        self.log("# Unique Items:", len(
            self.user_reviews['asin'].unique().tolist()))

        # Only use a fraction of the data to speed up training
        self.subset = self.user_reviews[0:int(
            len(self.user_reviews))].sample(frac=subset_percentage)
        self.log("Subset type: {}".format(type(self.subset)))
        self.log("Reviews type: {}".format(type(self.user_reviews)))

        # Only use training_percentage% of the data for training
        self.training = self.subset[0:int(
            len(self.subset) * training_percentage)]

        # Transform or normalize the ratings for Classification/Regression
        # self.training['overall'] = [1.0 if rating >= 3 else 0.0 for rating in self.training['overall']]
        # self.training['overall'] = [rating / 5.0 for rating in self.training['overall']]

        # Assemble the training input and output based on the training dataset
        self.training_user_x = self.training['reviewerID'].tolist()
        self.training_item_x = self.training['asin'].tolist()
        self.training_x = (self.training_user_x, self.training_item_x)  # Pair
        self.training_y = self.training['overall'].astype(float).tolist()

        # Only use (1-training_percentage)% of the data for testing
        self.testing = self.subset[int(
            len(self.subset) * training_percentage):]

        # Transform or normalize the ratings for Classification/Regression
        #self.testing['overall'] = [1.0 if rating >= 3 else 0.0 for rating in self.testing['overall']]
        #self.testing['overall'] = [rating / 5.0 for rating in self.testing['overall']]

        # Assemble the training input and output based on the training dataset
        self.testing_user_x = self.testing['reviewerID'].tolist()
        self.testing_item_x = self.testing['asin'].tolist()
        self.testing_x = (self.testing_user_x, self.testing_item_x)  # Pair
        self.testing_y = self.testing['overall'].astype(float).tolist()

    def get_training_and_testing(self):
        '''Returns the inputs/outputs (for the neural network) generated by the initialzer.
        '''
        return (self.training_x, self.training_y),  (self.testing_x, self.testing_y)

    # def __getitem__(self, index):
    #     if not constants.FOR_REVIEWS:
    #         '''
    #             return baskets & num_baskets
    #         '''
    #         if self.is_reordered_included is True:
    #             return self.basket[index], self.num_baskets[index], self.user_id[index], self.reorder_basket[index], self.history_item[index]
    #         else:
    #             return self.basket[index], self.num_baskets[index], self.user_id[index]
    #     else:
    #         '''
    #             return reviwed_item & num_baskets
    #         '''
    #         if self.is_reordered_included is True:
    #             return self.basket[index], self.num_baskets[index], self.user_id[index], self.reorder_basket[index], self.history_item[index]
    #         else:
    #             return self.reviewed_products[index], self.num_reviews[index], self.user_id[index]

    # def __len__(self):
    #     return len(self.user_id)


if __name__ == "__main__":
    preprocessor = DataPreproccessor(raw_folder_path=AllOptions.DataOptions.raw_folder_path,
                                     cache_folder_path=AllOptions.DataOptions.cache_folder_path,
                                     csv_file_name='Modified_Video_Games_5')
    user_reviews = preprocessor.get_user_reviews()
