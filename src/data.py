# -*- coding: utf-8 -*-
"""
    Prepare Input for DREAM 
    Based on the Instartcart Dataset
"""
import os
import pickle
import numpy as np
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pdb

class DataRetriever(object):
    def __init__(self, raw_data_dir, cache_dir):
        self.raw_data_dir = raw_data_dir
        self.cache_dir = cache_dir
    
    def get_reviews(self):
        """
            Get rating information
        """
        ratings = pd.read_csv(self.raw_data_dir + 'Modified_Video_Games_5.csv')

        # Get products
        data = ratings.asin
        values = array(data)
        print(values)

        # Integer encode products
        product_label_encoder = LabelEncoder()
        integer_encoded_products = product_label_encoder.fit_transform(values)
        print(integer_encoded_products)
        ratings['original_asin'] = ratings['asin']
        ratings['asin'] = integer_encoded_products.astype(int)

        # Get reviewers
        data = ratings.reviewerID
        values = array(data)

        # Integer encode reviewers
        reviewer_label_encoder = LabelEncoder()
        integer_encoded_reviewers = reviewer_label_encoder.fit_transform(values)
        print(integer_encoded_reviewers)
        ratings['original_reviewerID'] = ratings['reviewerID']
        ratings['reviewerID'] = integer_encoded_reviewers.astype(int)

        return ratings


    def get_user_reviews(self):
        '''
            get users' prior reviewed orders
        '''
        users_reviews = None
        if os.path.exists(self.cache_dir + 'users_reviews.pkl'):
            with open(self.cache_dir + 'users_reviews.pkl', 'rb') as f:
                users_reviews = pickle.load(f)
        else:
            reviews = self.get_reviews()
            users_reviews = reviews
            # order_products_prior = self.get_orders_items(prior_or_train)
            # users_reviews = pd.merge(order_products_prior, orders[['user_id', 'order_id', 'order_number', 'days_up_to_last']], 
            #             on = ['order_id'], how = 'left')
            with open(self.cache_dir + 'users_reviews.pkl', 'wb') as f:
                pickle.dump(users_reviews, f, pickle.HIGHEST_PROTOCOL)

        return users_reviews

    def get_baskets_reviews(self, prior_or_train, reconstruct = False, none_idx = 10673):
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
            user_reviews = user_reviews.sort_values(['reviewerID', 'unixReviewTime', 'asin', 'overall'], ascending = True)

            print("Sorted")
            print(user_reviews)

            # Filter data (reviewer and product)
            rid_pid = user_reviews[['reviewerID', 'asin']].drop_duplicates()

            print("RID, PID")
            print(rid_pid)
            user_reviews = user_reviews.groupby(['reviewerID', 'original_asin'])['asin'].apply(list).reset_index()
            print("User Reviews Array Form")
            print(user_reviews)

            user_reviews = user_reviews.groupby(['reviewerID'])['asin'].apply(list).reset_index()
            user_reviews.columns = ['user_id', 'reviewed_products']

            print("User Reviews")
            print(user_reviews)

            with open(filepath, 'wb') as f:
                pickle.dump(user_reviews, f, pickle.HIGHEST_PROTOCOL)
        return user_reviews
         
class Dataset(object):
    '''
        Dataset for user items
    '''
    def __init__(self):
        self.retriever = DataRetriever(raw_data_dir = "./tmp/raw/", cache_dir = './tmp/cache/')
        self.reviews = self.retriever.get_user_reviews()
        print("Dataset Reviews:")
        print(self.reviews)
        print("Unique Users:", len(self.reviews['reviewerID'].unique().tolist()))
        print("Unique Items:", len(self.reviews['asin'].unique().tolist()))

        # Only use a 10th of the data when developing the neural network
        # self.subset = self.reviews[0:int(len(self.reviews)/10)].sample(frac=1)
        self.subset = self.reviews[0:int(len(self.reviews))].sample(frac=1)
        print(type(self.subset))
        print(type(self.reviews))
        
        # Use 80% of subset to train
        training_percentage = 0.80
        self.training = self.subset[0:int(len(self.subset) * training_percentage)]
        self.training['overall'] = [1.0 if rating >= 3 else 0.0 for rating in self.training['overall']]
        #self.training['overall'] = [rating / 5.0 for rating in self.training['overall']]

        self.training_user_x = self.training['reviewerID'].tolist()
        self.training_item_x = self.training['asin'].tolist()
        self.training_x = (self.training_user_x, self.training_item_x) # Pair
        self.training_y = self.training['overall'].astype(float).tolist()

        self.testing = self.subset[int(len(self.subset) * training_percentage):]
        self.testing['overall'] = [1.0 if rating >= 3 else 0.0 for rating in self.testing['overall']]
        #self.testing['overall'] = [rating / 5.0 for rating in self.testing['overall']]

        self.testing_user_x = self.testing['reviewerID'].tolist() 
        self.testing_item_x = self.testing['asin'].tolist()
        self.testing_x = (self.testing_user_x, self.testing_item_x) # Pair
        self.testing_y = self.testing['overall'].astype(float).tolist()

        
    def get_training_and_testing(self):
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
