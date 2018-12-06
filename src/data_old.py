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
import constants

import pdb

class BasketConstructor(object):
    '''
        Group products into baskets(type: list)
    '''
    def __init__(self, raw_data_dir, cache_dir):
        self.raw_data_dir = raw_data_dir
        self.cache_dir = cache_dir
    
    ### MINE
    def get_reviews(self):
        """
            Get rating information
        """
        ratings = pd.read_csv(self.raw_data_dir + 'Modified_Video_Games_5.csv')

        

        # define example
        data = ratings.asin
        values = array(data)
        print(values)
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        print(integer_encoded)
        ratings['original_asin'] = ratings['asin']
        ratings['asin'] = integer_encoded.astype(int)
        #ratings['days'] = ratings 
        return ratings

    def get_orders(self):
        '''
            get order context information
        '''
        orders = pd.read_csv(self.raw_data_dir + 'orders.csv')
        orders = orders.fillna(0.0)
        orders['days'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
        orders['days_last'] = orders.groupby(['user_id'])['days'].transform(max)
        orders['days_up_to_last'] = orders['days_last'] - orders['days']
        del orders['days_last']
        del orders['days']
        return orders
    
    def get_orders_items(self, prior_or_train):
        '''
            get detailed information of prior or train orders 
        '''
        orders_products = pd.read_csv(self.raw_data_dir + 'order_products__%s.csv'%prior_or_train)
        return orders_products

    ## MINE
    def get_user_reviews(self, prior_or_train):
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
        
    # Performs left join using orders.csv and order_products__prior.csv using order_id.
    def get_users_orders(self, prior_or_train):
        '''
            get users' prior detailed orders
        '''
        if os.path.exists(self.cache_dir + 'users_orders.pkl'):
            with open(self.cache_dir + 'users_orders.pkl', 'rb') as f:
                users_orders = pickle.load(f)
        else:
            orders = self.get_orders()
            order_products_prior = self.get_orders_items(prior_or_train)
            users_orders = pd.merge(order_products_prior, orders[['user_id', 'order_id', 'order_number', 'days_up_to_last']], 
                        on = ['order_id'], how = 'left')
            with open(self.cache_dir + 'users_orders.pkl', 'wb') as f:
                pickle.dump(users_orders, f, pickle.HIGHEST_PROTOCOL)
        return users_orders
    

    # Performs get_users_orders and strips to contain unique user product pairs
    def get_users_products(self, prior_or_train):
        if constants.FOR_REVIEWS:
            '''
                get users' all purchased products
            '''
            if os.path.exists(self.cache_dir + 'users_products.pkl'):
                with open(self.cache_dir + 'users_products.pkl', 'rb') as f:
                    users_products = pickle.load(f)
            else:
                users_products = self.get_user_reviews(prior_or_train)[['reviewerID', 'asin']].drop_duplicates()
                users_products['asin'] = users_products.asin.astype(int)
                users_products['reviewerID'] = users_products.reviewerID.astype(str)
                users_products = users_products.groupby(['reviewerID'])['asin'].apply(list).reset_index()
                with open(self.cache_dir + 'users_products.pkl', 'wb') as f:
                    pickle.dump(users_products, f, pickle.HIGHEST_PROTOCOL)
            return users_products
        else:
            '''
                get users' all purchased products
            '''
            if os.path.exists(self.cache_dir + 'users_products.pkl'):
                with open(self.cache_dir + 'users_products.pkl', 'rb') as f:
                    users_products = pickle.load(f)
            else:
                users_products = self.get_users_orders(prior_or_train)[['user_id', 'product_id']].drop_duplicates()
                users_products['product_id'] = users_products.product_id.astype(int)
                users_products['user_id'] = users_products.user_id.astype(int)
                users_products = users_products.groupby(['user_id'])['product_id'].apply(list).reset_index()
                with open(self.cache_dir + 'users_products.pkl', 'wb') as f:
                    pickle.dump(users_products, f, pickle.HIGHEST_PROTOCOL)
            return users_products

    def get_items(self, gran):
        '''
            get items' information
            gran = [departments, aisles, products]
        '''
        items = pd.read_csv(self.raw_data_dir + '%s.csv'%gran)
        return items
    
    ## MINE
    def get_baskets_reviews(self, prior_or_train, reconstruct = False, none_idx = 10673):
        filepath = self.cache_dir + './reviews_' + prior_or_train + '.pkl'
       
        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                user_reviews = pickle.load(f)
                print(user_reviews)

        else:          
            # Retrieve data and sort
            user_reviews = self.get_user_reviews(prior_or_train)
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

    def get_baskets(self, prior_or_train, reconstruct = False, reordered = False, none_idx = 49689):
        '''
            get users' baskets
        '''
        if reordered:
            filepath = self.cache_dir + './reorder_basket_' + prior_or_train + '.pkl'
        else:
            filepath = self.cache_dir + './basket_' + prior_or_train + '.pkl'
       
        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                up_basket = pickle.load(f)
                #print(up_basket)

        else:          
            
            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'], ascending = True)
            uid_oid = up[['user_id', 'order_number']].drop_duplicates() # userid and orderid
            up = up[up.reordered == 1][['user_id', 'order_number', 'product_id']] if reordered else up[['user_id', 'order_number', 'product_id']] # Filter data
            up_basket = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index() # If an order number has multiple products, combine them into a list
            up_basket = pd.merge(uid_oid, up_basket, on = ['user_id', 'order_number'], how = 'left') # Merge
            for row in up_basket.loc[up_basket.product_id.isnull(), 'product_id'].index:
                up_basket.at[row, 'product_id'] = [none_idx] # Fill in empty baskets with [none]
            print("UP_BASKET BEFORE:")
            print(up_basket)

            up_basket = up_basket.sort_values(['user_id', 'order_number'], ascending = True).groupby(['user_id'])['product_id'].apply(list).reset_index()
            
            print("UP_BASKET:")
            print(up_basket)
            up_basket.columns = ['user_id', 'reorder_basket'] if reordered else ['user_id', 'basket']
            #pdb.set_trace()
            with open(filepath, 'wb') as f:
                pickle.dump(up_basket, f, pickle.HIGHEST_PROTOCOL)
                
        return up_basket
         
        
    def get_item_history(self, prior_or_train, reconstruct = False, none_idx = 49689):
        filepath = self.cache_dir + './item_history_' + prior_or_train + '.pkl'
        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                item_history = pickle.load(f)
        else:

            # # Retrieve data and sort
            # user_reviews = self.get_user_reviews(prior_or_train)
            # print("Original")
            # print(user_reviews)
            # user_reviews = user_reviews.sort_values(['reviewerID', 'unixReviewTime', 'asin', 'overall'], ascending = True)

            # print("Sorted")
            # print(user_reviews)

            # # Filter data (reviewer and product)
            # rid_pid = user_reviews[['reviewerID', 'asin']].drop_duplicates()

            # print("RID, PID")
            # print(rid_pid)

            # user_reviews = user_reviews.groupby(['reviewerID'])['asin'].apply(list).reset_index()
            # user_reviews.columns = ['user_id', 'reviewed_products'] if reordered else ['user_id', 'basket']

            # print("User Reviews")
            # print(user_reviews)

            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'], ascending = True)
            item_history = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            item_history.loc[item_history.order_number == 1, 'product_id'] = item_history.loc[item_history.order_number == 1, 'product_id'] + [none_idx]
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending = True)
            print("FIRST")
            print(item_history)

            # accumulate 
            item_history['product_id'] = item_history.groupby(['user_id'])['product_id'].transform(pd.Series.cumsum)
            print("SECOND")
            print(item_history)
            # get unique item list
            item_history['product_id'] = item_history['product_id'].apply(set).apply(list)
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending = True)
            print("THIRD")
            print(item_history)
            # shift each group to make it history
            item_history['product_id'] = item_history.groupby(['user_id'])['product_id'].shift(1)
            for row in item_history.loc[item_history.product_id.isnull(), 'product_id'].index:
                item_history.at[row, 'product_id'] = [none_idx]
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending = True).groupby(['user_id'])['product_id'].apply(list).reset_index()
            item_history.columns = ['user_id', 'history_items']
            print("FINAL")
            print(item_history)
            with open(filepath, 'wb') as f:
                pickle.dump(item_history, f, pickle.HIGHEST_PROTOCOL)
        return item_history 

class ReviewDataset(object):
    '''
        Dataset prepared from user-reviews
    '''
    def __init__(self, user_reviews, up_r_basket = None, up_his = None):
        if (up_r_basket is not None) and (up_his is not None):
            self.is_reordered_included = True
        else:
            self.is_reordered_included = False

        self.user_id = list(up_basket.user_id)
        self.basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.basket)]

        if self.is_reordered_included is True:
            up_basket = pd.merge(up_basket, up_r_basket, on = ['user_id'], how = 'left')
            up_basket = pd.merge(up_basket, up_his, on = ['user_id'], how = 'left')
            self.reorder_basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.reorder_basket)]
            self.history_item = [[[int(p) for p in b]for b in u] for u in list(up_basket.history_items)]


class Dataset(object):
    '''
        Dataset prepare from user-basket
    '''
    def __init__(self, up_basket, up_r_basket = None, up_his = None):

        if not constants.FOR_REVIEWS:        
            if (up_r_basket is not None) and (up_his is not None):
                self.is_reordered_included = True
            else:
                self.is_reordered_included = False

            up_basket['num_baskets'] = up_basket.basket.apply(len)
            self.user_id = list(up_basket.user_id)
            self.num_baskets = [int(n) for n in list(up_basket.num_baskets)]    
            self.basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.basket)]

            if self.is_reordered_included is True:
                up_basket = pd.merge(up_basket, up_r_basket, on = ['user_id'], how = 'left')
                up_basket = pd.merge(up_basket, up_his, on = ['user_id'], how = 'left')
                self.reorder_basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.reorder_basket)]
                self.history_item = [[[int(p) for p in b]for b in u] for u in list(up_basket.history_items)]
        else:
            if (up_r_basket is not None) and (up_his is not None):
                self.is_reordered_included = True
            else:
                self.is_reordered_included = False


            up_basket['num_reviews'] = up_basket.reviewed_products.apply(len)

            self.user_id = list(up_basket.user_id)
            self.num_reviews = [int(n) for n in list(up_basket.num_reviews)]    
            #print(self.num_reviews)

            # For product in basket. For basket in baskets list. For baskets list in users.
            print("DATASET LENGTH: ", len(self.user_id))
            self.reviewed_products = [[[int(p) for p in b]for b in u] for u in list(up_basket.reviewed_products)]

            if self.is_reordered_included is True:
                up_basket = pd.merge(up_basket, up_r_basket, on = ['user_id'], how = 'left')
                up_basket = pd.merge(up_basket, up_his, on = ['user_id'], how = 'left')
                self.reorder_basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.reorder_basket)]
                self.history_item = [[[int(p) for p in b]for b in u] for u in list(up_basket.history_items)]

    def __getitem__(self, index):
        if not constants.FOR_REVIEWS:
            '''
                return baskets & num_baskets
            '''
            if self.is_reordered_included is True:
                return self.basket[index], self.num_baskets[index], self.user_id[index], self.reorder_basket[index], self.history_item[index]
            else:
                return self.basket[index], self.num_baskets[index], self.user_id[index]
        else:
            '''
                return reviwed_item & num_baskets
            '''
            if self.is_reordered_included is True:
                return self.basket[index], self.num_baskets[index], self.user_id[index], self.reorder_basket[index], self.history_item[index]
            else:
                return self.reviewed_products[index], self.num_reviews[index], self.user_id[index]
    
    def __len__(self):
        return len(self.user_id)      
