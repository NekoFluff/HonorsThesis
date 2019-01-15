import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# make data matrix (user X items)
K = 10  # K most recent reviews
data_file = 'ml-100k/u.data'
data_matrix = np.zeros((943, 1682), dtype=np.double)
data_user_item_tmp = {}
data_user_item = {}
max_seq_len = 0

# Open up the primary data file.
# Read each line and retrieve the user id, item id, rating, and timestamp
# Create user-item dictionary (not yet sorted by timestamp)
with open(data_file, 'r') as fin:
    for line in fin.readlines():
        usr_id, itm_id, rate, timestmp = line.split('\t')

        usr_id = int(usr_id)
        itm_id = int(itm_id)
        rate = int(rate)
        timestmp = int(timestmp)

        # IMPORTANT: Use user 'rate' instead of 1 for non-binary
        #data_matrix[usr_id - 1, itm_id - 1] = 1
        data_matrix[usr_id - 1, itm_id - 1] = rate

        # build user-item dict (based on timestamp)
        if usr_id - 1 not in data_user_item_tmp:
            data_user_item_tmp[usr_id - 1] = [[itm_id - 1, timestmp]]
        else:
            data_user_item_tmp[usr_id - 1].append([itm_id - 1, timestmp])

# sort each users' items bsaed on timestamp
for key, items in data_user_item_tmp.items():
    sorted_items = sorted(items, key=lambda x: x[1])  # Sort based on timestamp
    data_user_item[key] = [item[0]
                           for item in sorted_items]  # Unwrap and remove timestamps
    if len(sorted_items) > max_seq_len:
        # Keep track of longest sequence (a.k.a highest # of ratings from 1 person)
        max_seq_len = len(sorted_items)

del data_user_item_tmp

# remove some data for testing :/
data_user_item_train = copy.deepcopy(data_user_item)
data_matrix_train = copy.deepcopy(data_matrix)

# remove K items from each user (K most recent rated items)
for user_id in range(data_matrix_train.shape[0]):
    items_rm = data_user_item_train[user_id][-K:]

    # remove K items
    data_user_item_train[user_id] = data_user_item_train[user_id][:-K]

    # update it in user_item matrix
    for item_id in items_rm:
        data_matrix_train[user_id, item_id] = 0

print('Data Loaded | Data Shape (Users, Items):', data_matrix.shape)
print('Max Seq Len:', max_seq_len)

# load other user data -> age, gender ...
user_info = {}

with open('ml-100k/u.user', 'r') as fin:
    for line in fin.readlines():
        user_id, age, gender, occu, zipcode = line.split('|')
        user_info[int(user_id) - 1] = {
            'age': int(age),
            'gender': 0 if gender == 'M' else 1,
            'occupation': occu,
            'zipcode': zipcode
        }
    print('User Info Loaded!')

# define data loader here
class user_item_loader(object):
    def __init__(self, user_item_matrix, user_info):
        self.user_item_matrix = user_item_matrix
        self.user_info = user_info
        self.user_genders = [v['gender'] for k, v in self.user_info.items()]
        ###################################
        # For Matrix Factorization
        self.MF_training_x = self.user_item_matrix
        self.MF_training_y = self.user_genders
        self.MF_testing_x = data_matrix # The original matrix with K more reviews
        self.MF_testing_y = self.user_genders # Should be the same as MF_training_y
        ###################################

    def __len__(self):
        return self.user_item_matrix.shape[0]

    def __getitem__(self, ind):
        # return (user vector, user ID)
        return (self.user_item_matrix[ind, :], self.user_ids[ind])

    def split_training_testing_for_NN(self, training_x, training_y, test_sample_percentage):
        '''Create a testing set from the training set
        '''
        num_test_samples = int(test_sample_percentage*len(training_x))

        #################################################
        # Dataset split the Neural Network
        self.NN_testing_x = training_x[:num_test_samples]
        self.NN_training_x = training_x[num_test_samples:]

        self.NN_testing_y = training_y[:num_test_samples]
        self.NN_training_y = training_y[num_test_samples:]

        self.NN_testing_user_ids = [i for i in range(num_test_samples)]
        self.NN_training_user_ids = [i for i in range(num_test_samples, len(training_x))]
        ###################################################

        return (self.NN_training_x, self.NN_training_y), (self.NN_testing_x, self.NN_testing_y), (self.NN_training_user_ids, self.NN_testing_user_ids)

# Dataset for use in other files
dataset = user_item_loader(data_matrix_train, user_info)