import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import options


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
    def __init__(self):
        '''
        inference_target: 'age', 'gender', or 'job'. Each option will modify the training_y
        '''
        self.user_item_matrix = data_matrix_train # Without k most recent ratings
        self.user_item_averages = np.mean(self.user_item_matrix, axis=0) # Column averages

        self.user_info = user_info
        self.user_genders = [v['gender'] for k, v in self.user_info.items()]
        self.user_ages = [0 if v['age'] > 45 else 1 if v['age'] < 35 else 2 for k, v in self.user_info.items()] # [Over 45, Below or 35, Over or Equal 35 & Below or Equal 45] - > [0, 1, 2]
        
        # Integer encode the occupations
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        temp = label_encoder.fit_transform([v['occupation'] for k, v in self.user_info.items()])
        self.user_jobs = [int(i) for i in temp]
        # Check transformed data to make sure it correlates correctly
        self.numerical_user_ages = [v['age'] for k, v in self.user_info.items()]
        self.worded_user_jobs = [v['occupation'] for k, v in self.user_info.items()]

        print("Sample Original User Ages:", self.numerical_user_ages[:10])
        print("Sample User Ages (translated for NN):", self.user_ages[:10])
        print("Sample Original User Jobs:", self.worded_user_jobs[:10])
        print("Sample User Jobs (translated for NN):", self.user_jobs[:10])

        print("Max value for age: ", max(self.user_ages))
        print("Max value for jobs: ", max(self.user_jobs))

        ###################################
        # For Matrix Factorization
        self.MF_training = self.user_item_matrix
        self.target_attribute = self.user_jobs if options.inference_target == 'job' else self.user_ages if options.inference_target == 'age' else self.user_genders
        self.MF_testing = data_matrix # The original matrix with K more reviews
        ###################################

    def __len__(self):
        return self.user_item_matrix.shape[0]

    def split_training_testing_for_NN(self, training_x, training_y, test_sample_percentage):
        '''Create a testing set from the training set
        '''
        num_test_samples = int(test_sample_percentage*len(training_x))
        print("Number of test samples:", num_test_samples)
        train_all = [(i, training_x[i], training_y[i]) for i in range(len(training_x))]

        #################################################
        # Dataset split the Neural Network
        np.random.shuffle(train_all)
        test_portion = train_all[:num_test_samples]
        train_portion = train_all[num_test_samples:]

        self.NN_test_user_ids, self.NN_testing_x, self.NN_testing_y = zip(*test_portion) if num_test_samples > 0 else (None, None, None)
        self.NN_test_user_ids, self.NN_testing_x, self.NN_testing_y = (np.array(self.NN_test_user_ids),  np.array(self.NN_testing_x),  np.array(self.NN_testing_y)) if num_test_samples > 0 else (None, None, None)
        
        self.NN_train_user_ids, self.NN_training_x, self.NN_training_y = zip(*train_portion)

        self.NN_train_user_ids, self.NN_training_x, self.NN_training_y =  np.array(self.NN_train_user_ids),  np.array(self.NN_training_x),  np.array(self.NN_training_y)
        ###################################################

        return (self.NN_training_x, self.NN_training_y), (self.NN_testing_x, self.NN_testing_y), (self.NN_train_user_ids, self.NN_test_user_ids)

    def get_training_testing_for_NN(self):
        return (self.NN_training_x, self.NN_training_y), (self.NN_testing_x, self.NN_testing_y), (self.NN_train_user_ids, self.NN_test_user_ids)

    def save_training_and_testing_split_for_NN(self, folder_location):
        np.save(folder_location+"/NN_training_x.npy", self.NN_training_x)    # .npy extension is added if not given
        np.save(folder_location+"/NN_training_y.npy", self.NN_training_y)
        np.save(folder_location+"/NN_testing_x.npy", self.NN_testing_x)  
        np.save(folder_location+"/NN_testing_y.npy", self.NN_testing_y)  
        np.save(folder_location+"/NN_training_user_ids.npy", self.NN_train_user_ids) 
        np.save(folder_location+"/NN_testing_user_ids.npy", self.NN_test_user_ids) 
         
    def load_training_and_testing_split_for_NN(self, folder_location):
        self.NN_training_x = np.load(folder_location+"/NN_training_x.npy")    # .npy extension is added if not given
        self.NN_training_y = np.load(folder_location+"/NN_training_y.npy")
        self.NN_testing_x = np.load(folder_location+"/NN_testing_x.npy")  
        self.NN_testing_y= np.load(folder_location+"/NN_testing_y.npy")  
        self.NN_train_user_ids = np.load(folder_location+"/NN_training_user_ids.npy") 
        self.NN_test_user_ids = np.load(folder_location+"/NN_testing_user_ids.npy") 
         
# Dataset for use in other files
def load_dataset(test_percentage):
    dataset = user_item_loader()

    NN_TRAINING_TESTING_FOLDER = './NN_train_test_data/{}/{:.2f}_split/'.format(options.inference_target, test_percentage)

    # If the data hasn't already been generated before, then create it
    if not os.path.exists(NN_TRAINING_TESTING_FOLDER):
        dataset.split_training_testing_for_NN(dataset.MF_training, dataset.target_attribute, test_percentage)
        os.makedirs(NN_TRAINING_TESTING_FOLDER)

        dataset.save_training_and_testing_split_for_NN(NN_TRAINING_TESTING_FOLDER)
    
    # Otherwise load the saved data
    else: 
        dataset.load_training_and_testing_split_for_NN(NN_TRAINING_TESTING_FOLDER)

    return dataset

if __name__ == "__main__":
    dataset = load_dataset(test_percentage=0.0)