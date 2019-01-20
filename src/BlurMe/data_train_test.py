import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import options

# define data loader here
class user_item_loader(object):
    def __init__(self, remove_k):
        '''
        remove_k: How many ratings/reviews to remove from each user
        '''
        self.remove_k = remove_k

        if options.chosen_dataset == 'attriguard':

            attriguard_MF_data_folder = './MF_train_test_attriguard_data/{}k_removed/'.format(remove_k)
            if os.path.exists(attriguard_MF_data_folder): # Use already existing data
                self.MF_training= np.load(attriguard_MF_data_folder+"/MF_training.npy")  
                self.target_attribute = np.load(attriguard_MF_data_folder+"/target_attribute.npy") 
                self.MF_testing = np.load(attriguard_MF_data_folder+"/MF_testing.npy") 
                self.user_item_averages = np.mean(self.MF_training, axis=0) # Column averages

            else: # Data hasn't already been created

                from attriguard_input_data import Data_Class
                X = Data_Class()
                X.input_train_app() # 90% users (app ratings)
                X.input_test_app() # 10% users (app ratings)
                X.input_train_label() # 90% users (city they live in)
                X.input_test_label() # 10% users (city they live in)

                # Combine and train_app and test_app
                #TODO: Uncomment the lower two lines to use entire dataset
                #app_ratings = np.concatenate((X.train_app, X.test_app), axis=0)
                #app_ids = np.concatenate((X.train_app_ids, X.test_app_ids), axis=0)
                app_ratings = X.test_app
                app_ids = np.array(X.test_app_ids)
                app_ratings_K_removed = copy.deepcopy(app_ratings)

                # remove K items from each user (K most recent rated items)
                if remove_k > 0:
                    for user_id in range(app_ids.shape[0]):
                        #print("Removing user_id", user_id)
                        items_rm = app_ids[user_id][-remove_k:]

                        # remove K items
                        app_ids[user_id] = app_ids[user_id][:-remove_k]

                        # update it in app_ratings matrix
                        for item_id in items_rm:
                            app_ratings_K_removed[user_id, item_id] = 0

                print('Data Loaded | Data Shape (Users, Items):', app_ratings.shape)

                ###################################
                # For Matrix Factorization
                # Use set with removed app reviews for MF_training
                self.MF_training = app_ratings_K_removed # Matrix with K missing reviews
                self.user_item_averages = np.mean(self.MF_training, axis=0) # Column averages

                # Set target_attribute to combination of train_label and test_label (cities)
                print("test label: ", X.test_label[0])
                self.target_attribute = np.concatenate((X.train_label, X.test_label), axis=0)

                # Use set with all app reviews for MF_testing
                self.MF_testing = app_ratings # The original matrix with K more reviews
                ###################################
                
                os.makedirs(attriguard_MF_data_folder)
                np.save(attriguard_MF_data_folder+"/MF_training.npy", self.MF_training)    # .npy extension is added if not given
                np.save(attriguard_MF_data_folder+"/target_attribute.npy", self.target_attribute)
                np.save(attriguard_MF_data_folder+"/MF_testing.npy", self.MF_testing)  

        elif options.chosen_dataset == 'movielens':

            # make data matrix (user X items)
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
                items_rm = data_user_item_train[user_id][-remove_k:]

                # remove K items
                data_user_item_train[user_id] = data_user_item_train[user_id][:-remove_k]

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
        else:
            print("CRITICAL ERROR: Please use either attriguard or movielens dataset")

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
def load_dataset(test_percentage, remove_k):
    dataset = user_item_loader(remove_k)

    if options.chosen_dataset == 'movielens':
        NN_TRAINING_TESTING_FOLDER = './NN_train_test_data/{}/{:.2f}_split_with_{}k_ratings_removed/'.format(options.inference_target, test_percentage, remove_k)
    else:
        NN_TRAINING_TESTING_FOLDER = './NN_train_test_data/{}/{:.2f}_split_with_{}k_ratings_removed/'.format('attriguard_city', test_percentage, remove_k)

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
    for k in options.k_values:
        dataset = load_dataset(test_percentage=0.2, remove_k=k)