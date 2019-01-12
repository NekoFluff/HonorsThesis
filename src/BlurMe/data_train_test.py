import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

# make data matrix (user X items)
K = 10  # percent
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
        data_matrix[usr_id - 1, itm_id - 1] = 1

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

    def __len__(self):
        return self.user_item_matrix.shape[0]

    def __getitem__(self, ind):
        # return (user vector, user ID)
        return (self.user_item_matrix[ind, :], self.user_ids[ind])

    def create_validation_from_training(self, training_x, training_y, num_validation_samples):
        '''Create a validation set from the training set
        '''
        validation_data = training_x[:num_validation_samples]
        partial_train_data = training_x[num_validation_samples:]

        validation_labels = training_y[:num_validation_samples]
        partial_train_labels = training_y[num_validation_samples:]
        return (partial_train_data, partial_train_labels), (validation_data, validation_labels)

    def get_training_and_testing(self):
        '''Returns the inputs/outputs (for the neural network)
        '''
        self.training_x = self.user_item_matrix
        self.training_y = [v['gender'] for k, v in self.user_info.items()]
        self.testing_x = None
        self.testing_y = None

        return (self.training_x, self.training_y),  (self.testing_x, self.testing_y)

TRAINING_BATCH_SIZE = 32
VALIDATION_PERCENTAGE = 0.99
EPOCHS = 5


dataset = user_item_loader(data_matrix_train, user_info)
(train_ratings, train_labels), (test_ratings,
                                test_labels) = dataset.get_training_and_testing()
(train_ratings, train_labels), (validation_ratings, validation_labels) = dataset.create_validation_from_training(
    train_ratings, train_labels, int(VALIDATION_PERCENTAGE*len(train_ratings)))

###############################################
#  Build the model
###############################################


def build_model():
    '''Build the model and store in 'self.model'
    Edit this model as you please to obtain better results.
    '''
    # Input layers
    user_input_layer = keras.layers.Input(
        name='user_input_layer', shape=[1682])

    hidden_layer = keras.layers.Dense(10000,
                                      name='hidden_layer', activation='relu')(user_input_layer)

    # Reshape to be a single number (shape will be (None, 1))
    output_layer = keras.layers.Dense(2, activation='softmax')(hidden_layer)

    model = keras.models.Model(
        inputs=[user_input_layer], outputs=[output_layer])

    # The resulting dimensions are: (batch, sequence, embedding).
    model.summary()
    print("Built the model")

    return model


model = build_model()


# model = keras.Sequential([
#     keras.layers.Dense(1682), # First layer flattens images from 28x28 to a 1d-array of 1x784 (doesn't learn anything, just reformats data)
#     keras.layers.Dense(10000, activation=tf.nn.relu), # Second layer is a hidden layer that "learns". Uses relu as the activation function
#     keras.layers.Dense(2, activation=tf.nn.softmax) # Third layer is the output layer
# ])

###############################################
#  Compile the model (optimizer, loss, and metrics)
###############################################
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###############################################
#  Train the model
###############################################
# Feed the model the training data, with the associated training labels
print(train_ratings.shape)
print(len(train_labels))

# BASELINE
results = model.evaluate(validation_ratings, validation_labels)
print("Results (Validation Data)\nLoss: {}\nAccuracy: {}".format(
    results[0], results[1]))

# TRAIN
training_history = model.fit(train_ratings, train_labels, epochs=EPOCHS, batch_size=TRAINING_BATCH_SIZE, validation_data=(validation_ratings, validation_labels), verbose=1, callbacks=[])
results = model.evaluate(validation_ratings, validation_labels)
print("Results (Validation Data)\nLoss: {}\nAccuracy: {}".format(
    results[0], results[1]))