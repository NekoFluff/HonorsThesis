import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
from datetime import datetime

from data_train_test import dataset

###############################################
#  Initialize the variables and retrieve the data
###############################################
TRAINING_BATCH_SIZE = 32
TEST_PERCENTAGES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] # Percentage of users to have in the test set
NUM_ITEMS = 1682
EPOCHS = 10
model_folder = './models/'


def split_data(test_percentage = 0.20):
    '''
    Returns (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids, test_user_ids)
    '''
    NN_TRAINING_TESTING_FOLDER = './NN_train_test_data/{:.2f}_split/'.format(test_percentage)

    # If the data hasn't already been generated before, then create it
    if not os.path.exists(NN_TRAINING_TESTING_FOLDER):
        os.makedirs(NN_TRAINING_TESTING_FOLDER)
        data = dataset.split_training_testing_for_NN(dataset.MF_training_x, dataset.MF_training_y, test_percentage)
        dataset.save_training_and_testing_split_for_NN(NN_TRAINING_TESTING_FOLDER)
    
    # Otherwise load the saved data
    else: 
        dataset.load_training_and_testing_split_for_NN(NN_TRAINING_TESTING_FOLDER)
        data = dataset.get_training_testing_for_NN()

    return data

def get_NN_model_location(test_percentage):
    return model_folder + '/gender_inference_NN_{:.2f}_split.h5'.format(test_percentage)
        

###############################################
#  Build the model
###############################################
def build_model():
    '''Build the model and store in 'self.model'
    Edit this model as you please to obtain better results.
    '''
    # Input layers
    user_input_layer = keras.layers.Input(
        name='user_input_layer', shape=[NUM_ITEMS])

    hidden_layer = keras.layers.Dense(30000,
                                      name='hidden_layer', activation='relu')(user_input_layer)

    # Reshape to be a single number (shape will be (None, 1))
    output_layer = keras.layers.Dense(2, activation='softmax')(hidden_layer)

    model = keras.models.Model(
        inputs=[user_input_layer], outputs=[output_layer])

    # The resulting dimensions are: (batch, sequence, embedding).
    model.summary()
    print("Built the model. Details are above.")

    return model


###############################################
#  Compile many models, each using their own 
###############################################
if __name__ == "__main__":

    for test_percentage in TEST_PERCENTAGES:
        (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids,
                                                                    test_user_ids) = split_data(test_percentage=test_percentage)
        model = build_model()

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
        print("Training Shape:", train_ratings.shape)
        print("Number of labels:", len(train_labels))

        # Observce the BASELINE
        results = model.evaluate(test_ratings, test_labels)

        print('-'*100)
        print("Gender inference results prior to training (Test Data)\nLoss: {}\nAccuracy: {}".format(
            results[0], results[1]))
        print('-'*100)

        # Then TRAIN
        training_history = model.fit(train_ratings, train_labels, epochs=EPOCHS, batch_size=TRAINING_BATCH_SIZE, validation_data=(test_ratings, test_labels), verbose=1, callbacks=[])

        # Then Observe if there was an improvement
        results = model.evaluate(test_ratings, test_labels)
        print('-'*100)
        print("Gender inference results after training (Test Data)\nLoss: {}\nAccuracy: {}".format(
            results[0], results[1]))
        print('-'*100)

        ###############################################
        #  Save the model
        ###############################################
        model_save_path = get_NN_model_location(test_percentage)
        # Save entire model to a HDF5 file
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model.save(model_save_path)


        ###############################################
        #  Load the model (Just to make sure it saved correctly)
        ###############################################
        new_model = keras.models.load_model(model_save_path)
        # new_model.summary()
        new_model.compile(optimizer=tf.train.AdamOptimizer(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        loss, acc = new_model.evaluate(test_ratings, test_labels)
        print("Restored model accuracy: {:5.2f}%".format(100*acc))
