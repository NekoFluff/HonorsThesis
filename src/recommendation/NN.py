import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
from datetime import datetime

from Dataset import DataPreproccessor, Dataset
from Logger import default_logger
from options.AllOptions import AllOptions
from options.ModelOptions import ModelOptions
from options.DataOptions import DataOptions


class NN():
    '''TODO: Add comment here
    '''

    # Variables for the embedding layer
    # num_users = 24303
    # num_items = 10672

    def log(self, output):
        '''A simple logging function that pre-appends a [NN] tag to the beginning of any message passed in.
        '''
        default_logger.log("[NN]: " + output)

    def __init__(self, dataset, model_options=AllOptions.ModelOptions, data_options=AllOptions.DataOptions):
        default_logger.log_time()

        # Make sure you are running an adequate version of Tensorflow
        self.log("Tensorflow version: {}".format(tf.__version__))

        # Set variables
        self.review_dataset = dataset
        self.data_options = data_options
        self.model_options = model_options
        self.model = None

        # Get the data
        (self.train_data, self.train_labels), (self.test_data,
                                               self.test_labels) = self.review_dataset.get_training_and_testing()
        self.log("Retrieved training and testing data from the dataset")

        self.create_validation_from_training(
            model_options.num_validation_samples)

        self.log("Initialized NN")

    def __call__(self, checkpoint=None):
        '''Builds/loads a model. Then evaluates it, saves it, and then generates graphs for it.
        '''
        if checkpoint is None:
            self.build_model()
        else:
            self.load_model_from_checkpoint(checkpoint)
        training_history = self.train()

        # Evaluate the trained model
        self.evaluate_model()

        # Save the model
        saved_file_name = self.save_model(training_history)

        # Graph the results from training
        self.generate_graphs(
            training_history,
            self.data_options.graphs_folder_path + saved_file_name)

    def examine_data(self):
        '''Explore Data
        '''
        self.log("First 3 training entries: {}, labels: {}".format(
            len(self.train_data[:3]), len(self.train_labels[:3])))

    def build_model(self, model=None):
        '''Build the model and store in 'self.model'
        '''

        self.log(
            "Implement the build_model() method in a subclass. Store in 'self.model' and return the model")
        return self.model

    def compile_model(self):
        '''Compiles the keras model stored in 'self.model'
        '''
        self.model.compile(optimizer=self.model_options.optimizer,
                           loss=self.model_options.loss_function,
                           metrics=self.model_options.metrics)
        self.log("Compiled the model")
        return self.model

    def create_validation_from_training(self, num_validation_samples):
        '''Create a validation set from the training set
        '''
        self.validation_data = [data[:num_validation_samples]
                                for data in self.train_data]
        self.partial_train_data = [
            data[num_validation_samples:] for data in self.train_data]

        self.validation_labels = self.train_labels[:num_validation_samples]
        self.partial_train_labels = self.train_labels[num_validation_samples:]
        self.log("Created validation set from training set")

    def train(self):
        '''Train the model
        '''

        # Check if a model exists
        if self.model is None:
            self.log("ERROR: Please create a model using build_model() first.")
            return

        # A model exists. Begin training
        default_logger.log_time()
        self.log("Beginning training...")

        callback_list = []

        if self.model_options.checkpoint_enabled:
            checkpoints_callback = keras.callbacks.ModelCheckpoint(self.data_options.checkpoints_folder_path + self.model_options.checkpoint_string,
                                                                   monitor=self.model_options.checkpoint_monitor,
                                                                   verbose=self.model_options.checkpoint_verbose,
                                                                   save_weights_only=True,
                                                                   period=self.model_options.checkpoint_period)
            callback_list.append(checkpoints_callback)

        if self.model_options.early_stopping_enabled:
            early_stopping_callback = keras.callbacks.EarlyStopping(monitor=self.model_options.early_stopping_monitor,
                                                                    min_delta=self.model_options.early_stopping_min_delta,
                                                                    patience=self.model_options.early_stopping_patience,
                                                                    verbose=self.model_options.early_stopping_verbose)
            callback_list.append(early_stopping_callback)

        training_history = self.model.fit(self.partial_train_data,
                                          self.partial_train_labels,
                                          epochs=self.model_options.num_epochs,
                                          batch_size=self.model_options.training_batch_size,
                                          validation_data=(
                                              self.validation_data, self.validation_labels),
                                          verbose=1,
                                          callbacks=callback_list)

        self.log("Finished training at:")
        default_logger.log_time()
        self.log("You can override this training function in a subclass")
        return training_history

    def evaluate_model(self):
        '''Evaluate the model after it has been trained.
        '''
        if self.model is None:
            self.log(
                "ERROR: The model hasn't been created yet. Call build_model() and then train()")
            return

        # Evaluate the test data
        results = self.model.evaluate(self.test_data, self.test_labels)
        self.log("Results (Test Data)\nLoss: {}\nAccuracy: {}".format(
            results[0], results[1]))

        # Just for sanity's sake, let's check the validation data as well
        if self.validation_data is not None and self.validation_labels is not None:
            results = self.model.evaluate(
                self.validation_data, self.validation_labels)
            self.log("Results (Validation Data)\nLoss: {}\nAccuracy: {}".format(
                results[0], results[1]))

    def save_model(self, training_history):
        '''Save the model stored in the 'model' attribute.
        Returns '/date_folder/file_name' without .hd5 extension
        '''

        self.log("Saving model...")
        history_dict = training_history.history

        # acc, val_acc, loss, val_loss
        results = ['[{:.3f}{}]'.format(history_dict[key][-1], key)
                   for key in history_dict.keys()]
        results_joined = '_'.join(results)

        now = datetime.now()
        today_folder = '/{:%Y-%m-%d}/'.format(now)
        if not os.path.exists(self.data_options.models_folder_path + today_folder):
            os.makedirs(self.data_options.models_folder_path + today_folder)

        hour_min_second = '{:%Hh-%Mm-%Ss}'.format(now)
        file_name = '{}_user_item_NN_model_{}'.format(
            hour_min_second, results_joined)
        path = self.data_options.models_folder_path + today_folder + file_name + '.h5'

        self.model.save(path)
        return today_folder + file_name

    def load_model_from_checkpoint(self, model_weights_location):
        '''Loads a model with the weights stored in the 'model_weights_location' file
        '''

        file_location = self.data_options.checkpoints_folder_path + model_weights_location
        self.build_model()
        self.model.load_weights(file_location)
        self.compile_model()
        self.log("Loaded model with weights at {}".format(file_location))

    def load_model(self, model_location):
        '''Loads a model that was saved using save_model().
        
        [model_location example: /datefolder/file_name.h5]
        [model_location example: /2018-12-20/13h-18m.....h5]
        '''
        file_location = self.data_options.models_folder_path + model_location
        self.model = keras.models.load_model(file_location)
        self.compile_model()
        self.log("Loaded model stored at {}".format(file_location))

    def generate_graphs(self, training_history, graph_save_folder_path, show_graphs=False):
        '''Create a graph of accuracy and loss over time
        '''
        import matplotlib.pyplot as plt

        if training_history is None:
            self.log("ERROR: Training history is None")
            return

        if not os.path.exists(graph_save_folder_path):
            os.makedirs(graph_save_folder_path)

        history_dict = training_history.history

        train_keys = list(
            filter(lambda key: 'val_' not in key, history_dict.keys()))
        for key in train_keys:
            training_values = history_dict[key]
            validation_values = history_dict['val_'+key]

            epochs = range(1, len(training_values) + 1)

            # "bo" is for "blue dot"
            plt.plot(epochs, history_dict[key],
                     'bo', label='Training ' + key)
            # b is for "solid blue line"
            plt.plot(
                epochs, history_dict['val_' + key], 'b', label='Validation ' + key)
            plt.title('Training and validation ' + key)
            plt.xlabel('Epochs')
            plt.ylabel(key)
            plt.legend()
            plt.savefig('{}/{}.png'.format(graph_save_folder_path, key))
            if show_graphs:
                plt.show()
            plt.clf()
