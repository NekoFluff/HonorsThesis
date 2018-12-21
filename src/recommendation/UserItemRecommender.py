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

from NN import NN


class UserItemRecommender(NN):
    '''TODO: Add comment here
    '''

    # Variables for the embedding layer
    # num_users = 24303
    # num_items = 10672

    def log(self, output):
        '''A simple logging function that pre-appends a [UserItemRecommender] tag to the beginning of any message passed in.
        '''
        default_logger.log("[UserItemRecommender]: " + output)

    def __init__(self, dataset, model_options=AllOptions.ModelOptions, data_options=AllOptions.DataOptions):
        default_logger.log_time()

        NN.__init__(self, dataset, model_options, data_options)

        # self.num_items = dataset.get_num_items()
        # self.num_users = dataset.get_num_users()
        self.num_users = 24303
        self.num_items = 10672

        # Get the data
        self.log("Initialized UserItemRecommender")

    def __call__(self, checkpoint=None):
        '''Builds/loads a model. Then evaluates it, saves it, and then generates graphs for it.
        '''
        NN.__call__(self, checkpoint)

    def examine_data(self):
        '''Explore Data
        '''
        NN.examine_data(self)

    def build_model(self, model=None):
        '''Build the model and store in 'self.model'
        Edit this model as you please to obtain better results.
        '''
        # Input layers
        user_input_layer = keras.layers.Input(
            name='user_input_layer', shape=[1])
        item_input_layer = keras.layers.Input(
            name='item_input_layer', shape=[1])

        # Embedding
        # Each user as self.num_item(s). self.num_item-dimension to self.model_options.embedding_output_size-dimension (Each user has their own representation)
        user_embedding_layer = keras.layers.Embedding(
            name="user_embedding", input_dim=recommendation_system.num_users, output_dim=self.embedding_output_size)(user_input_layer)
        # Each item as self.num_user(s). self.num_user-dimension to self.model_options.embedding_output_size-dimension (Each item has their own representation)
        item_embedding_layer = keras.layers.Embedding(
            name="item_embedding", input_dim=recommendation_system.num_items, output_dim=self.embedding_output_size)(item_input_layer)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged_layer = keras.layers.Dot(name='dot_product', normalize=False, axes=2)([
            user_embedding_layer, item_embedding_layer])

        # Reshape to be a single number (shape will be (None, 1))
        output_layer = keras.layers.Reshape(target_shape=[1])(merged_layer)

        self.model = keras.models.Model(
            inputs=[user_input_layer, item_input_layer], outputs=output_layer)

        # The resulting dimensions are: (batch, sequence, embedding).
        self.model.summary()
        self.compile_model()
        self.log("Built and compiled the model")

        return self.model

    def compile_model(self):
        '''Compiles the keras model stored in 'self.model'
        '''
        return NN.compile_model(self)

    def create_validation_from_training(self, num_validation_samples):
        '''Create a validation set from the training set
        '''
        NN.create_validation_from_training(self, num_validation_samples)
        n = 250

        self.log("Sample user X: {} item X: {}". format(
            self.validation_data[0][n], self.validation_data[1][n]))
        self.log("Sample rating Y: {} type: {}".format(
            self.validation_labels[n], type(self.validation_labels[n])))
        self.log("Created validation set from training set")

    def train(self):
        '''Train the model
        '''
        return NN.train(self)

    def evaluate_model(self):
        '''Evaluate the model after it has been trained.
        '''
        NN.evaluate_model(self)

        self.log("Evaluating model...")
        self.log("First Test Sample: ({}, {}) -> Expected: {} Guessed: {}".format(
            self.test_data[0][0], self.test_data[1][0], self.test_labels[0], self.model.predict(([self.test_data[0][0]], [self.test_data[1][0]]))[0][0]))

    def save_model(self, training_history):
        '''Save the model stored in the 'model' attribute.
        Returns '/date_folder/file_name' without .hd5 extension
        '''

        return NN.save_model(self, training_history)

    def load_model_from_checkpoint(self, model_weights_location):
        '''Loads a model with the weights stored in the 'model_weights_location' file
        '''
        NN.load_model_from_checkpoint(self, model_weights_location)

    def load_model(self, model_location):
        '''Loads a model that was saved using save_model().
        
        [model_location example: /datefolder/file_name.h5]
        [model_location example: /2018-12-20/13h-18m.....h5]
        '''
        NN.load_model(self, model_location)

    def generate_graphs(self, training_history, graph_save_folder_path, show_graphs=False):
        '''Create a graph of accuracy and loss over time
        '''
        NN.generate_graphs(self, training_history,
                           graph_save_folder_path, show_graphs)

    def get_embeddings(self):
        '''Extracts the user and item embeddings from the model
        '''
        if self.model is None:
            self.log("ERROR: model is None. Build and train a model using build_model() and train_model() or load a model using load_model_from_checkpoint()")
            return

        self.log("Retrieving user and item embeddings...")
        user_embedded_layer = self.model.get_layer('user_embedding')
        user_weights = user_embedded_layer.get_weights()[0]
        self.log("User embedded layer weights shape: {}".format(
            user_weights.shape))

        # Normalize the embeddings
        user_weights = user_weights / \
            np.linalg.norm(user_weights, axis=1).reshape((-1, 1))
        self.log("First 10 normalized user weights: {}".format(
            user_weights[0][:10]))
        self.log("Squared sum of normalized user weights (should equal 1): {}".format(
            np.sum(np.square(user_weights[0]))))

        item_embedded_layer = self.model.get_layer('item_embedding')
        item_weights = item_embedded_layer.get_weights()[0]
        self.log("Item embedded layer weights shape: {}".format(
            user_weights.shape))

        # Normalize the embeddings
        item_weights = item_weights / \
            np.linalg.norm(item_weights, axis=1).reshape((-1, 1))
        self.log("First 10 normalized item weights: {}".format(
            item_weights[0][:10]))
        self.log("Squared sum of normalized item weights (should equal 1): {}".format(
            np.sum(np.square(item_weights[0]))))

        return (user_weights, item_weights)

    def recommend_items_for_user(self, user, num_items=10):
        # Get every user-item pair for a specific user
        input_user_list = [user for i in range(self.num_items)]
        input_item_list = [i for i in range(self.num_items)]
        input_data = (input_user_list, input_item_list)

        predictions = self.model.predict(input_data, batch_size=512)
        predictions_unwrapped = [val[0] for val in predictions]
        dataframe = pd.DataFrame(
            data={'user': input_user_list, 'item': input_item_list, 'rating': predictions_unwrapped})
        dataframe = dataframe.sort_values(by=['rating'], ascending=[False])

        self.log("{} best items for user#{}:\n{}".format(
            num_items, user, dataframe[:num_items]))

        return dataframe


if __name__ == "__main__":
    preprocessor = DataPreproccessor(raw_folder_path=AllOptions.DataOptions.raw_folder_path,
                                     cache_folder_path=AllOptions.DataOptions.cache_folder_path,
                                     csv_file_name='Modified_Video_Games_5')
    dataset = Dataset(preprocessor, reconstruct_files=False)

    user_item_recommender = UserItemRecommender(dataset=dataset)

    create_new_model = False
    if create_new_model == True:
        # This is the equivalent of calling all of the below methods in the if statement
        # user_item_recommender('weights_020_0.73loss.hdf5')

        # Build and train the model
        user_item_recommender.build_model()
        training_history = user_item_recommender.train()

        # Evaluate the trained model
        user_item_recommender.evaluate_model()

        # Save the model
        saved_file_name = user_item_recommender.save_model(training_history)

        # Graph the results from training
        user_item_recommender.generate_graphs(
            training_history, AllOptions.DataOptions.graphs_folder_path + saved_file_name)

    else:
        print("Loading model...")
        # Load model from a checkpoint
        # user_item_recommender.load_model_from_checkpoint('weights_020_0.73loss.hdf5')
        user_item_recommender.load_model(
            '/2018-12-20/13h-44m-53s_user_item_NN_model_[1.851val_loss]_[1.066val_mean_absolute_error]_[0.341loss]_[0.424mean_absolute_error].h5')
        user_item_recommender.evaluate_model()
        default_logger.log_time()

    (user_embeddings, item_embeddings) = user_item_recommender.get_embeddings()
    default_logger.log_time()
    user_item_predictions = user_item_recommender.recommend_items_for_user(546)
    default_logger.log_time()
