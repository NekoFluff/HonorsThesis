import tensorflow as tf
from tensorflow import keras
from Dataset import Dataset
import os
import numpy as np
from datetime import datetime

from Dataset import DataPreproccessor, Dataset
from Logger import default_logger
from options.AllOptions import AllOptions
from options.ModelOptions import ModelOptions
from options.DataOptions import DataOptions

class UserItemRecommender():
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

        # Make sure you are running an adequate version of Tensorflow
        self.log("Tensorflow version: {}".format(tf.__version__))

        # Set variables
        self.review_dataset = dataset
        self.model_save_folder_path = data_options.models_folder_path
        self.checkpoints_folder_path = data_options.checkpoints_folder_path
        self.graphs_folder_path = data_options.graphs_folder_path

        # Set variables from ModelOptions
        self.optimizer = model_options.optimizer
        self.loss_function = model_options.loss_function
        self.metrics = model_options.metrics
        self.num_epochs = model_options.num_epochs
        self.embedding_output_size = model_options.embedding_output_size

        # self.num_items = dataset.get_num_items()
        # self.num_users = dataset.get_num_users()
        self.num_users = 24303
        self.num_items = 10672
        self.model = None

        # Get the data
        (self.train_data, self.train_labels), (self.test_data,
                                               self.test_labels) = self.review_dataset.get_training_and_testing()
        self.log("Retrieved training and testing data from the dataset")
        
        self.create_validation_from_training(model_options.num_validation_samples)

        self.log("Initialized UserItemRecommender")

    def __call__(self, checkpoint=None):
        if checkpoint is None:
            self.build_model()
        else:
            self.load_model_from_checkpoint('weights_020_0.73loss.hdf5')
        training_history = self.train()

        # Evaluate the trained model
        self.evaluate_model()

        # Save the model
        saved_file_name = self.save_model(training_history)

        # Graph the results from training
        self.generate_graphs(
            training_history, 
            self.graphs_folder_path + saved_file_name)
            
    def examine_data(self):
        '''Explore Data
        '''
        self.log("Training entries: {}, labels: {}".format(
            len(self.train_data), len(self.train_labels)))
        self.log("First training sample -> User: {} Item: {} Label: {}".format(
            self.train_data[0][0], self.train_data[1][0]))

        # # Movie reviews may be of different length:
        # print("Reviews of different length: ", len(self.train_data[0]), len(self.train_data[1]))

        # # A dictionary mapping words to an integer index
        # word_index = imdb.get_word_index()

        # # The first indices are reserved
        # word_index = {k:(v+3) for k,v in word_index.items()}
        # word_index["<PAD>"] = 0
        # word_index["<START>"] = 1
        # word_index["<UNK>"] = 2  # UNK = unknown
        # word_index["<UNUSED>"] = 3

        # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        # def decode_review(text):
        #     return ' '.join([reverse_word_index.get(i, '?') for i in text])

        # # Now we can decode a review
        # print(decode_review(self.train_data[0]))

        ##############################################################
        #  Preprocess the Data
        ##############################################################
        # self.train_data = keras.preprocessing.sequence.pad_sequences(self.train_data,
        #                                                         value=word_index["<PAD>"],
        #                                                         padding='post',
        #                                                         maxlen=256)

        # self.test_data = keras.preprocessing.sequence.pad_sequences(self.test_data,
        #                                                        value=word_index["<PAD>"], # Pad with 0s
        #                                                        padding='post', # Pad to the end of the list
        #                                                        maxlen=256) # max size

        # print("New feature tensor length after padding: ", len(self.train_data[0][0]), len(self.train_data[1][0]))
        # print(self.train_data[0])

    def build_model(self):
        '''Build the model
        '''

        # Input layers
        user_input_layer = keras.layers.Input(
            name='user_input_layer', shape=[1])
        item_input_layer = keras.layers.Input(
            name='item_input_layer', shape=[1])

        # Embedding
        # Each user as self.num_item(s). self.num_item-dimension to self.embedding_output_size-dimension (Each user has their own representation)
        user_embedding_layer = keras.layers.Embedding(
            name="user_embedding", input_dim=self.num_users, output_dim=self.embedding_output_size)(user_input_layer)
        # Each item as self.num_user(s). self.num_user-dimension to self.embedding_output_size-dimension (Each item has their own representation)
        item_embedding_layer = keras.layers.Embedding(
            name="item_embedding", input_dim=self.num_items, output_dim=self.embedding_output_size)(item_input_layer)

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
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=self.metrics)
        return self.model

    def create_validation_from_training(self, num_validation_samples):
        ##############################################################
        #  Create a validation set
        ##############################################################
        train_user_data = self.train_data[0]
        train_item_data = self.train_data[1]

        n = 250

        self.validation_data = [train_user_data[:num_validation_samples],
                                train_item_data[:num_validation_samples]]
        self.log("Sample user X: {} item X: {}". format(
            self.validation_data[0][n], self.validation_data[1][n]))
        self.partial_train_data = [train_user_data[num_validation_samples:],
                                   train_item_data[num_validation_samples:]]

        self.validation_labels = self.train_labels[:num_validation_samples]
        self.log("Sample rating Y: {} type: {}".format(
            self.validation_labels[n], type(self.validation_labels[n])))
        self.partial_train_labels = self.train_labels[num_validation_samples:]
        self.log("Created validation set from training set")

    def train(self):
        '''Train the model
        '''

        if self.model is not None:
            default_logger.log_time()
            self.log("Beginning training...")

            checkpoints_callback = keras.callbacks.ModelCheckpoint(self.checkpoints_folder_path + '/weights_{epoch:03d}_{val_loss:.2f}loss.hdf5', monitor='val_loss', verbose=1, save_weights_only=True, period=10)
            training_history = self.model.fit(self.partial_train_data,
                                              self.partial_train_labels,
                                              epochs=self.num_epochs,
                                              batch_size=512,
                                              validation_data=(
                                                  self.validation_data, self.validation_labels),
                                              verbose=1,
                                              callbacks=[checkpoints_callback])

            self.log("Finished training at:")
            default_logger.log_time()
            return training_history
        else:
            self.log("ERROR: Please create a model using build_model() first.")

    def evaluate_model(self):
        '''Evaluate the model after it has been trained.
        '''
        if self.model is None:
            self.log(
                "ERROR: The model hasn't been created yet. Call build_model() and then train()")
        else:
            self.log("First Test Sample: ({}, {}) -> {}".format(
                self.test_data[0][0], self.test_data[1][0], self.test_labels[0]))

            # Evaluate the test data
            results = self.model.evaluate(self.test_data, self.test_labels)
            self.log("Results (Test Data)\nLoss: {}\nAccuracy: {}".format(
                results[0], results[1]))  # Approximately 87% accuracy

            # Just for sanity's sake, let's check the validation data as well
            if self.validation_data is not None and self.validation_labels is not None:
                results = self.model.evaluate(
                    self.validation_data, self.validation_labels)
                self.log("Results (Validation Data)\nLoss: {}\nAccuracy: {}".format(
                    results[0], results[1]))  # Approximately 87% accuracy

    def save_model(self, training_history):
        '''Save the model stored in the 'model' attribute.
        '''

        history_dict = training_history.history

        # acc, val_acc, loss, val_loss
        results = ['[{:.3f}]{}'.format(history_dict[key][0], key)
                   for key in history_dict.keys()]
        results_joined = '_'.join(results)

        now = datetime.now()
        today_folder = '/{:%Y-%m-%d}/'.format(now)
        if not os.path.exists(self.model_save_folder_path + today_folder):
            os.makedirs(self.model_save_folder_path + today_folder)

        hour_min_second = '{:%Hh-%Mm-%Ss}'.format(now)
        file_name = '{}_user_item_NN_model_{}'.format(
            hour_min_second, results_joined)
        path = self.model_save_folder_path + today_folder + file_name + '.h5'

        self.model.save(path)
        return today_folder + file_name
    
    def load_model_from_checkpoint(self, model_weights_location):
        '''Loads a model with the weights stored in the 'model_weights_location' file
        '''
        file_location = self.checkpoints_folder_path + model_weights_location
        self.build_model()
        self.model.load_weights(file_location)
        self.compile_model()
        self.log("Loaded model with weights at {}".format(file_location))


    def generate_graphs(self, training_history, graph_save_folder_path, show_graphs=False):
        '''Create a graph of accuracy and loss over time
        '''
        import matplotlib.pyplot as plt

        if training_history is None:
            self.log("ERROR: Training history is None")
        else:
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

            # TODO: Move this to get_embeddings.py or delete completely
            ##############################################################
            #  Extract Embeddings
            ##############################################################
            # user_embedded=model.get_layer('user_embedding')
            # user_weights=user_embedded.get_weights()[0]
            # print(user_weights.shape)

            # # normalize the embeddings
            # user_weights=user_weights / \
            #     np.linalg.norm(user_weights, axis=1).reshape((-1, 1))
            # print(user_weights[0][:10])
            # print(np.sum(np.square(user_weights[0])))


if __name__ == "__main__":
    preprocessor = DataPreproccessor(raw_folder_path=AllOptions.DataOptions.raw_folder_path,
                                     cache_folder_path=AllOptions.DataOptions.cache_folder_path,
                                     csv_file_name='Modified_Video_Games_5')
    dataset = Dataset(preprocessor)

    user_item_recommender = UserItemRecommender(dataset=dataset)

    # This is the equivalent of calling all of the below methods
    # user_item_recommender()

    # Build the model
    #user_item_recommender.build_model()
    user_item_recommender.load_model_from_checkpoint('weights_020_0.73loss.hdf5')
    training_history = user_item_recommender.train()

    # Evaluate the trained model
    user_item_recommender.evaluate_model()

    # Save the model
    saved_file_name = user_item_recommender.save_model(training_history)

    # Graph the results from training
    user_item_recommender.generate_graphs(
        training_history, AllOptions.DataOptions.graphs_folder_path + saved_file_name)
