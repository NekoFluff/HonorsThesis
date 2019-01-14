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
VALIDATION_PERCENTAGE = 0.10
NUM_ITEMS = 1682
EPOCHS = 5


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
        name='user_input_layer', shape=[NUM_ITEMS])

    hidden_layer = keras.layers.Dense(10000,
                                      name='hidden_layer', activation='relu')(user_input_layer)

    # Reshape to be a single number (shape will be (None, 1))
    output_layer = keras.layers.Dense(2, activation='softmax')(hidden_layer)

    model = keras.models.Model(
        inputs=[user_input_layer], outputs=[output_layer])

    # The resulting dimensions are: (batch, sequence, embedding).
    model.summary()
    print("Built the model. Details are above.")

    return model


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
results = model.evaluate(validation_ratings, validation_labels)
print('-'*100)
print("Results prior to training (Validation Data)\nLoss: {}\nAccuracy: {}".format(
    results[0], results[1]))
print('-'*100)

# Then TRAIN
training_history = model.fit(train_ratings, train_labels, epochs=EPOCHS, batch_size=TRAINING_BATCH_SIZE, validation_data=(validation_ratings, validation_labels), verbose=1, callbacks=[])

# Then Observe if there was an improvement
results = model.evaluate(validation_ratings, validation_labels)
print('-'*100)
print("Results after training (Validation Data)\nLoss: {}\nAccuracy: {}".format(
    results[0], results[1]))
print('-'*100)

###############################################
#  Save the model
###############################################
# Save entire model to a HDF5 file
model_folder = './models/'
model_save_path = model_folder + '/gender_inference_NN.h5'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model.save(model_save_path)


###############################################
#  Load the model (jJust to make sure it saved correctly)
###############################################
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model(model_save_path)
# new_model.summary()
new_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

loss, acc = new_model.evaluate(validation_ratings, validation_labels)
print("Restored model accuracy: {:5.2f}%".format(100*acc))