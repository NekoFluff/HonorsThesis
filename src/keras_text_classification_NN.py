import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

##############################################################
#  Download IMDB dataset
##############################################################
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

##############################################################
#  Explore Data
##############################################################
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

# Movie reviews may be of different length:
print("Reviews of different length: ", len(train_data[0]), len(train_data[1]))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # UNK = unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Now we can decode a review
print(decode_review(train_data[0]))

##############################################################
#  Preprocess the Data
##############################################################
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"], # Pad with 0s
                                                       padding='post', # Pad to the end of the list
                                                       maxlen=256) # max size

print("New feature tensor length after padding: ", len(train_data[0]), len(train_data[1]))
print(train_data[0])

##############################################################
#  Build the model
##############################################################
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

#[<tf.Tensor 'embedding_input:0' shape=(?, ?) dtype=float32>]
#[<tf.Tensor 'dense_1/Sigmoid:0' shape=(?, 1) dtype=float32>]

# model = keras.Sequential()
# model.add(keras.layers.Embedding(name = "embedding", input_dim = vocab_size, output_dim = 16)) #10000-dimension to 16-dimension (Each word has its own representation)
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

#[<tf.Tensor 'review_input_layer:0' shape=(?, 256) dtype=float32>]
#[<tf.Tensor 'dense_1/Sigmoid:0' shape=(?, 1) dtype=float32>]
review_input_layer = keras.layers.Input(name = 'review_input_layer', shape = [256])
embedding_layer = keras.layers.Embedding(name = "embedding", input_dim = vocab_size, output_dim = 16)(review_input_layer) #10000-dimension to 16-dimension (Each word has its own representation)
average_layer = keras.layers.GlobalAveragePooling1D()(embedding_layer)
dense_relu_layer = keras.layers.Dense(16, activation = tf.nn.relu)(average_layer)
output_layer = keras.layers.Dense(1, activation = tf.nn.sigmoid)(dense_relu_layer)
model = keras.models.Model(inputs = [review_input_layer], outputs = output_layer)

model.summary() # The resulting dimensions are: (batch, sequence, embedding).
# print(model.inputs)
# print(model.outputs)


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

##############################################################
#  Create a validation set
##############################################################
training_samples = 10000
x_val = train_data[:training_samples]
print("X:", x_val[0])
partial_x_train = train_data[training_samples:]

y_val = train_labels[:training_samples]
print("Y:", y_val[0])
partial_y_train = train_labels[training_samples:]

##############################################################
#  Train the model
##############################################################
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

##############################################################
#  Evaluate the model
##############################################################
results = model.evaluate(test_data, test_labels)
print("Results\nLoss: {}\nAccuracy: {}".format(results[0], results[1])) # Approximately 87% accuracy

##############################################################
#  Create a graph of accuracy and loss over time
##############################################################
history_dict = history.history
history_dict.keys() # acc, val_acc, loss, val_loss

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

###################################### LOSS
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

##############################################################
#  Create a graph of accuracy and loss over time
##############################################################
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

