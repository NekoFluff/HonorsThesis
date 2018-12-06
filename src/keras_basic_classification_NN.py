# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Each of these are numpy arrays

# Classifiers for each output. 0 = T-shirt/top
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(train_images.shape)
# 60000 x 28 x 28 (60000 images of pixel size 28 by 28)

print(len(train_labels))
# 60000 training labels

print(train_labels)
# Each of the labels

print(test_images.shape)
# 10000 x 28 x 28

print(len(test_labels))
# 10000 labels, 1 for each test image

plt.figure() # Create a new figure
plt.imshow(train_images[0]) # Show the first training image
plt.colorbar() # Show a color bar alongside it
plt.grid(False) # Do not show a grid
plt.show()
# Colorful image!

###############################################
#  Preprocess
###############################################
# Preprocess the image to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10)) # width, height = 10 inches
for i in range(25): # first 25 images
    plt.subplot(5,5,i+1) # subplot with 5 rows, 5 columns, 1 indexed
    plt.xticks([]) # show xticks
    plt.yticks([]) # show yticks
    plt.grid(False) # remove grid
    plt.imshow(train_images[i], cmap=plt.cm.binary) # Show images
    plt.xlabel(class_names[train_labels[i]]) # put label beneath subplot
plt.show()

###############################################
#  Build the model
###############################################
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # First layer flattens images from 28x28 to a 1d-array of 1x784 (doesn't learn anything, just reformats data)
    keras.layers.Dense(128, activation=tf.nn.relu), # Second layer is a hidden layer that "learns". Uses relu as the activation function
    keras.layers.Dense(10, activation=tf.nn.softmax) # Third layer is the output layer
])

###############################################
#  Compile the model (optimizer, loss, and metrics)
###############################################
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###############################################
#  Train the model
###############################################
# Feed the model the training data, with the associatedtraining labels
# Only performs 5 epochs
model.fit(train_images, train_labels, epochs=5)

###############################################
#  Evaluate the model
###############################################
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

###############################################
#  Make predictions
###############################################
predictions = model.predict(test_images)

print(predictions[0])


print("Guessed:", np.argmax(predictions[0]))

print("Actual:", test_labels[0])

###############################################
#  Graphical View
###############################################
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777") # 10 bars
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

###############################################
#  Predict a single image
###############################################
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plt.figure(figsize=(5,5))
plt.subplot(1,1,1)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

plt.show()