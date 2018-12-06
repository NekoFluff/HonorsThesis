import tensorflow as tf
from tensorflow import keras
import numpy as np

##############################################################
#  Extract Embeddings
##############################################################
model = keras.models.load_model('./tmp/models/user_item_NN_model_attempt_0.h5')

user_embedded = model.get_layer('user_embedding')
user_weights = user_embedded.get_weights()[0]
print("User embedding: ", user_weights.shape)

#normalize the embeddings
user_weights = user_weights / np.linalg.norm(user_weights, axis = 1).reshape((-1, 1))
print("Normalized User Weight: ", user_weights[0])
print(np.sum(np.square(user_weights[0])))

item_embedded = model.get_layer('item_embedding')
item_embedded = item_embedded.get_weights()[0]
print("Item embedding: ", item_embedded.shape)

#normalize the embeddings
item_embedded = item_embedded / np.linalg.norm(item_embedded, axis = 1).reshape((-1, 1))
print("Normalized Item Weight: ", item_embedded[0])
print(np.sum(np.square(item_embedded[0])))