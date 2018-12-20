import tensorflow as tf

class ModelOptions():
    loss_function = "mse"
    optimizer = tf.train.AdamOptimizer()
    metrics = ['accuracy', 'mae']
    num_epochs = 100
    embedding_output_size = 16

    # How many samples to take out of the training set to use for validation during training
    num_validation_samples = 10000 

