import tensorflow as tf

class ModelOptions():
    # Different options for training 
    loss_function = "mse"
    optimizer = tf.train.AdamOptimizer()
    metrics = ['mae']#['accuracy', 'mae']
    num_epochs = 100
    embedding_output_size = 16

    # How many samples to take out of the training set to use for validation during training
    num_validation_samples = 10000 

    # Options for the checkpoint callback
    checkpoint_enabled = True
    checkpoint_string = '/weights_{epoch:03d}_{val_loss:.2f}loss.hdf5' # Checkpoint filename 
    checkpoint_monitor = 'val_loss' # We look at the val_loss
    checkpoint_period = 10 # Creates a checkpoint every 10 epochs
    checkpoint_verbose = 1

    # Options for 
    early_stopping_enabled = True
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 0
    early_stopping_patience = 3 # Number of epochs with no improvement
    early_stopping_verbose = 1

    # Not included in the current version of tensorflow
    early_stopping_restore_best_weights = True

