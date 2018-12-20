import tensorflow as tf
from tensorflow import keras

class ModelOptions():
    # Different options for training 
    loss_function = "mse"
    optimizer = tf.train.AdamOptimizer()
    metrics = ['mae']#['accuracy', 'mae']
    num_epochs = 100
    embedding_output_size = 32

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

    def generate_model(self, recommendation_system):
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

        return keras.models.Model(
            inputs=[user_input_layer, item_input_layer], outputs=output_layer)

    def generate_overwatch_model(self):
        # Character (Embedding)
        # Avg Damage per game
        # Avg Death per game
        # Avg Accuracy
        # Avg Kills per game
        # Avg Objective Kills per game
        # Avg Healing
        # Avg Objective time
        # TODO: Time played + others
        num_characters = 10

        character_input_layer = keras.layers.Input(
            name='character_input_layer', shape=[1])
        
        # Includes the 7 averages listed above
        stats_input_layer = keras.layers.Input(
            name='stats_input_layer', shape=[7]) 
        
        # Each character 
        character_embedding_layer = keras.layers.Embedding(
            name="character_embedding", input_dim=num_characters, output_dim=self.embedding_output_size)(character_input_layer)
        character_reshape_layer = keras.layers.Reshape(name="character_reshape_layer", target_shape=[self.embedding_output_size])(character_embedding_layer)

        # That stats input (7) -> becomes (128) -> becomes (256)
        stats_dense_layer = keras.layers.Dense(name="stats_dense_layer", units=512)(stats_input_layer)
        stats_dense_layer_2 = keras.layers.Dense(name="stats_dense_layer_2", units=1025)(stats_dense_layer)

        # We then stack the character and stats output and put another dense layer in front (256)
        combined_layer = keras.layers.Concatenate(name="combined_layer", axis=1)([character_reshape_layer, stats_dense_layer_2])
        combined_layer_dense = keras.layers.Dense(name="combined_layer_dense", units=512)(combined_layer)
        combined_layer_dense_2 = keras.layers.Dense(name="combined_layer_dense_2", units=128)(combined_layer_dense)

        # Finally we convert the 256 layer to a single output (your ranking)
        output_layer = keras.layers.Dense(name="output_layer", units=1)(combined_layer_dense_2)

        # Generate the model and return
        return keras.models.Model(
            inputs=[character_input_layer, stats_input_layer], outputs=output_layer)


def create_random_stats(player_count=1000):

    # TODO: Normalize input?
    from random import randint
    players_stats = []
    for player_index in range(player_count):
        stats = []
        for stat in range(7):
            stats.append(randint(1,100)) # 1 to 10 inclusive

        players_stats.append(stats)
    return players_stats
            
def product(arr, noise=0.0):
    # TODO: Normalize output?
    import random

    result = 1
    for x in arr:
        noise = random.uniform(0, noise)
        result += x + noise
    return result


if __name__ == "__main__":
    model_options = ModelOptions()
    model = model_options.generate_overwatch_model()
    model.compile(optimizer=model_options.optimizer,
                  loss=model_options.loss_function,
                  metrics=model_options.metrics)

    callback_list = []
    model.summary()

    # We get the stats for the most played hero of each user
    different_players = 1000
    heros = [1 for i in range(different_players)] # 1000 Ana's
    player_stats = create_random_stats(player_count=different_players) # 1000 different player stats for Ana
    rankings = [product(arr, noise=10.0) for arr in player_stats]

    partial_train_data = (heros, player_stats)
    partial_train_labels = rankings


    different_players = 1000
    heros = [1 for i in range(different_players)] # 1000 Ana's
    player_stats = create_random_stats(player_count=different_players) # 1000 different player stats for Ana
    rankings = [product(arr) for arr in player_stats]

    validation_data = (heros, player_stats)
    validation_labels = rankings
    training_history = model.fit(partial_train_data,
                                      partial_train_labels,
                                      epochs=1000,#model_options.num_epochs,
                                      batch_size=512,
                                      validation_data=(
                                          validation_data, validation_labels),
                                      verbose=1,
                                      callbacks=callback_list)

    print("Predictions:\n{}".format(model.predict((validation_data[:10], validation_labels[:10]))))
        
