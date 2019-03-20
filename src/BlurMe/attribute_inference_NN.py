import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
from datetime import datetime
import options

from data_train_test import load_dataset



def get_NN_model_location(test_percentage, k_value):
    return options.model_folder + '/{}_inference_NN_{:.2f}_split_{}k_removed.h5'.format(options.inference_target, test_percentage, k_value)

###############################################
#  Build the model
###############################################
def build_model():
    '''Build the model and store in 'self.model'
    Edit this model as you please to obtain better results.
    '''
    # Input layers
    user_input_layer = keras.layers.Input(
        name='user_input_layer', shape=[options.NUM_ITEMS])

    hidden_layer = keras.layers.Dense(options.hidden_layer_size,
                                      name='hidden_layer', activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(user_input_layer)
    dropout_layer = keras.layers.Dropout(0.7)(hidden_layer)
    # Reshape to be a single number (shape will be (None, 1))
    output_layer = keras.layers.Dense(options.NUM_CLASSES, activation='softmax')(dropout_layer)

    model = keras.models.Model(
        inputs=[user_input_layer], outputs=[output_layer])

    # The resulting dimensions are: (batch, sequence, embedding).
    model.summary()
    print("Built the model. Details are above.")

    return model

def auc(y_test, y_score, test_percentage, k):
    ################################################################################################
    # Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    ################################################################################################

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    from scipy import interp
    from itertools import cycle
    from sklearn.preprocessing import label_binarize

    # Binarize the expected output
    if options.NUM_CLASSES == 2:
        y_test = np.array([[0, 1] if i == 1 else [1, 0] for i in y_test])
    else:
        y_test = label_binarize(y_test, classes=[i for i in range(options.NUM_CLASSES)])
    n_classes = y_score.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # TOOD: There is an issue if there are no true positive values for some classes
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i]) # Compute the curve (get false positive rate and true positive rates)
        roc_auc[i] = auc(fpr[i], tpr[i]) # Compute the area under the curve (x-axis is fpr, y-axis is tpr)

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    ##############################################################################
    # Plot of a ROC curve for a specific class
    # plot_class = 0
    # plt.figure()
    lw = 2
    # plt.plot(fpr[plot_class], tpr[plot_class], color='darkorange',
    #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[plot_class])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC (class {})'.format(plot_class))
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.close()

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i]) # Guess y values given fpr (x-values) and tpr(y-values)

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciever Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(options.plots_folder + '{}_{}_ROC_test{:.2f}%_{:.2f}k_obfuscation.png'.format(options.chosen_dataset, options.inference_target, test_percentage, k))
    plt.close()
    # plt.show()
    return roc_auc["micro"], roc_auc["macro"]


def main():
    
    for k in options.k_values:
        for test_percentage in options.TEST_PERCENTAGES:
            (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids,
                                                                        test_user_ids) = load_dataset(test_percentage, k).get_training_testing_for_NN()
            model = build_model()

            ###############################################
            #  Compile the model (optimizer, loss, and metrics)
            ###############################################
            model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss='sparse_categorical_crossentropy',#keras.losses.categorical_crossentropy,
                        metrics=['accuracy'])

            ###############################################
            #  Train the model
            ###############################################
            # Feed the model the training data, with the associated training labels
            print("Training Shape:", train_ratings.shape)
            print("Testing Shape:", test_ratings.shape)

            print("Number of train labels:", len(train_labels))
            print("Number of test labels:", len(test_labels))

            print("First Test Rating: ", test_ratings[0])
            print("First test label:", test_labels[0])
            print("Second Test Rating: ", test_ratings[1])
            print("Second test label:", test_labels[1])
            # Observce the BASELINE
            results = model.evaluate(test_ratings, test_labels)

            print('-'*100)
            print("{} Attribute inference results prior to training (Test Data)\nLoss: {}\nAccuracy: {}".format(
                options.inference_target, results[0], results[1]))
            print('-'*100)

            callback_list = []
            early_stopping_monitor = 'val_loss'
            early_stopping_min_delta = 0
            early_stopping_patience = 10 # Number of epochs with no improvement
            early_stopping_verbose = 1
            early_stopping_callback = keras.callbacks.EarlyStopping(monitor=early_stopping_monitor,
                                                                    min_delta=early_stopping_min_delta,
                                                                    patience=early_stopping_patience,
                                                                    verbose=early_stopping_verbose)
            callback_list.append(early_stopping_callback)
            # Then TRAIN
            training_history = model.fit(train_ratings, train_labels, epochs=options.EPOCHS, batch_size=options.TRAINING_BATCH_SIZE, validation_data=(test_ratings, test_labels), verbose=1, callbacks=callback_list)

            # Then Observe if there was an improvement
            results = model.evaluate(test_ratings, test_labels)
            print('-'*100)
            print("{} Attribute inference results after training (Test Data)\nLoss: {}\nAccuracy: {}".format(
                options.inference_target, results[0], results[1]))
            print('-'*100)

            ###############################################
            #  Save the model
            ###############################################
            model_save_path = get_NN_model_location(test_percentage, k)
            # Save entire model to a HDF5 file
            if not os.path.exists(options.model_folder):
                os.makedirs(options.model_folder)

            model.save(model_save_path)


            ###############################################
            #  Load the model (Just to make sure it saved correctly)
            ###############################################
            new_model = keras.models.load_model(model_save_path)
            # new_model.summary()
            new_model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            loss, acc = new_model.evaluate(test_ratings, test_labels)
            print("Restored model accuracy: {:5.2f}%".format(100*acc))

###############################################
#  Compile many models, each using their own test percentage
###############################################
if __name__ == "__main__":
    main()