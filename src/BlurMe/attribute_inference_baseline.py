import options
from data_train_test import load_dataset
import tensorflow as tf
import numpy as np

###############################################
#  Compile many models, each using their own test percentage
###############################################
if __name__ == "__main__":

    non_obfuscated_baseline_accuracies = []
    obfuscated_baseline_accuracies = []

    for test_percentage in options.TEST_PERCENTAGES:
        dataset = load_dataset(test_percentage)
        (train_ratings, train_labels), (test_ratings, test_labels), (train_user_ids,
                                                                    test_user_ids) = dataset.get_training_testing_for_NN()


        # Step 1: Find the majority class/attribute
        print("Train Labels:", train_labels)
        counts = np.bincount(train_labels)
        majority_attribute = np.argmax(counts)
        print("Majority Attribute:", majority_attribute)

        count2 = np.bincount(dataset.user_genders)
        print("Distribution in Entire Dataset: ", count2)
        # Step 2: Initialize variables to keep track of accuracy
        correct_prediction_count = 0
        total_users = len(test_labels)
        print("Test Label Size: ", total_users)
        # Step 3: Calculate accuracy using test set.
        for label in test_labels:
            #print("{} == {}? -> {}".format(label, majority_attribute, label == majority_attribute))

            if label == majority_attribute: # match!
                correct_prediction_count += 1

        non_obfuscated_accuracy = correct_prediction_count/total_users
        non_obfuscated_baseline_accuracies.append(non_obfuscated_accuracy)

        # Step 4: Perform obfuscation
        # Step 5: Re-calculate accuracy using obfuscated data
        obfuscated_baseline_accuracies.append(non_obfuscated_accuracy) # Because obfuscation does not change the label and the baseline is so naiive as to not look at the data, the result should be the same regardless


    # Evaluated on test set without obfuscation and with k% obfuscation
    results = np.array([options.TEST_PERCENTAGES, non_obfuscated_baseline_accuracies, obfuscated_baseline_accuracies])

    print('-' * 100)
    print("[First row is test percentages]")
    print("[Second row is non_obfuscated_accuracies]")
    print("[Third row is obfuscated_accuracies]")

    print(results)
    save_location = options.results_folder + "/{}_inference_Baseline.out".format(options.inference_target)
    print("Saved at " + save_location)
    np.savetxt(save_location, results)
