import os
##################################
# Obfuscation options:
##################################
inference_target = 'gender' # 'gender', 'age', or 'job'
average_or_predicted_ratings = 'average' #'predicted' or 'average'
chosen_strategy = 'greedy' # 'sampled', 'random', or 'greedy'
chosen_dataset = 'attriguard' # 'either 'movielens' or attriguard'

if chosen_dataset == 'attriguard':
    inference_target = 'city'

##################################
# Folder Location options:
##################################
results_folder = './{}_rating_results/'.format(average_or_predicted_ratings)
model_folder = './models/'
MF_results_folder = results_folder + '/MF_{}/'.format(inference_target)
plots_folder = results_folder + '/plots/'

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

if not os.path.isdir(MF_results_folder):
    os.mkdir(MF_results_folder)

if not os.path.isdir(plots_folder):
    os.mkdir(plots_folder)

##################################
# Data options:
##################################
if chosen_dataset == 'movielens':
    NUM_ITEMS = 1682
elif chosen_dataset == 'attriguard':
    NUM_ITEMS = 10000
else:
    NUM_ITEMS = 0

TEST_PERCENTAGES = [0.20] # Percentage of users to have in the test set
#k_values = [0.01, 0.05, 0.10, 0.15, 0.20, 25, 30, 35, 40, 45, 50] # If k < 1, then it is a percentage addition. Otherwise it is a flat addition of movies in order to obfuscate user attributes
k_values = [0, 10, 25, 35, 40, 45]
precision_at_k_values = [35]

NUM_GENDER_CLASSES = 2 # Male and female
NUM_JOB_CLASSES = 21 # Number of classes (This is the output layer sizefor the NN)
NUM_AGE_CLASSES = 3 # Over 45, Under 35, Between 45 and 35
NUM_CITY_CLASSES = 25

if chosen_dataset == 'movielens':
    if inference_target == 'gender':
        NUM_CLASSES = NUM_GENDER_CLASSES
    elif inference_target == 'job':
        NUM_CLASSES = NUM_JOB_CLASSES
    elif inference_target == 'age':
        NUM_CLASSES = NUM_AGE_CLASSES
    else:
        NUM_CLASSES = 0
elif chosen_dataset == 'attriguard':
    NUM_CLASSES = NUM_CITY_CLASSES
else:
    print("ERROR: Please select either attriguard or movielens")
    NUM_CLASSES = 0

##################################
# Neural Network Training options:
##################################
TRAINING_BATCH_SIZE = 100
hidden_layer_size = 101

if chosen_dataset == 'movielens':
    EPOCHS = 50
else:
    EPOCHS = 50


##################################
# Matrix Factorization Training options:
##################################

if chosen_dataset == 'movielens':
    MF_TRAINING_ITERATIONS = 100
else:
    MF_TRAINING_ITERATIONS = 750
latent_matrix_dimension = 128 # Dimension of latent matrix

if __name__ == "__main__":
    import attribute_inference_NN
    attribute_inference_NN.main()

    import attribute_obfuscation
    attribute_obfuscation.main()

    import rating_predictor
    rating_predictor.main()    