import os
##################################
# Obfuscation options:
##################################
inference_target = 'age' # 'gender', 'age', or 'job'
average_or_predicted_ratings = 'average' #'predicted' or 'average'
chosen_strategy = 'greedy' # 'sampled', 'random', or 'greedy'

##################################
# Folder Location options:
##################################
results_folder = './{}_rating_results/'.format(average_or_predicted_ratings)
model_folder = './models/'
MF_results_folder = results_folder + '/MF_{}'.format(inference_target)

if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

if not os.path.isdir(MF_results_folder):
    os.mkdir(MF_results_folder)

##################################
# Data options:
##################################
NUM_ITEMS = 1682
TEST_PERCENTAGES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] # Percentage of users to have in the test set
#k_values = [0.01, 0.05, 0.10, 0.15, 0.20, 25, 30, 35, 40, 45, 50] # If k < 1, then it is a percentage addition. Otherwise it is a flat addition of movies in order to obfuscate user attributes
k_values = [30, 40, 45]
precision_at_k_values = [35]

NUM_GENDER_CLASSES = 2 # Male and female
NUM_JOB_CLASSES = 21 # Number of classes (This is the output layer sizefor the NN)
NUM_AGE_CLASSES = 3 # Over 45, Under 35, Between 45 and 35

if inference_target == 'gender':
    NUM_CLASSES = NUM_GENDER_CLASSES
elif inference_target == 'job':
    NUM_CLASSES = NUM_JOB_CLASSES
elif inference_target == 'age':
    NUM_CLASSES = NUM_AGE_CLASSES
else:
    NUM_CLASSES = 0

##################################
# Neural Network Training options:
##################################
TRAINING_BATCH_SIZE = 32
EPOCHS = 17

##################################
# Matrix Factorization Training options:
##################################
MF_TRAINING_ITERATIONS = 100
latent_matrix_dimension = 128 # Dimension of latent matrix


