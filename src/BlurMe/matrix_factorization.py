import numpy as np
import options
# Code was taken from:
# http://www.albertauyeung.com/post/python-matrix-factorization/

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

    def train(self):
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse(self.R)
            training_process.append((i, mse))
            if (i+1) % 5 == 0:
                print("Iteration: %d ; RMSE = %.4f" % (i+1, mse))

        return training_process

    def mse(self, comparison_matrix):
        """
        A function to compute the total mean square error
        """

        xs, ys = comparison_matrix.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(comparison_matrix[x, y] - predicted[x, y], 2)
        #print("Comparison Matrix:", comparison_matrix)

        if len(xs) > 0:        
            error /= len(xs)
        return np.sqrt(error)

    def mae(self, comparison_matrix):
        """
        A function to compute the total mean square error
        """

        xs, ys = comparison_matrix.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += abs(comparison_matrix[x, y] - predicted[x, y])
        error /= len(xs)
        return error
    
    def precision_and_recall_at_k(self, comparison_matrix, k = 20):
        # Knowledge taken from:
        # https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
        
        # Relevant items are already known in the data set
        # Relevant item: Has a True/Actual rating >= 3.5
        # Irrelevant item: Has a True/Actual rating < 3.5

        # # Recommended items are generated by recommendation algorithm
        # Recommended item: has a predicted rating >= 3.5
        # Not recommended item: Has a predicted rating < 3.5

        # Precision at k is the proportion of recommended items in the top-k set that are relevant
        if options.chosen_dataset == 0:
            rating_bar = 3.5
        else:
            rating_bar = 0.7 # On a scale of 0 to 1

        user_precision_list = []
        user_recall_list = []
        relevant_set = set()
        recommended_set = set()
        predicted = self.full_matrix()

        for user in range(len(comparison_matrix)):

            # Filter out movies without true ratings
            movie_ratings_for_user = [(i,v) for i,v in enumerate(predicted[user]) if comparison_matrix[user][i] != 0]
            # Sort movies by ratings
            movie_ratings_for_user = sorted(movie_ratings_for_user, key=lambda x: x[1], reverse=True)  # Sort based on rating
            print("Predicted Movie Ratings (First 5): ", movie_ratings_for_user[:5])

            # Find k ratings whose PREDICTED rating is >= 3.5
            for i in range(min(k, len(movie_ratings_for_user))):
                if (movie_ratings_for_user[i][1] >= rating_bar):
                    recommended_set.add(movie_ratings_for_user[i][0])
                else:
                    break
            
            # Filter out movies without true ratings
            true_movie_ratings_for_user = [(i,v) for i,v in enumerate(comparison_matrix[user]) if comparison_matrix[user][i] != 0]
       
            # Sort movies by ratings
            true_movie_ratings_for_user = sorted(true_movie_ratings_for_user, key=lambda x: x[1], reverse=True)  # Sort based on rating
            print("True Movie Ratings (First 5): ", true_movie_ratings_for_user[:5])

            # Find k ratings whose TRUE rating is >= 3.5   
            for i in range(len(true_movie_ratings_for_user)):
                if (true_movie_ratings_for_user[i][1] >= rating_bar):
                    relevant_set.add(movie_ratings_for_user[i][0])
                else:
                    break

            # Find intersection
            both_set = relevant_set.intersection(recommended_set)

            if not len(recommended_set) == 0:
                precision = len(both_set)/(len(recommended_set))
            else:
                precision = 0
                
            if not len(relevant_set) == 0:
                recall = len(both_set)/len(relevant_set)
            else:
                recall = 0

            user_precision_list.append(precision)
            user_recall_list.append(recall)
            print("User Precision: ", precision)
            print("User Recall: ", recall)
            print("-"*100)

        F1_list = [2 * (p * r) / (p + r) if p + r > 0 else 0 for p, r in zip(user_precision_list, user_recall_list) ]
        return (user_precision_list, user_recall_list, F1_list)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        #print("Dot: ", self.P[i, :].dot(self.Q[j, :].T))

        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
    
    def save(self, folder_location):
        np.save(folder_location+"/b.npy", self.b)    # .npy extension is added if not given
        np.save(folder_location+"/b_u.npy", self.b_u)
        np.save(folder_location+"/b_i.npy", self.b_i)  
        np.save(folder_location+"/P.npy", self.P)  
        np.save(folder_location+"/Q.npy", self.Q)
         
    def load(self, folder_location):
        self.b = np.load(folder_location+"/b.npy")
        self.b_u = np.load(folder_location+"/b_u.npy")
        self.b_i = np.load(folder_location+"/b_i.npy")
        self.P = np.load(folder_location+"/P.npy")
        self.Q = np.load(folder_location+"/Q.npy")





