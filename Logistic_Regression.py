"""
In this lab, we will implement logistic regression and apply it to two different datasets.
The main purpose of this exercise is to build a logistic regression model to predict whether a student gets admitted into a university.

Suppose that we are the administrator of a university department, and we want to determine each applicant’s chance of
admission based on their results on two exams.
We have historical data from previous applicants that we can use as a training set for logistic regression.
For each training example, we have the applicant’s scores on two exams and the admissions decision.
Our task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams.
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from public_tests import *
import copy
import math


# Load the dataset
X_train, y_train = load_data("data/ex2data1.txt")

# View the variables
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train is: ", type(X_train))

# View the elements of y_train
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

# Check the dimension of the variables
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# Visualize the data
plot_data(X_train, y_train[:], pos_label = "Admitted", neg_label = "Not Admitted")
# Set y-axis label
plt.ylabel("Exam 2 Score")
# Set x-axis label
plt.xlabel("Exam 1 Score")
plt.legend(loc="upper right")
plt.show()

"""
Exercise 1: Sigmoid Function
We need to implement the sigmoid function as the first step to build the logistic regression.

Args:
    z (ndarray): A scalar, numpy array of any size.

Returns:
    g (ndarray): sigmoid(z), with the same shape as z
"""
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

print("Exercise 1: Sigmoid Function")
print("==========")
# Test the sigmoid function
print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))
sigmoid_test(sigmoid)
print("========================================")

"""
Exercise 2: Cost Function for Logistic Regression
We also need to implement the cost function to build the logistic regression.

Args:
    X : (ndarray Shape (m,n)) data, m examples by n features
    y : (ndarray Shape (m,))  target value 
    w : (ndarray Shape (n,))  values of parameters of the model      
    b : (scalar)              value of bias parameter of the model
    *argv : unused, for compatibility with regularized version below

Returns:
    total_cost : (scalar) cost 
"""

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    loss_sum = 0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j] * X[i][j]
            z_wb += z_wb_ij
        z_wb += b

        f_wb = sigmoid(z_wb)
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        loss_sum += loss

    total_cost = loss_sum / m
    return total_cost

print("Exercise 2: Cost Function for Logistic Regression")
print("==========")
m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))

# UNIT TESTS
compute_cost_test(compute_cost)
print("========================================")

"""
Exercise 3: Gradient for Logistic Regression
We also need to implement the gradient to build the logistic regression.

Args:
    X : (ndarray Shape (m,n)) data, m examples by n features
    y : (ndarray Shape (m,))  target value 
    w : (ndarray Shape (n,))  values of parameters of the model      
    b : (scalar)              value of bias parameter of the model
    *argv : unused, for compatibility with regularized version below

Returns:
    dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w
    dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b
"""

def compute_gradient(X, y, w, b, *argv):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i

        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i][j]
            dj_dw[j] += dj_dw_ij
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

print("Exercise 3: Compute Gradient for Logistic Regression")
print("==========")
# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}')

# Compute and display cost and gradient with non-zero w and b
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test w and b:', dj_db)
print('dj_dw at test w and b:', dj_dw.tolist())

# UNIT TESTS
compute_gradient_test(compute_gradient)
print("========================================")

"""
Exercise 4: Learning Parameters using Gradient Descent
Now, we will find the optimal parameters of a logistic regression model by using gradient descent.
We will perform batch gradient descent to learn theta. Update theta by taking num_iters gradient steps with learning rate alpha

Args:
    X :    (ndarray Shape (m, n) data, m examples by n features
    y :    (ndarray Shape (m,))  target value 
    w_in : (ndarray Shape (n,))  Initial values of parameters of the model
    b_in : (scalar)              Initial value of parameter of the model
    cost_function :              function to compute cost
    gradient_function :          function to compute gradient
    alpha : (float)              Learning rate
    num_iters : (int)            number of iterations to run gradient descent
    lambda_ : (scalar, float)    regularization constant

Returns:
    w : (ndarray Shape (n,)) Updated values of parameters of the model after running gradient descent
    b : (scalar)                Updated value of parameter of the model after running gradient descent
"""

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    # Number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration for graphing
    J_history = []
    w_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, db_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_db
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w_in, b_in, J_history, w_history

print("Exercise 4: Learning Parameters using Gradient Descent")
print("==========")
# Now let's run the gradient descent algorithm above to learn the parameters for our dataset.
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

# Plotting the decision boundary
plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score')
# Set the x-axis label
plt.xlabel('Exam 1 score')
plt.legend(loc = "upper right")
plt.show()
print("========================================")
