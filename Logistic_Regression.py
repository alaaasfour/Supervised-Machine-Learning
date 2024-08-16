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

