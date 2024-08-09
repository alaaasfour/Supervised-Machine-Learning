"""
In this script, we will extend the Linear Regression routines to support multiple features
"""
import copy, math
import numpy as np
import matplotlib.pyplot as plt

"""
We will build a linear regression model using these values so we can then predict the price for other houses. 
For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.
"""
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Matrix X containing the examples. Data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)}")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)}")
print(y_train)

"""
Parameter vector w, b
w is a vector with n elements
    - Each element contains the parameter associated with one feature
    - In our dataset, n is 4
    - Notionally, we draw this as a column vector
b is a scalar parameter
"""
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

"""

"""