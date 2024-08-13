"""
In this script, we will explore feature engineering and Polynomial Regression which allows us to use the machinery of
linear regression to fit very complicated, even very non-linear functions.
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)


"""
Polynomial Features
Let's try using what we know so far to fit a non-linear curve. We'll start with a simple quadratic:  y = 1+x^2

"""
# Create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations = 1000, alpha = 1e-2)
plt.scatter(x, y, marker = 'x', color = 'red', label = 'Actual Value'); plt.title("No Feature Engineering")
plt.plot(x, X@model_w + model_b, label = 'Predicted Value'); plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()

"""
From the figure, we can notice that, it's not a great fit. What is needed is something like a second degree equation, or a polynomial feature. 
To accomplish this, we can modify the input data to engineer the needed features. 
If you swap the original data with a version that squares the 𝑥 value, then we can achieve what we need.
"""

# Create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer feature
X = x**2
X = X.reshape(-1, 1) # X should be a 2-D matrix
model_w, model_b = run_gradient_descent_feng(X, y, iterations = 10000, alpha = 1e-5)
plt.scatter(x, y, marker = 'x', color = 'red', label = 'Actual Value'); plt.title("Added x**2 Feature")
plt.plot(x, np.dot(X, model_w) + model_b, label = 'Predicted Value'); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

"""
From the figure, we can notice that, it's near perfect. We can also notice the values of w and b printed right above the graph: w,b 
found by gradient descent: w: [1.], b: 0.0490. 
Gradient descent modified our initial values of w,b to be (1.0,0.049), very close to our target of y = 1 x^2 + 1 . 
If we ran it longer, it could be a better match.
"""

