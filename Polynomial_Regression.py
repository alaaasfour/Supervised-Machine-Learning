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
From the figure, we can notice that, it's not a great fit. What is needed is something like a quadratic equation, or a polynomial feature. 
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

"""
Selecting Features
In the previous example, we knew that an x^2 term was required. It may not always be obvious which features are required. 
One could add a variety of potential features to try and find the most useful. 
For example, what if we had instead tried a cubic equation as follows:
"""

# Create target data
x = np.arange(0, 20, 1)
y = x**2

# Engineer feature
X = np.c_[x, x**2, x**3]
model_w, model_b = run_gradient_descent_feng(X, y, iterations = 10000, alpha = 1e-7)
plt.scatter(x, y, marker = 'x', c = 'r', label = "Actual Value"); plt.title("x, x**2, x**3 Features")
plt.plot(x, X@model_w + model_b, label = "Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

"""
From the previous example, we can notice that the value of w, [0.08 0.54 0.03] and b is 0.0106.
Gradient descent has emphasized the data that is the best fit to the x^2 data by increasing the 𝑤1 term relative to the others. 
If we were to run for a very long time, it would continue to reduce the impact of the other terms.

** Gradient descent is picking the 'correct' features for us by emphasizing its associated parameter **

- Less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very close to zero, 
the associated feature is not useful in fitting the model to the data.
"""

"""
An Alternate View
Above, polynomial features were chosen based on how well they matched the target data. Another way to think about this is
to note that we are still using linear regression once we have created new features. Given that, the best features will 
be linear relative to the target.
"""

# Create target data
x = np.arange(0, 20, 1)
y = x**2

# Engineer features
X = np.c_[x, x**2, x**3]
X_features = ['x','x^2','x^3']


# In the following figure, it is clear that the x^2  feature mapped against the target value y is linear.
# Linear regression can then easily generate a model using that feature.
fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()

"""
Scaling features
If the data set has features with significantly different scales, one should apply feature scaling to speed gradient descent. 
In the example above, there is x, x^2 and x^3  which will naturally have very different scales. 
Let's apply Z-score normalization to our example.
"""

# Create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization
X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# Now we can try again with a more aggressive value of alpha:
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X)

model_w, model_b = run_gradient_descent_feng(X, y, iterations = 100000, alpha = 1e-1)

# Note, Feature scaling allows this to converge much faster.
# Note again the values of w. The w1 term, which is the x^2 term is the most emphasized. Gradient descent has all but eliminated the x^3 term.
plt.scatter(x, y, marker='x', c='red', label="Actual Value"); plt.title("Normalized x x**2, x**3 Feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


"""
Complex Functions
With feature engineering, even quite complex functions can be modeled as follows:
"""

x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X)

model_w,model_b = run_gradient_descent_feng(X, y, iterations = 1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='red', label="Actual Value"); plt.title("Normalized x x**2, x**3 Feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
