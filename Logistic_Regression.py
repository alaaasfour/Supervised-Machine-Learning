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
import copy
import math


# Load the dataset
X_train, y_train = load_data("data/ex2data1.txt")
