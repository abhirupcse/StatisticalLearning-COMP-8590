from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create a logistic regression model
model = LogisticRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=10)

# Calculate variance and bias
variance = np.var(scores)
bias = (1 - np.mean(scores))**2

# Print the accuracy scores for each fold
print("Accuracy scores:", scores)

# Print the mean accuracy, variance, and bias
print("Mean accuracy:", np.mean(scores))
print("Variance:", variance)
print("Bias:", bias)
