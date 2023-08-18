from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Make predictions on the testing set
y_test_pred = model.predict(X_test)

# Calculate bias
train_bias = np.mean((y_train - y_train_pred) ** 2)
test_bias = np.mean((y_test - y_test_pred) ** 2)

# Calculate variance
train_variance = np.var(y_train_pred)
test_variance = np.var(y_test_pred)

# Print the bias and variance
print("Training Bias:", train_bias)
print("Testing Bias:", test_bias)
print("Training Variance:", train_variance)
print("Testing Variance:", test_variance)
