import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the regularization parameter
C = 0.1

# Train the logistic regression model with L1 regularization
model = LogisticRegression(penalty='l1', C=C, solver='saga', random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate bias
y_pred_train = model.predict(X_train)
bias = np.mean(y_pred_train != y_train)
print("Bias:", bias)

# Calculate variance
variance = np.var(y_pred_train != y_train)
print("Variance:", variance)

# Print the coefficients
print("Coefficients:")
print(model.coef_)
