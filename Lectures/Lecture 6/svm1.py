import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (replace with your own data)
X = np.array([[1, 2], [2, 3], [3, 1], [6, 7], [7, 8], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
classifier = svm.SVC(kernel='linear')

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
