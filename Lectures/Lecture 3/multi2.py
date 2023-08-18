
#one-vs-one multi-class classification using the sigmoid

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Number of classes
num_classes = len(np.unique(y))

# Create an array to store the trained models
models = []

# Train multiple binary logistic regression models (one-vs-one approach)
for i in range(num_classes):
    for j in range(i+1, num_classes):
        # Filter the data for the current two classes
        class_indices = np.logical_or(y_train == i, y_train == j)
        binary_labels = np.where(y_train[class_indices] == i, 1, 0)
        binary_data = X_train[class_indices]
        
        # Create an instance of the logistic regression model
        logreg = LogisticRegression()
        
        # Fit the model to the binary data
        logreg.fit(binary_data, binary_labels)
        
        # Append the trained model to the models array
        models.append((i, j, logreg))

# Predict the class labels using the trained models
predictions = []
for model in models:
    i, j, clf = model
    class_probs = clf.predict_proba(X_test)
    predicted_classes = np.where(class_probs[:, 1] > 0.5, i, j)
    predictions.append(predicted_classes)

# Voting to determine the final class labels
predicted_classes = np.array(predictions).T
final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predicted_classes)

# Calculate the accuracy of the multi-class classification
accuracy = accuracy_score(y_test, final_predictions)
print("Accuracy:", accuracy)
