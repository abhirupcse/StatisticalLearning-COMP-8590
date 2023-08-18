import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target
#one-vs-all multi-class classification using the sigmoid




# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Number of classes
num_classes = len(np.unique(y))

# Create an array to store the trained models
models = []

# Train multiple binary logistic regression models (one-vs-all approach)
for i in range(num_classes):
    # Create binary labels for the current class (1) and other classes (0)
    binary_labels = np.where(y_train == i, 1, 0)
    
    # Create an instance of the logistic regression model
    logreg = LogisticRegression()
    
    # Fit the model to the training data
    logreg.fit(X_train, binary_labels)
    
    # Append the trained model to the models array
    models.append(logreg)

# Predict the class probabilities for each test sample
predictions = np.zeros((X_test.shape[0], num_classes))
for i in range(num_classes):
    # Get the predicted probabilities for the current class
    class_probs = models[i].predict_proba(X_test)
    
    # Store the probability of the positive class (class i)
    predictions[:, i] = class_probs[:, 1]

# Convert the probabilities to class labels by selecting the class with the highest probability
predicted_classes = np.argmax(predictions, axis=1)

# Calculate the accuracy of the multi-class classification
accuracy = accuracy_score(y_test, predicted_classes)
print("Accuracy:", accuracy)
