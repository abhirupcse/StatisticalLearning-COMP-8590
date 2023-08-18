from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
breast_cancer = datasets.load_breast_cancer()

# Split the dataset into features (X) and labels (y)
X = breast_cancer.data
y = breast_cancer.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with polynomial kernel
classifier = svm.SVC(kernel='poly')

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
