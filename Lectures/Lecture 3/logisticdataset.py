from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()

# Separate the features (X) and the labels (y)
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an instance of the logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model to the scaled training data
logreg.fit(X_train_scaled, y_train)

# Predict the labels for the scaled test data
y_pred = logreg.predict(X_test_scaled)

# Evaluate the model performance using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
