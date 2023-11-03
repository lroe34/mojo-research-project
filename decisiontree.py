import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load the iris dataset
iris_df = pd.read_csv('/Users/lukeroe/Documents/GitHub/mojo-research-project/iris/iris.csv')

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('class', axis=1), iris_df['class'])

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Start the timer
start_time = time.time()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Stop the timer
end_time = time.time()

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the time taken to train the classifier
print(f"Time taken to train the classifier: {end_time - start_time} seconds")
