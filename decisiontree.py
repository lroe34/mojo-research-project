import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np

timers = np.array([])
accuracies = np.array([])

# Load the iris dataset

iris_df = pd.read_csv('/Users/lukeroe/Documents/GitHub/mojo-research-project/iris/iris.csv')

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('class', axis=1), iris_df['class'])

# Create a decision tree classifier
clf = DecisionTreeClassifier()

for i in range(100):
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    timers = np.append(timers, end_time - start_time)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies = np.append(accuracies, accuracy)


print(f"Average accuracy of the classifier: {np.mean(accuracies)}")
print(f"Average time taken to train the classifier: {np.mean(timers)} seconds")