


# 3. Read the dataset using numpy's built-in function np.genfromtxt()
import numpy as np

dataset_path = "E:\CSE317 (Numerical Methods)\My Course\Lab\Assignment 3\seeds_dataset.txt" #Location of the dataset
delimiter = '\t'
dataset = np.genfromtxt(dataset_path, delimiter=delimiter)

# 4. Shuffle the dataset
np.random.shuffle(dataset)

# 5. Split the dataset into features and labels
X = dataset[:, 0:7]  # Features
y = dataset[:, 7]     # Labels

# 6. Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
X_train, X_test = np.split(X, [train_size])
y_train, y_test = np.split(y, [train_size])

# 7. Find the minimum and maximum values of each feature in the training set
min_values = np.min(X_train, axis=0)
max_values = np.max(X_train, axis=0)

# 8. Normalize the training and test sets
X_train_normalized = (X_train - min_values) / (max_values - min_values)
X_test_normalized = (X_test - min_values) / (max_values - min_values)

# 9. Build the KNN classifier with automatic selection of optimal 'k'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_values = list(range(1, 21))  # Consider k values from 1 to 20
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_normalized, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[cv_scores.index(max(cv_scores))]

# 10. Build the KNN classifier with the optimal k value
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_normalized, y_train)

# Predict using the classifier
y_pred = knn.predict(X_test_normalized)

# Count the number of data points in the testing set
num_testing_points = len(X_test_normalized)

# Count the number of correct predictions for each class
correct_predictions_by_class = [0, 0, 0]
for i in range(num_testing_points):
    if y_pred[i] == y_test[i]:
        correct_predictions_by_class[int(y_pred[i]) - 1] += 1

# Print the results
print("Number of data points in the testing set:", num_testing_points)
for i in range(3):
    print(f"Number of correct predictions for class {i+1}: {correct_predictions_by_class[i]}")
