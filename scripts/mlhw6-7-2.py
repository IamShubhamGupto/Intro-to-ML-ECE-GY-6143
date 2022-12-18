import numpy as np

X = np.array([[-8, -2, -3], [-1, 0 , 1], [-6, 5, 12], [12, -4, 7], [4, -1, -14], [-4, 2, 13]])
y = np.array([1, 1, 1, 0, 0, 1])
X_test = [5, -3, 5]
k = 1
distances = np.array([np.linalg.norm(X_test - x_train, ord=2) for x_train in X])
nn = np.argsort(distances)[:k]
y_pred = y[nn]
print(y_pred)