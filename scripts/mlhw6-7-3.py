import numpy as np

X = np.array([[10, 13, -11], [-1, 6, -9], [-8, -9, -9], [-12, 12, 8], [-12, -15, -14], [14, 11, 9]])
y = np.array([-22, -7, -34, 94, -55, 19])
X_test = [-5, -9, 3]
k = 1
distances = np.array([np.linalg.norm(X_test - x_train, ord=1) for x_train in X])
nn = np.argsort(distances)[:k]
y_pred = np.mean(y[nn])
print(y_pred)