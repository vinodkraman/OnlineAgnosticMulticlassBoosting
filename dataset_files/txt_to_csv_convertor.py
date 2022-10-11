import numpy as np

data = np.loadtxt("data/sat.txt", delimiter= " ")
print(data)
y = data[:, -1].reshape(-1, 1)
X = data[:, :-1]

new_data = np.hstack((y, X))
print(y.shape)
print(X.shape)
print(new_data.shape)
print(new_data)
np.savetxt("data/sat.csv", new_data, delimiter=",")