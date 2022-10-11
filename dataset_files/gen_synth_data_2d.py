from cProfile import label
from sklearn.datasets import make_classification
import numpy as np
import csv
import matplotlib.pyplot as plt
from add_noise import add_noise

# n = 2000
# X,y = make_classification(
#     n_samples=n, 
#     n_features = 3,
#     n_classes = 3, 
#     flip_y=0,
#     n_redundant = 0,
#     n_informative= 3,
#     n_clusters_per_class=1, 
#     class_sep= 0.50)


# # y = np.array([chr(ord("A") + y[i]) for i in range(len(y))])
# y = y.reshape(n, 1)
# new_rows = np.concatenate((y, X), axis=1)
# print(np.unique(y))

# plt.scatter(new_rows[:, 1], new_rows[:, 2])
# plt.show()


num_classes = 3
d = 2
n = 500

X = np.random.multivariate_normal(np.array([1, 1]),0.10*np.identity(2),n)
y = np.array([0] * n)

for i in range(1, num_classes):
    X_i = np.random.multivariate_normal(np.array([3*i + 1, 3*i + 1]), 0.10*np.identity(2),n)
    y_i = np.array([i] * n)
    X = np.concatenate((X, X_i), axis=0)
    y = np.concatenate((y, y_i), axis=0)

y = y.reshape(-1, 1).astype(int)

# noise = 0
# label_set = set(np.unique(y))

# #add noise
# for i in range(len(y)):
#     yes = np.random.binomial(1, noise, 1)
#     if yes:
#         new_label_list = list(label_set - set(y[i]))
#         y[i] = np.random.choice(new_label_list)


new_rows = np.concatenate((y, X), axis=1)

indices = np.arange(new_rows.shape[0])
np.random.shuffle(indices)

new_rows = new_rows[indices]
print(new_rows)

plt.scatter(X[:, 0],X[:, 1])
plt.show()

np.savetxt('data/2d_synth.csv', new_rows, delimiter=',')

# with open("data/2d_synth.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(new_rows)

