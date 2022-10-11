from sklearn.datasets import make_classification
import numpy as np
import csv

X,y = make_classification(
    n_samples=1000, 
    n_features = 3,
    n_classes = 5, 
    flip_y=0,
    random_state=17,
    n_redundant = 0,
    n_informative= 3,
    n_clusters_per_class=1)

# print(X.shape)
# print(y.shape)
# print(y)
y = np.array([chr(ord("A") + y[i]) for i in range(len(y))])
y = y.reshape(1000, 1)
X = np.round(np.array(X), decimals=3)
# print(y.shape)
new_rows = np.concatenate((y, X), axis=1)
# print(new_rows.shape)

with open("data/synthetic.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(new_rows)