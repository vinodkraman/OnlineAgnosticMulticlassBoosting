from numpy.random import RandomState
import numpy as np

k = 5

W_vec = np.zeros(k)
W_vec[np.random.randint(0,k)] = 1
print(W_vec)

y_vec = np.zeros(k)
y_vec[np.random.randint(0,k)] = 1
print(y_vec)

p = np.random.dirichlet(np.ones(k))
print(p)

gamma = 0.25

q1 = (2*W_vec - np.ones(k))/gamma - (2*y_vec - np.ones(k))
q2 = 2*(W_vec/gamma - y_vec)


test1 = np.dot(p, ((2*W_vec - 1)/gamma - (2*y_vec - 1)))
test2 = np.dot(2*p-1, W_vec/gamma - y_vec)

print(test1, test2)
assert test1 == test2
