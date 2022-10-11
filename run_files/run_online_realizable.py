# from hoeffdingtree import *    
import utils
import numpy as np
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
from add_noise import add_noise
from run_dt import run_dt
from run_agnostic import run_agnostic
from run_realizable import run_realizable
from numpy.random import RandomState
from sklearn.model_selection import KFold

parser.add_argument("--N")
parser.add_argument("--filename")
parser.add_argument("--noise")
parser.add_argument("--seed")
args = parser.parse_args()
d = vars(args)

num_wl = int(d["N"])
input_filename = d["filename"]
noise_rate = float(d["noise"])
seed = int(d["seed"])
print("seed", seed)

filename = add_noise(input_filename, noise_rate)

gammas_A = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
gammas_R = [0.5, 0.3, 0.1, 0.05, 0.01, 0.001]

print("filename", filename)
print("noise_rate", noise_rate)
class_index = 0

N = utils.get_num_instances(filename)
rows = utils.get_rows(filename)
rows = np.array(rows)

X = np.array([rows[i][1:] for i in range(len(rows))])
y = np.array([rows[i][0] for i in range(len(rows))])

num_splits = 10
kf = KFold(n_splits=num_splits, shuffle= True, random_state = seed)
wl_baseline = []
agnostic = []
realizable = []

train_indices, test_indices = [], []
for train_index, test_index in kf.split(X):
    train_indices.append(train_index)
    test_indices.append(test_index)

for index in range(num_splits):
    train_index = train_indices[index]
    test_index = test_indices[index]
    agnostic_run_perf = []
    real_run_perf = []

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    result, ov_result = run_dt(filename, class_index, N, 1, X_train, X_test, y_train, y_test)
    wl_baseline.append(result)
    print("basline", result)

    for gamma_index in range(len(gammas_A)):
        # result_A, ov_result_A = run_agnostic(gammas_A[gamma_index], 
        # filename, class_index, N, num_wl, 1, 
        # X_train, X_test, y_train, y_test)

        # agnostic_run_perf.append((result_A, gammas_A[gamma_index]))
        # print("agnostic", (result_A, gammas_A[gamma_index]))

        result_R, ov_result_R = run_realizable(gammas_R[gamma_index], 
        filename, class_index, N, num_wl, 1, 
        X_train, X_test, y_train, y_test)

        real_run_perf.append((result_R, gammas_R[gamma_index]))
        print("realizable", (result_R, gammas_R[gamma_index]))

    # print("agnostic", max(agnostic_run_perf))
    print("realizable", max(real_run_perf))
    # agnostic.append(max(agnostic_run_perf))
    realizable.append(max(real_run_perf))

print('baseline', wl_baseline)
# agnostic_ans = [entry[0] for entry in agnostic]
# print("Agnostic", agnostic_ans, np.sum(agnostic_ans)/num_splits)
realizable_ans = [entry[0] for entry in realizable]
print("Realizable", realizable_ans, np.sum(realizable_ans)/num_splits)
