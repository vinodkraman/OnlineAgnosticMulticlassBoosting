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
from batchAgnostic import AgnosticBoost
from batchRealizable import BatchMM
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

parser.add_argument("--T")
parser.add_argument("--filename")
parser.add_argument("--noise")
parser.add_argument("--exp")
# parser.add_argument("--m0")
args = parser.parse_args()
d = vars(args)

T = int(d["T"])
input_filename = d["filename"]
noise_rate = float(d["noise"])
num_exp = int(d["exp"])
# m0_frac = float(d["m0"])

filename = add_noise(input_filename, noise_rate)
rs = RandomState(10)

gammas_A = [0.1, 0.3, 0.5, 0.7, 1.0]
gammas_R = [0.3, 0.1, 0.05, 0.01, 0.001]
m0_values = [0.1, 0.3, 0.5, 0.7, 1.0]

shuffle_seeds = np.random.randint(1, 900, num_exp)
# shuffle_seeds = rs.randint(1, 900, num_exp)
shuffle_seeds = [423,51,172,729,887]
print(shuffle_seeds)

print(filename)
class_index = 0
training_ratio = 0.8

N = utils.get_num_instances(filename)
train_N = int(N*training_ratio)
rows = utils.get_rows(filename)

#Baseline
wl_baseline = []
for index in tqdm(range(num_exp)):

    shuff_rows = utils.shuffle(rows, seed=shuffle_seeds[index])

    train_rows = np.array(shuff_rows[:train_N])
    test_rows = np.array(shuff_rows[train_N:])

    X_train = np.array([train_rows[i][1:] for i in range(len(train_rows))])
    y_train = np.array([train_rows[i][0] for i in range(len(train_rows))])

    X_test = np.array([test_rows[i][1:] for i in range(len(test_rows))])
    y_test = np.array([test_rows[i][0] for i in range(len(test_rows))])

    baseline = DecisionTreeClassifier(max_depth=1)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    accuracy = accuracy_score(y_test, baseline_pred)
    wl_baseline.append(accuracy)

print('baseline', wl_baseline)

best_avg_A = []
for index in tqdm(range(num_exp)):
    wl_perf_A = []
    shuff_rows = utils.shuffle(rows, seed=shuffle_seeds[index])
    train_rows = shuff_rows[:train_N]
    test_rows = shuff_rows[train_N:]
    X_train = np.array([train_rows[i][1:] for i in range(len(train_rows))])
    y_train = np.array([train_rows[i][0] for i in range(len(train_rows))])

    X_test = np.array([test_rows[i][1:] for i in range(len(test_rows))])
    y_test = np.array([test_rows[i][0] for i in range(len(test_rows))])

    m = len(train_rows)
    for gamma_index in range(len(gammas_A)):
        for m0_frac in m0_values:
            m0 = int(m*m0_frac)
            model = AgnosticBoost(m, gamma=gammas_A[gamma_index])
            model.initialize_dataset(filename, class_index, N)
            model.fit(X_train, y_train, T, m0)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            wl_perf_A.append((accuracy, gammas_A[gamma_index], m0_frac))

    print("Agnostic", index, wl_perf_A)
    best_avg_A.append(max(wl_perf_A))
print("Agnostic", best_avg_A)





# #agnostic
# best_avg_R = []
# best_avg_A = []
# for index in tqdm(range(num_exp)):
#     wl_perf_A = []
#     wl_perf_R = []
#     shuff_rows = utils.shuffle(rows, seed=shuffle_seeds[index])
#     train_rows = shuff_rows[:train_N]
#     test_rows = shuff_rows[train_N:]

#     X_train = np.array([train_rows[i][1:] for i in range(len(train_rows))])
#     y_train = np.array([train_rows[i][0] for i in range(len(train_rows))])

#     X_test = np.array([test_rows[i][1:] for i in range(len(test_rows))])
#     y_test = np.array([test_rows[i][0] for i in range(len(test_rows))])

#     m = len(train_rows)
#     m0 = int(m*m0_frac)

#     for gamma_index in range(len(gammas_A)):

#         model = AgnosticBoost(m, gamma=gammas_A[gamma_index])
#         model.initialize_dataset(filename, class_index, N)
#         model.fit(X_train, y_train, T, m0)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         wl_perf_A.append((accuracy, gammas_A[gamma_index]))


#         # model = BatchMM(T, gamma= gammas_R[gamma_index])
#         # model.initialize_dataset(filename, class_index, N)
#         # model.fit(X_train, y_train, T)
#         # print("yes")
#         # y_pred = model.predict(X_test)
#         # accuracy = accuracy_score(y_test, y_pred)
#         # wl_perf_R.append((accuracy, gammas_R[gamma_index]))

#     #take the gamma with best performance
#     print("Agnostic", index, wl_perf_A)
#     best_avg_A.append(max(wl_perf_A))

#     # print("Realizable", index, wl_perf_R)
#     # best_avg_R.append(max(wl_perf_R))

# print("Agnostic", best_avg_A)
# print("Realizable", best_avg_R)


# model = BatchMM(T, gamma= gamma)
# model.initialize_dataset(filename, class_index, N)
# model.fit(X_train, y_train, T)

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

# baseline = DecisionTreeClassifier(max_depth=1)
# baseline.fit(X_train, y_train)
# baseline_pred = baseline.predict(X_test)
# accuracy = accuracy_score(y_test, baseline_pred)
# print(accuracy)

# ov_cnt = 0
# num_exp = 1
# for exp in range(num_exp):
# 	for i in tqdm(range(len(train_rows))):
# 		X = train_rows[i][1:]
# 		Y = train_rows[i][0]
# 		pred = model.predict(X)
# 		model.update(Y)
# 		ov_cnt += (pred == Y)*1

# cnt = 0

# for i in tqdm(range(len(test_rows))):
# 	X = test_rows[i][1:]
# 	Y = test_rows[i][0]
# 	pred = model.predict(X)
# 	model.update(Y)
# 	ov_cnt += (pred == Y)*1
# 	cnt += (pred == Y)*1

# result = round(100 * cnt / float(len(test_rows)), 2)
# ov_result = round(100 * ov_cnt / float(num_exp*len(train_rows) + len(test_rows)), 2)
# print ('Agnostic:', result, ov_result, gamma)
