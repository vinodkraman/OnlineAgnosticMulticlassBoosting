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

parser.add_argument("--num_wls")
parser.add_argument("--filename")
parser.add_argument("--noise")
parser.add_argument("--exp")
args = parser.parse_args()
d = vars(args)

num_wl = int(d["num_wls"])
input_filename = d["filename"]
noise_rate = float(d["noise"])
num_exp = int(d["exp"])

filename = add_noise(input_filename, noise_rate)
rs = RandomState(10)

gammas_A = [0.1, 0.3, 0.5, 0.7, 1.0]
gammas_R = [0.3, 0.1, 0.05, 0.01, 0.001]
wl_seeds = np.random.randint(1, 900, num_exp)
# wl_seeds = rs.randint(1, 900, num_exp)
print(wl_seeds)
shuffle_seeds = np.random.randint(1, 900, num_exp)
# shuffle_seeds = rs.randint(1, 900, num_exp)
shuffle_seeds = [4, 722, 342, 274, 636]
print(shuffle_seeds)

print(filename)
class_index = 0
training_ratio = 0.8

N = utils.get_num_instances(filename)
train_N = int(N*training_ratio)
rows = utils.get_rows(filename)

#Baseline
# avg_result = 0.0
# avg_ov_result = 0.0
wl_baseline = []
for index in tqdm(range(num_exp)):

    shuff_rows = utils.shuffle(rows, seed=shuffle_seeds[index])

    train_rows = shuff_rows[:train_N]
    test_rows = shuff_rows[train_N:]

    result, ov_result = run_dt(filename, class_index, N, wl_seeds[index], train_rows, 
    test_rows)
    wl_baseline.append(result)

    # avg_result += result / num_exp
    # avg_ov_result += ov_result / num_exp
# print 'OnlineDT:', avg_result, avg_ov_result
print('OnlineDT', wl_baseline)

#Print OG and OR
best_avg_R = []
best_avg_A = []
# for gamma_index in tqdm(range(len(gammas_A))):
    # avg_result_A, avg_ov_result_A = 0.0, 0.0
    # avg_result_R, avg_ov_result_R = 0.0, 0.0
for index in tqdm(range(num_exp)):
    wl_perf_A = []
    wl_perf_R = []
    shuff_rows = utils.shuffle(rows, seed=shuffle_seeds[index])
    train_rows = shuff_rows[:train_N]
    test_rows = shuff_rows[train_N:]
    #try every gamma for each index
    for gamma_index in range(len(gammas_A)):

        result_A, ov_result_A = run_agnostic(gammas_A[gamma_index], 
        filename, class_index, N, num_wl, wl_seeds[index], 
        train_rows, test_rows)
        wl_perf_A.append((result_A, ov_result_A, gammas_A[gamma_index]))

        # result_R, ov_result_R = run_realizable(gammas_R[gamma_index], 
        # filename, class_index, N, num_wl, wl_seeds[index], 
        # train_rows, test_rows)
        # wl_perf_R.append((result_R, ov_result_R, gammas_R[gamma_index]))

    #take the gamma with best performance
    print("Agnostic", index, wl_perf_A)
    best_avg_A.append(max(wl_perf_A))
    # print("Realizable", index, wl_perf_R)
    # best_avg_R.append(max(wl_perf_R))

print("Agnostic", best_avg_A)
# print("Realizable", best_avg_R)
