import sys
from tokenize import String
# sys.path.insert(0, '/Users/vkraman/Research/agnostic_boosting/AgnosticMulticlassBoosting')
sys.path.insert(0, '/home/vkraman/AgnosticMulticlassBoosting')
from algorithms.onlineAdaptive_v3 import AdaBoostOLM
from algorithms.oneVSall import oneVSall, oneVSallBoost
from algorithms.onlineAgnostic import AgnosticBoost
from algorithms.OCORealizableBoost import OCORealizableBoost
from algorithms.onlineDT import OnlineDT
from dataset_files.add_noise import add_noise
from run_files.run_boosting import run_boosting
from numpy.random import RandomState
import utils
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from timeit import default_timer as timer
parser = argparse.ArgumentParser()

######################################
parser.add_argument("--num_wls")
parser.add_argument("--filename")
parser.add_argument("--noise")
parser.add_argument("--exp")
parser.add_argument("--leaf")
parser.add_argument("--dep")
parser.add_argument("--nom_att")
parser.add_argument("--model")
args = parser.parse_args()
d = vars(args)

num_wl = int(d["num_wls"])
input_filename = d["filename"]
noise_rate = float(d["noise"])
num_exp = int(d["exp"])
max_depth = int(d["dep"])
leaf_pred = d["leaf"]
nom_att = d["nom_att"].split(',')
which_model = d["model"]

if nom_att[0] == "":
    nom_att = None
else:
    nom_att = [int(nom_att[i]) for i in range(len(nom_att))]

filename, dataset = add_noise(input_filename, noise_rate)
print(filename, len(dataset))
class_index = 0
training_ratio = 0.8

# preprocess dataset
# dataset = pd.read_csv(filename, header=None)  
unique_vals = dataset.iloc[:, 0].unique()
dataset.iloc[:, 0].replace(to_replace=unique_vals,
        value= list(range(len(unique_vals))),
        inplace=True)


dataset = dataset.to_numpy() 
N = len(dataset)
train_N = int(N*training_ratio)
print(nom_att)

num_classes = len(np.unique(dataset[:, 0]))
print("classes", np.unique(dataset[:, 0]))
print("num classes", num_classes)

gammas_A = [0.1, 0.3, 0.5, 0.7, 1.0]
gammas_R = [0.3, 0.1, 0.05, 0.01, 0.001]
print("which model", which_model)

#Print OG and OR
best_avg_R = []
best_avg_A = []
best_avg_ROCO = []

#shuffle dataset
dataset = np.array(utils.shuffle(dataset, seed=120))

for index in tqdm(range(num_exp)):
    wl_perf_A = []
    wl_perf_R = []
    wl_perf_ROCO = []
    
    train_rows = dataset[:train_N]
    test_rows = dataset[train_N:]

    #try every gamma for each index
    for gamma_index in range(len(gammas_A)):

        if which_model == "all" or which_model == "OR":
            start = timer()
            model = OCORealizableBoost(num_classes, loss='zero_one', gamma=gammas_A[gamma_index], nom_att=nom_att)
            model.M = 100
            model.gen_weaklearners(num_wl, leaf_pred= leaf_pred, max_depth = max_depth) 
            result_ROCO, ov_result_ROCO = run_boosting(model, train_rows, test_rows)
            wl_perf_ROCO.append((result_ROCO, ov_result_ROCO, gammas_A[gamma_index]))
            end = timer()
            print("OCOR complete", end - start)

        if which_model == "all" or which_model == "A":
            start = timer()
            model = AgnosticBoost(num_classes, loss='zero_one', gamma=gammas_A[gamma_index], nom_att=nom_att)
            model.gen_weaklearners(num_wl, leaf_pred= leaf_pred, max_depth = max_depth) 
            result_A, ov_result_A = run_boosting(model, train_rows, test_rows)
            wl_perf_A.append((result_A, ov_result_A, gammas_A[gamma_index]))
            end = timer()
            print("Agnostic complete", end- start)

        if which_model == "all" or which_model == "R":
            start = timer()
            model = AdaBoostOLM(num_classes, loss='zero_one', gamma=gammas_R[gamma_index], nom_att=nom_att)
            model.M = 100
            model.gen_weaklearners(num_wl, leaf_pred= leaf_pred, max_depth = max_depth) 
            result_R, ov_result_R = run_boosting(model, train_rows, test_rows)
            wl_perf_R.append((result_R, ov_result_R, gammas_R[gamma_index]))
            end = timer()
            print("Realizable complete", end - start)

    if which_model == "all" or which_model == "A":
        #take the gamma with best performance
        print("Agnostic", index, wl_perf_A)
        best_avg_A.append(max(wl_perf_A))
    if which_model == "all" or which_model == "R":
        print("Realizable", index, wl_perf_R)
        best_avg_R.append(max(wl_perf_R))
    if which_model == "all" or which_model == "OR":
        print("OCO Realizable", index, wl_perf_ROCO)
        best_avg_ROCO.append(max(wl_perf_ROCO))

print("Final")
if which_model == "all" or which_model == "A":
    print("Agnostic", best_avg_A)
if which_model == "all" or which_model == "R":
    print("Realizable", best_avg_R)
if which_model == "all" or which_model == "OR":
    print("OCO Realizable", best_avg_ROCO)
