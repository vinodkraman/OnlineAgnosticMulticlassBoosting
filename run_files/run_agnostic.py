from algorithms.onlineAgnostic import AgnosticBoost
import utils
import numpy as np
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

def run_agnostic(gamma, num_classes, nom_att, N, leaf_pred, max_depth, X_train, X_test, y_train, y_test):
    model = AgnosticBoost(num_classes, loss='zero_one', gamma=gamma, nom_att=nom_att)
    model.gen_weaklearners(N, leaf_pred= leaf_pred, max_depth = max_depth) 


    ov_cnt = 0
    for i in range(len(X_train)):
        X = X_train[i][1:]
        Y = y_train[i][0]
        pred = model.predict(X)
        model.update(Y,X)
        ov_cnt += (pred == Y)*1

    cnt = 0

    for i in range(len(X_test)):
        X = X_test[i][1:]
        Y = y_test[i][0]
        pred = model.predict(X)
        model.update(Y,X)
        ov_cnt += (pred == Y)*1
        cnt += (pred == Y)*1

    result = round(100 * cnt / float(len(X_test)), 2)
    ov_result = round(100 * ov_cnt / float(len(X_train) + len(X_test)), 2)
    # print 'Agnostic:', result, ov_result, gamma
    return result, ov_result

# def main():
#     parser.add_argument("--wl_seed")
#     parser.add_argument("--shuffle_seed")
#     parser.add_argument("--gamma")
#     parser.add_argument("--num_wls")
#     parser.add_argument("--filename")
#     parser.add_argument("--noise")
#     args = parser.parse_args()
#     d = vars(args)

#     wl_seed = int(d["wl_seed"])
#     shuffle_seed = int(d["shuffle_seed"])
#     gamma = float(d["gamma"])
#     num_wl = int(d["num_wls"])
#     input_filename = d["filename"]
#     noise_rate = float(d["noise"])

#     filename = add_noise(input_filename, noise_rate)
#     # filename = 'cars_correct.csv'
#     print(filename)
#     class_index = 0
#     training_ratio = 0.8

#     N = utils.get_num_instances(filename)
#     train_N = int(N*training_ratio)
#     rows = utils.get_rows(filename)
#     rows = utils.shuffle(rows, seed=shuffle_seed)

#     train_rows = rows[:train_N]
#     test_rows = rows[train_N:]

#     model = AgnosticBoost(loss='zero_one', gamma=gamma)
#     model.initialize_dataset(filename, class_index, N)
#     model.gen_weaklearners(num_wl,
#                         min_grace=5, max_grace=20,
#                         min_tie=0.01, max_tie=0.9,
#                         min_conf=0.01, max_conf=0.9,
#                         min_weight=5, max_weight=200, 
#                         seed= wl_seed) 

#     ov_cnt = 0
#     num_exp = 1
#     for exp in range(num_exp):
#         for i in tqdm(range(len(train_rows))):
#             X = train_rows[i][1:]
#             print(X)
#             Y = train_rows[i][0]
#             pred = model.predict(X)
#             model.update(Y)
#             ov_cnt += (pred == Y)*1

#     cnt = 0

#     for i in tqdm(range(len(test_rows))):
#         X = test_rows[i][1:]
#         Y = test_rows[i][0]
#         pred = model.predict(X)
#         model.update(Y)
#         ov_cnt += (pred == Y)*1
#         cnt += (pred == Y)*1

#     result = round(100 * cnt / float(len(test_rows)), 2)
#     ov_result = round(100 * ov_cnt / float(num_exp*len(train_rows) + len(test_rows)), 2)
#     print ('Agnostic:', result, ov_result, gamma)


# if __name__ == '__main__':
# 	main()