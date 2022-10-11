import sys
sys.path.insert(0, '/Users/vkraman/Research/agnostic_boosting/AgnosticMulticlassBoosting')
import algorithms.onlineAdaptive_v3
from algorithms.oneVSall import oneVSall, oneVSallBoost
from algorithms.onlineAgnostic import AgnosticBoost
from algorithms.onlineDT import OnlineDT
import utils
import numpy as np
from tqdm import tqdm


def main():
    test = np.loadtxt('data/2d_synth.txt', delimiter=',')
	# Load data
    filename = 'data/2d_synth.csv'
    class_index = 0
    training_ratio = 0.8

    N = utils.get_num_instances(filename)
    train_N = int(N*training_ratio)
    rows = utils.get_rows(filename)
    rows = utils.shuffle(rows, seed=10)

    train_rows = rows[:train_N]
    test_rows = rows[train_N:]

	# agnostic boost
    gamma = 0.50
	# print 'Num weak learners:', num_weaklearners

    model = AgnosticBoost(loss='zero_one', gamma=gamma)
    model.initialize_dataset(filename, class_index, N)
    model.gen_weaklearners(100,
                        min_grace=5, max_grace=20,
                        min_tie=0.01, max_tie=0.9,
                        min_conf=0.01, max_conf=0.9,
                        min_weight=5, max_weight=200) 

    ov_cnt = 0
    num_exp = 1
    for exp in range(num_exp):
        for i in tqdm(range(len(train_rows))):
            X = train_rows[i][1:]
            Y = train_rows[i][0]
            pred = model.predict(X)
            model.update(Y, X)
            ov_cnt += (pred == Y)*1

    cnt = 0

    for i in tqdm(range(len(test_rows))):
        X = test_rows[i][1:]
        Y = test_rows[i][0]
        pred = model.predict(X)
        model.update(Y, X)
        ov_cnt += (pred == Y)*1
        cnt += (pred == Y)*1

    result = round(100 * cnt / float(len(test_rows)), 2)
    ov_result = round(100 * ov_cnt / float(num_exp*len(train_rows) + len(test_rows)), 2)
    print('Agnostic:', result, ov_result, gamma)

    #online DT
    model = OnlineDT(loss='zero_one', gamma=1)
    model.initialize_dataset(filename, class_index, N)
    model.gen_weaklearners(1,
                        min_grace=5, max_grace=20,
                        min_tie=0.01, max_tie=0.9,
                        min_conf=0.01, max_conf=0.9,
                        min_weight=5, max_weight=200) 

    ov_cnt = 0
    for exp in range(1):
        for i in tqdm(range(len(train_rows))):
            X = train_rows[i][1:]
            Y = train_rows[i][0]
            pred = model.predict(X)
            model.update(Y, X)
            ov_cnt += (pred == Y)*1

    cnt = 0

    for i in tqdm(range(len(test_rows))):
        X = test_rows[i][1:]
        Y = test_rows[i][0]
        pred = model.predict(X)
        model.update(Y, X)
        ov_cnt += (pred == Y)*1
        cnt += (pred == Y)*1

    result = round(100 * cnt / float(len(test_rows)), 2)
    ov_result = round(100 * ov_cnt / float(1*len(train_rows) + len(test_rows)), 2)
    print('OnlineDT:', result, ov_result)

if __name__ == '__main__':
	main()