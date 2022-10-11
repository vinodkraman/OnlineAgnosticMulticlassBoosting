from algorithms.onlineAgnostic import AgnosticBoost
import utils
import numpy as np
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

def run_boosting(model, train_rows, test_rows):
    ov_cnt = 0
    for i in range(len(train_rows)):
        X = train_rows[i][1:]
        Y = train_rows[i][0]
        pred = model.predict(X)
        model.update(Y, X)
        ov_cnt += (pred == Y)*1

    cnt = 0

    for i in range(len(test_rows)):
        X = test_rows[i][1:]
        Y = test_rows[i][0]
        pred = model.predict(X)
        model.update(Y, X)
        ov_cnt += (pred == Y)*1
        cnt += (pred == Y)*1

    result = round(100 * cnt / float(len(test_rows)), 2)
    ov_result = round(100 * ov_cnt / float(len(train_rows) + len(test_rows)), 2)
    # print 'Agnostic:', result, ov_result, gamma
    return result, ov_result