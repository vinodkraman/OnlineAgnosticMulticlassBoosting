import os
import csv
import numpy as np
import copy
import random
import time
from numpy.random import RandomState
from collections import Counter
import pandas as pd


def get_rows(filepath):
    ''' Read the file and returns list of lists
    Args:
        filepath (string): File path
    Returns:
        (list): List of row lists
    '''
    rows = []
    with open(filepath, 'rt') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rows.append(row)
    return rows


def add_noise(filename, rate):
    rs_yes = RandomState(0)
    rs_label = RandomState(1)
    filename_csv = filename + ".csv"

    df = pd.read_csv(filename_csv, sep = ',', header=None)
    classes = df.iloc[:, 0].unique()
    indices = np.array(range(len(df)))
    for index in indices:  
        yes = rs_yes.binomial(1, rate, 1)
        if yes[0]:
            org_class = df.iloc[index, 0]
            all_other_class = classes[classes != org_class]
            new_class = rs_label.choice(all_other_class)
            df.iloc[index, 0] = new_class
    
    output_filename_csv = filename + "_noisy.csv"
    # df.to_csv(output_filename_csv, index=False, header=False)
    return output_filename_csv, df



