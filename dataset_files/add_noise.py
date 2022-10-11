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

# t = 1000 * time.time() # current time in milliseconds
# np.random.seed(int(t) % 2**32)

def add_noise(filename, rate):
    # rs = RandomState(20)
    filename_csv = filename + ".csv"

    df = pd.read_csv(filename_csv, sep = ',', header=None)
    classes = df.iloc[:, 0].unique()
    indices = np.array(range(len(df)))
    indices = np.random.choice(indices, int(rate*(len(df))), replace= False)
    print(len(indices))
    print(len(set(indices)))
    for index in indices:
        # t = 1000 * time.time() # current time in milliseconds
        # np.random.seed(int(t) % 2**32)      
        org_class = df.iloc[index, 0]
        all_other_class = classes[classes != org_class]
        new_class = np.random.choice(all_other_class)
        # new_class = np.random.choice(classes)
        # print(class_index)
        # rows[index][0] = classes[class_index]
        # print(new_class_index)
        df.iloc[index, 0] = new_class
    
    output_filename_csv = filename + "_noisy.csv"
    df.to_csv(output_filename_csv, index=False, header=False)
    return output_filename_csv, df

    

# def add_noise(filename, rate):
#     rs = RandomState(0)
#     filename_csv = filename + ".csv"
#     # filename = "balance-scale.csv"
#     rows = get_rows(filename_csv)
#     rows = np.array(rows)
#     print(rows.shape)
#     print("before", Counter(rows[:,0]))
#     classes = np.unique(rows[:,0])
#     indices = np.array(range(len(rows)-1))
#     indices = rs.choice(indices, int(rate*(len(rows)-1)), replace= False)
#     # indices = np.random.choice(indices, int(rate*(len(rows)-1)), replace= False)
#     # indices = np.random.randint(0, len(rows)-1, int(rate*(len(rows)-1)))
#     print(len(indices))
#     print(len(set(indices)))
#     # print(indices)
#     for index in indices:
#         # t = 1000 * time.time() # current time in milliseconds
#         # np.random.seed(int(t) % 2**32)      
#         org_class = rows[index][0]
#         all_other_class = classes[classes != org_class]
#         new_class = rs.choice(all_other_class)
#         # new_class = np.random.choice(classes)
#         # print(class_index)
#         # rows[index][0] = classes[class_index]
#         # print(new_class_index)
#         rows[index][0] = new_class

#     print("after", Counter(rows[:,0]))
#     output_filename_csv = filename + "_noisy.csv"
#     with open(output_filename_csv,"w+") as my_csv:
#         csvWriter = csv.writer(my_csv,delimiter=',')
#         csvWriter.writerows(rows)

#     return output_filename_csv



# add_noise('data/cars_correct_v2', 0.3)
# df = pd.read_csv('data/cars_correct_v2.csv', sep = ',')
# print(df.head())
# classes = df["class"].unique()
# print(classes)
# print(len(df))



