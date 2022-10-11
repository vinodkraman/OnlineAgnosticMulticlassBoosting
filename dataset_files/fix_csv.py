import os
import csv
import numpy as np
import copy
from hoeffdingtree import * 
import pandas as pd


def get_rows(filepath):
    ''' Read the file and returns list of lists
    Args:
        filepath (string): File path
    Returns:
        (list): List of row lists
    '''
    rows = []
    with open(filepath, 'rb') as f:
        r = csv.reader(f)
        r.next()
        for row in r:
            rows.append(row)
    return rows

df = pd.read_csv('data/abalone.csv', sep = ',')
test = pd.get_dummies(df.yolo, prefix='yolo')
print(test)
df['yolo_f'] = test["yolo_F"]
df['yolo_m'] = test["yolo_M"]
df['yolo_i'] = test["yolo_I"]
df.pop("yolo")
first_column = df.pop("class")
df.insert(0, 'class', first_column)
# mice.fillna(0, inplace=True)
print(df)
df.to_csv('data/abalone_OHE.csv', index=False)


# rows = get_rows("data/yeast_correct.csv")
# rows = np.array(rows)
# # print(rows.shape)
# # # print(rows)
# # print(np.unique(rows[:, -1]))
# # print(len(np.unique(rows[:, -1])))
# unique_vals = []
# for col in range(len(rows[0])):
#     print(np.unique(rows[:, col]))
#     unique_vals.append(np.unique(rows[:, col]))
# print(unique_vals)

# for row in range(len(rows)):
#     for col in range(len(rows[row])-1):
#         rows[row][col] = list(unique_vals[col]).index(rows[row][col])

# print(unique_vals)
# print(rows)

# rows[:, [0, -1]] = rows[:, [-1, 0]]
# # new_rows = []
# # for i in range(len(rows)):
# #     label = rows[i][-1]
# #     test = rows[i,-2]
# #     test.
# #     rows[i][0] = label
# #     print(rows[i])
# #     new_rows.append(rows[i][:len(rows[i])-2])

# with open("nursery_correct.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(rows)
# # rows = get_rows('abalone.csv')
# # # print(rows)
# # thing_to_num = {"M":1, "F":2, "I":3}

# # for i in range(len(rows)):
# #     rows[i][0] = thing_to_num[rows[i][0]]
# #     rows[i][-1] = chr(ord("A") + int(rows[i][-1]))
# #     rows[i][-1], rows[i][0] = rows[i][0], rows[i][-1] 

# # with open("new_file.csv","w+") as my_csv:
# #     csvWriter = csv.writer(my_csv,delimiter=',')
# #     csvWriter.writerows(rows)

# # def write_rows(filepath):
# #     with open(filepath) as infile, open('holy.csv', 'w') as outfile:
# #         for line in infile:
# #             outfile.write(" ".join(line.split()).replace(' ', ','))
# #             outfile.write(",") # trailing comma shouldn't matter
# #             outfile.write("\n")

# # write_rows("cars_correct.csv")
# # rows = get_rows("holy.csv")
# # new_rows = []
# # for i in range(len(rows)):
# #     label = rows[i][-2]
# #     rows[i][0] = label
# #     print(rows[i])
# #     new_rows.append(rows[i][:len(rows[i])-2])
# # with open("cars_correct.csv","w+") as my_csv:
# #     csvWriter = csv.writer(my_csv,delimiter=',')
# #     csvWriter.writerows(new_rows)