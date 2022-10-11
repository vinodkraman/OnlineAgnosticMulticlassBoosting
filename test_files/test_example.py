import sys
sys.path.insert(0, '/Users/vkraman/Research/agnostic_boosting/AgnosticMulticlassBoosting')
from algorithms.onlineAdaptive_v3 import AdaBoostOLM
from algorithms.oneVSall import oneVSall, oneVSallBoost
from algorithms.onlineAgnostic import AgnosticBoost
from algorithms.onlineDT import OnlineDT
from dataset_files.add_noise import add_noise
from run_files.run_boosting import run_boosting
import utils
import numpy as np
from tqdm import tqdm


def main():
	# Load data
	filename = 'data/abalone'
	filename, dataset = add_noise(filename, 0.0)
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

	num_classes = len(np.unique(dataset[:, 0]))
	print("classes", np.unique(dataset[:, 0]))
	print("num classes", num_classes)

	#shuffle dataset
	dataset = np.array(utils.shuffle(dataset, seed=100))

	# print 'Num weak learners:', num_weaklearners
	nom_att = [0]
	model =  AdaBoostOLM(num_classes, loss='zero_one', gamma=1, nom_att=nom_att)
	model.gen_weaklearners(100, leaf_pred = "mc", max_depth=1) 
	ov_cnt = 0

	train_rows = dataset[:train_N]
	test_rows = dataset[train_N:]

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
	print('Realizable:', result, ov_result)


if __name__ == '__main__':
	main()