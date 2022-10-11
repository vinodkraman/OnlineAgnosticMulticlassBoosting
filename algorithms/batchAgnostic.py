import csv
import numpy as np
import copy
from hoeffdingtree import *
from onlineGradientDescent import OnlineGradientDescent
from scipy.special import softmax
import time
from numpy.random import RandomState
import matplotlib  
from sklearn.tree import DecisionTreeClassifier
matplotlib.use('TkAgg')   
# from skmultiflow.trees import HoeffdingTreeClassifier
# from skmultiflow.rules import VeryFastDecisionRulesClassifier
from river import tree




class AgnosticBoost:
    '''
    Main class for Online Multiclass AdaBoost algorithm using VFDT.

    Notation conversion table: 

    v = expert_weights
    alpha =  wl_weights
    sVec = expert_votes
    yHat_t = expert_preds
    C = cost_mat

    '''

    def __init__(self, m, loss='logistic', gamma=0.1):
        '''
        The kwarg loss can take values of 'logistic', 'zero_one', or 'exp'. 
        'zero_one' option corresponds to OnlineMBBM. 

        The value gamma becomes meaningful only when the loss is 'zero_one'. 
        '''
        # Initializing computational elements of the algorithm
        self.num_wls = None
        self.num_classes = None
        self.num_data = 0
        self.dataset = None
        self.class_index = None
        self.num_errors = 0
        self.loss = loss
        self.gamma = gamma
        self.M = 100
        self.start = True
        self.m = m


        self.wl_edges = None
        self.weaklearners = None
        self.wl_preds = None
        self.cum_votes = None

        # Initializing data states
        self.X = None
        self.Yhat_index = None
        self.Y_index = None
        self.Yhat = None
        self.Y = None
        self.pred_conf = None
        self.P = {}
        self.hypotheses = []


        #OCO
        # self.oco = OnlineGradientDescent(self.num_classes)

    ########################################################################

    # Helper functions

    def find_Y(self, Y_index):
        '''Get class string from its index
        Args:
            Y_index (int): The index of Y
        Returns:
            Y (string): The class of Y
        '''

        Y = self.dataset.attribute(index=self.class_index).value(Y_index)
        return Y

    def find_Y_index(self, Y):
        '''Get class index from its string
        Args:
            Y (string): The class of Y
        Returns:
            Y_index (int): The index of Y
        '''

        Y_index = int(self.dataset.attribute(index=self.class_index)
                    .index_of_value(Y))
        return Y_index

    ########################################################################
    def sample_relabel_random(self, X, y, P, m0):
        sample_indices = np.random.choice(range(self.m), size= m0)
        sample_X = []
        sample_y = []
        sample_weights = np.ones(m0)
        for sample_index in sample_indices:
            sample_X.append(X[sample_index])
            sample_y.append(int(np.random.choice(self.num_classes, p = P[sample_index])))
        
        return np.array(sample_X), np.array(sample_y), sample_weights

    def sample_relabel_det_v2(self, X, y, P, m0):
        sample_X = []
        sample_y = []
        sample_weight = []
        max_weight = -1

        for sample_index in range(len(X)):
            for label in range(self.num_classes):
                sample_X.append(X[sample_index])
                sample_y.append(label)
                weight =  P[sample_index][label]
                max_weight = max(max_weight, weight)
                sample_weight.append(weight)

        sample_X = np.array(sample_X)
        sample_y = np.array(sample_y)
        sample_weight = np.array(sample_weight)

        # permutation = np.random.permutation(len(sample_weight))
        # sample_X = [sample_X[i] for i in permutation]
        # sample_y = [sample_y[i] for i in permutation]
        # sample_weight = [sample_weight[i] for i in permutation]

        # if max_weight > 0:
        #     sample_weight /= max_weight

        return np.array(sample_X), np.array(sample_y), np.array(sample_weight)

    def sample_relabel_det(self, X, y, P, m0):
        sample_indices = np.random.choice(range(self.m), size= m0)
        sample_X = []
        sample_y = []
        sample_weight = []
        max_weight = -1

        for sample_index in sample_indices:
            for label in range(self.num_classes):
                sample_X.append(X[sample_index])
                sample_y.append(label)
                max_weight = max(max_weight, P[sample_index][label])
                sample_weight.append(P[sample_index][label])

        sample_weight = np.array(sample_weight)
        
        # permutation = np.random.permutation(len(sample_weight))
        # sample_X = [sample_X[i] for i in permutation]
        # sample_y = [sample_y[i] for i in permutation]
        # sample_weight = [sample_weight[i] for i in permutation]
        # if max_weight > 0:
        #     sample_weight /= max_weight
        return np.array(sample_X), np.array(sample_y), np.array(sample_weight)

    def grab_y_indices(self, y):
        y_indices = []
        for label in y:
            y_indices.append(self.find_Y_index(label))
        return y_indices

    def fit(self, X, y, T, m0):
        self.gen_weaklearners(T)
        y_indices = self.grab_y_indices(y)
        for t in range(T):
            self.P = self.get_OCO_predictions(y_indices)
            sample_X, sample_y, sample_weights = self.sample_relabel_random(X, y_indices, self.P, m0)
            self.weaklearners[t].fit(sample_X, sample_y, sample_weights)
            self.update_OCO_oracles(X, y_indices, t)


    def get_OCO_predictions(self, y_indices):
        for i in range(self.m):
            self.P[i] = self.oco[i].predict(y_index=y_indices[i])
        return self.P

    def construct_distribution(self, t, prob):
        if len(prob) == self.num_classes:
            return prob
        else:
            dist = np.zeros(self.num_classes)
            wl_classes = self.weaklearners[t].classes_
            for j in range(len(prob)):
                dist[wl_classes[j]] = prob[j]
            return np.array(dist)


    def update_OCO_oracles(self, X, y, t):
        pred = self.weaklearners[t].predict(X)
        pred_prob = self.weaklearners[t].predict_proba(X)
        # print(len(pred_prob))
        for i in range(self.m):
            # dist = self.construct_distribution(t, pred_prob[i])
            # self.oco[i].update(dist, self.index_to_vector(y[i]), self.gamma)
            self.oco[i].update(self.index_to_vector(pred[i]), self.index_to_vector(y[i]), self.gamma)


    def predict(self, X, random= True):
        cum_votes = np.zeros((len(X), self.num_classes))

        for t in range(len(self.weaklearners)):
            # prob = self.weaklearners[t].predict_proba(X)
            # cum_votes += prob
            predictions = self.weaklearners[t].predict(X)
            for j in range(len(predictions)):
                cum_votes[j][predictions[j]] += 1

        cum_votes /= len(self.weaklearners)
        predictions = []
        # print(cum_votes)

        for i in range(len(X)):
            max_index = np.argmax(cum_votes[i])
            # if not random and cum_votes[i][max_index] > 0.5:
            if not random and cum_votes[i][max_index]/self.gamma >= 1:
                label_index = max_index
            else:
                label_index = np.random.choice(self.num_classes, p= cum_votes[i])
            label = self.find_Y(label_index)
            predictions.append(label)

        return np.array(predictions)

    def index_to_vector(self, index):
        tmp = np.zeros(self.num_classes)
        tmp[int(index)] = 1
        return tmp


    def initialize_dataset(self, filename, class_index, probe_instances=10000):
        """ CODE HERE TAKEN FROM main.py OF HOEFFDINGTREE 
        Open and initialize a dataset in CSV format.
        The CSV file needs to have a header row, from where the attribute 
        names will be read, and a set of instances containing at least one 
        example of each value of all nominal attributes.

        Args:
            filename (str): The name of the dataset file (including filepath).
            class_index (int): The index of the attribute to be set as class.
            probe_instances (int): The number of instances to be used to 
                initialize the nominal attributes. (default 100)

        Returns:
            It does not return anything. Internal dataset will be updated. 
        """
        self.class_index = class_index
        if not filename.endswith('.csv'):
            message = 'Unable to open \'{0}\'. Only datasets in \
                CSV format are supported.'
            raise TypeError(message.format(filename))
        with open(filename) as f:
            fr = csv.reader(f)
            headers = next(fr)

            att_values = [[] for i in range(len(headers))]
            instances = []
            try:
                for i in range(probe_instances):
                    inst = next(fr)
                    instances.append(inst)
                    for j in range(len(headers)):
                        try:
                            inst[j] = float(inst[j])
                            att_values[j] = None
                        except ValueError:
                            inst[j] = str(inst[j])
                        if isinstance(inst[j], str):
                            if att_values[j] is not None:
                                if inst[j] not in att_values[j]:
                                    att_values[j].append(inst[j])
                            else:
                                raise ValueError(
                                    'Attribute {0} has both Numeric and Nominal values.'
                                    .format(headers[j]))
            # Tried to probe more instances than there are in the dataset file
            except StopIteration:
                pass

        attributes = []
        for i in range(len(headers)):
            if att_values[i] is None:
                attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
            else:
                attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))

        dataset = Dataset(attributes, class_index)
        self.num_classes = dataset.num_classes()
        print("num_classes", self.num_classes)
        self.oco = [OnlineGradientDescent(self.num_classes) for i in range(self.m)]

        if self.loss == 'zero_one':
            self.biased_uniform = \
                        np.ones(self.num_classes)*(1-self.gamma)/self.num_classes
            self.biased_uniform[0] += self.gamma

        self.dataset = dataset

    def gen_weaklearners(self, T):
        for i in range(len(self.oco)):
            self.oco[i].N = T
        self.weaklearners = [DecisionTreeClassifier(max_depth=1) for t in range(T)]

    def get_num_errors(self):
        return self.num_errors

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_num_wls(self, n):
        self.num_wls = n

    def set_class_index(self, class_index):
        self.class_index = class_index

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_exp_step_size(self, exp_step_size):
        self.exp_step_size = exp_step_size


