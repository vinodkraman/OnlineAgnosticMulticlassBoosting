import sys
sys.path.insert(0, '/Users/vkraman/Research/agnostic_boosting/AgnosticMulticlassBoosting')
import csv
import numpy as np
import copy
from algorithms.onlineGradientDescent import OnlineGradientDescent, projection_simplex_sort
from core.attribute import *
from core.dataset import *
from scipy.special import softmax
import time
from numpy.random import RandomState
import matplotlib  
matplotlib.use('TkAgg')   
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

    def __init__(self, num_classes, loss='logistic', gamma=0.1, nom_att= None):
        '''
        The kwarg loss can take values of 'logistic', 'zero_one', or 'exp'. 
        'zero_one' option corresponds to OnlineMBBM. 

        The value gamma becomes meaningful only when the loss is 'zero_one'. 
        '''
        # Initializing computational elements of the algorithm
        self.num_wls = None
        self.num_data = 0
        self.dataset = None
        self.class_index = None
        self.num_errors = 0
        self.loss = loss
        self.gamma = gamma
        self.M = 100
        self.start = True
        self.nom_att = nom_att


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

        self.num_classes = num_classes
        self.oco = OnlineGradientDescent(self.num_classes)


        #OCO
        # self.oco = OnlineGradientDescent(self.num_classes)

    ########################################################################

    # Helper functions

    def expit_diff(self, x, y):
        '''Calculates the logistic (expit) difference between two numbers
        Args:
            x (float): positive value
            y (float): negative value
        Returns:
            value (float): the expit difference
        '''
        value = 1/(1 + np.exp(x - y))
        return value

    def exp_diff(self, x, y):
        '''Calculates the exponential of difference between two numbers
        Args:
            x (float): positive value
            y (float): negative value
        Returns:
            value (float): the exponential difference
        '''
        value = np.exp(y - x)
        return value


    def make_cov_instance(self, X):
        '''Turns a list of covariates into an Instance set to self.datset 
        with None in the location of the class of interest. This is required to 
        pass to a HoeffdingTree so it can make predictions.

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset

        Returns:
            pred_instance (Instance): An Instance with the covariates X and 
                      None in the correct locations

        '''
        inst_values = list(copy.deepcopy(X))
        inst_values.insert(self.class_index, None)

        indices = list(range(len(inst_values)))
        del indices[self.class_index]
        for i in indices:
            if self.dataset.attribute(index=i).type() == 'Nominal':
                inst_values[i] = int(self.dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
            else:
                inst_values[i] = float(inst_values[i])

        pred_instance = Instance(att_values = inst_values)
        pred_instance.set_dataset(self.dataset)
        return pred_instance

    def make_full_instance(self, X, Y):
        '''Makes a complete Instance set to self.dataset with 
        class of interest in correct place

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset
            Y (string): the class of interest corresponding to these covariates.
        
        Returns:
            full_instance (Instance): An Instance with the covariates X and Y 
                            in the correct locations

        '''

        inst_values = list(copy.deepcopy(X))
        inst_values.insert(self.class_index, Y)
        for i in range(len(inst_values)):
            if self.dataset.attribute(index=i).type() == 'Nominal':
                inst_values[i] = int(self.dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
            else:
                inst_values[i] = float(inst_values[i])

        
        full_instance = Instance(att_values=inst_values)
        full_instance.set_dataset(self.dataset)
        return full_instance

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
    def construct_distribution(self, in_dict):
        if len(in_dict) == 0:
            return np.ones(self.num_classes) * 1.0/float(self.num_classes)
        dist = np.zeros(self.num_classes)
        for key, value in in_dict.items():
            dist[key] = value
        return dist

    def predict(self, X, random= True, verbose=False):
        # t = 1000 * time.time() # current time in milliseconds
        # np.random.seed(int(t) % 2**32)
        '''Runs the entire prediction procedure, updating internal tracking 
        of wl_preds and Yhat, and returns the randomly chosen Yhat

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset
            verbose (bool): If true, the function prints logs. 

        Returns:
            Yhat (string): The final class prediction
        '''

        self.X = np.array(X)
        self.X_dict = {i: self.X[i] for i in range(len(self.X))}

        # Initialize values

        cum_votes = np.zeros(self.num_classes)
        wl_preds = []
        
        for i in range(self.num_wls):
            # Get our new weak learner prediction and our new expert prediction
            prob = self.weaklearners[i].predict_proba_one(self.X_dict)
            out = self.construct_distribution(prob)
           
            out = np.array(out) #distribution
            # print(out)
            # tmp = [x for x in range(self.num_classes) 
            #             if out[x] == max(out)]
            # label = np.random.choice(tmp)
            label = np.argmax(out)
           
            label_vector = self.index_to_vector(label) #one hot encoding
            # print(label_vector)
            cum_votes += label_vector
            # wl_preds.append(out) #change this to out if worse
            wl_preds.append(label_vector) #change this to out if worse


        label_dist = cum_votes/(self.num_wls * self.gamma)
        y_hat_index = np.random.choice(self.num_classes, p = projection_simplex_sort(label_dist))

        self.wl_preds = wl_preds
        self.cum_votes = cum_votes
        self.Yhat_index = y_hat_index
        # self.Yhat = self.find_Y(self.Yhat_index)
        self.Yhat = self.Yhat_index
        return self.Yhat


    def index_to_vector(self, index):
        tmp = np.zeros(self.num_classes)
        tmp[int(index)] = 1
        return tmp


    def update(self, Y, X=None, verbose=False):
        # t = 1000 * time.time() # current time in milliseconds
        # np.random.seed(int(t) % 2**32)
        '''Runs the entire updating procedure, updating interal 
        tracking of wl_weights and expert_weights
        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset. If not given
                      the last X used for prediction will be used.
            Y (string): The true class
            verbose (bool): If true, the function prints logs. 
        '''

        if X is None:
            X = self.X

        self.X = X
        self.Y = Y
        # self.Y_index = int(self.find_Y_index(Y))
        self.Y_index = int(Y)

        for i in range(self.num_wls):
            p = self.oco.predict(self.Y_index)
            # perm = np.random.permutation(len(p))
            # for jndex in perm:
            #     self.weaklearners[i].learn_one(self.X_dict, int(jndex), sample_weight= float(p[jndex]+1e-10))

            new_wl_label_index = int(np.random.choice(self.num_classes, p = p))
            self.weaklearners[i].learn_one(self.X_dict, new_wl_label_index)
            self.oco.update(self.wl_preds[i],self.index_to_vector(self.Y_index),self.gamma)
        self.oco.reset()

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

        # print([att.get_type() for att in attributes])
        dataset = Dataset(attributes, class_index)
        self.num_classes = dataset.num_classes()
        self.oco = OnlineGradientDescent(self.num_classes)

        if self.loss == 'zero_one':
            self.biased_uniform = \
                        np.ones(self.num_classes)*(1-self.gamma)/self.num_classes
            self.biased_uniform[0] += self.gamma

        self.dataset = dataset

    def gen_weaklearners(self, num_wls, leaf_pred= "nba", max_depth=None):
        self.num_wls = num_wls
        self.oco.N = self.num_wls
        self.weaklearners = [tree.HoeffdingTreeClassifier(split_criterion= "gini", leaf_prediction= leaf_pred, max_depth=max_depth, nominal_attributes= self.nom_att) for i in range(num_wls)]


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


