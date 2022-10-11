import sys
sys.path.insert(0, '/Users/vkraman/Research/agnostic_boosting/AgnosticMulticlassBoosting')
import csv
import numpy as np
import copy
from numpy.random import RandomState
import matplotlib  
matplotlib.use('TkAgg')   
# from skmultiflow.trees import HoeffdingTreeClassifier
# from skmultiflow.rules import VeryFastDecisionRulesClassifier
from river import tree
from core.attribute import *
from core.dataset import *
from graphviz import Source
import pandas as pd


class OnlineDT:
    '''
    Main class for Online Multiclass AdaBoost algorithm using VFDT.

    Notation conversion table: 

    v = expert_weights
    alpha =  wl_weights
    sVec = expert_votes
    yHat_t = expert_preds
    C = cost_mat

    '''

    def __init__(self, num_classes, loss='logistic', gamma=0.1, nom_att = None):
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


    def construct_distribution(self, in_dict):
        if len(in_dict) == 0:
            return np.ones(self.num_classes) * 1.0/float(self.num_classes)
        dist = np.zeros(self.num_classes)
        for key, value in in_dict.items():
            dist[key] = value
        return dist

    def predict(self, X, verbose=False):
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

        prob = self.weaklearner.predict_proba_one(self.X_dict)
        out = self.construct_distribution(prob)
        # print("dist", prob, out)
        # tmp = [x for x in range(self.num_classes) 
        #                 if out[x] == max(out)]
        label = np.argmax(out)
        # assert label == self.weaklearner.predict_one(self.X_dict)
        self.Yhat_index = label
        # self.Yhat = self.find_Y(self.Yhat_index)
        self.Yhat = self.Yhat_index
        return self.Yhat

    def index_to_vector(self, index):
        tmp = np.zeros(self.num_classes)
        tmp[int(index)] = 1
        return tmp


    def update(self, Y, X=None, verbose=False):
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
        self.Y_index = int(Y)
        self.weaklearner.learn_one(self.X_dict, self.Y_index)
        

    def gen_weaklearners(self, max_depth= None, leaf_pred= "mc", split_criterion= "gini", nom_att = None):
        self.weaklearner = tree.HoeffdingTreeClassifier(max_depth= max_depth, leaf_prediction= leaf_pred, split_criterion= split_criterion, nominal_attributes= nom_att)
    
    def draw(self):
        print(self.weaklearner.summary)
        print(self.weaklearner.draw())
        df = self.weaklearner.to_dataframe()
        print(df)

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


# split_criterion='gini',
# ...             split_confidence=1e-5,
# ...             grace_period=2000