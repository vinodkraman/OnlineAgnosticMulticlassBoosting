import csv
import numpy as np
import copy
from hoeffdingtree import *
from numpy.random import RandomState
from sklearn.tree import DecisionTreeClassifier

class BatchMM:
    '''
    Main class for Online Multiclass AdaBoost algorithm using VFDT.

    Notation conversion table: 

    v = expert_weights
    alpha =  wl_weights
    sVec = expert_votes
    yHat_t = expert_preds
    C = cost_mat

    '''

    def __init__(self, T, loss='logistic', gamma=0.1):
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
        self.exp_step_size = 1
        self.loss = loss
        self.gamma = gamma
        self.M = 100
        self.T = T

        self.potentials = {}

        self.wl_edges = None
        self.weaklearners = None
        self.expert_weights = None
        self.wl_weights = None
        self.wl_preds = None
        self.expert_preds = None
        self.cost_mat_diag = None

        # Initializing data states
        self.X = None
        self.Yhat_index = None
        self.Y_index = None
        self.Yhat = None
        self.Y = None
        self.pred_conf = None

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

    def mc_potential(self, t, b, s):
        '''Approximate potential via Monte Carlo simulation
        Arbs:
            t (int)     : number of weak learners until final decision
            b (list)    : baseline distribution
            s (list)    : current state
        Returns:
            potential value (float)
        '''
        k = len(b)
        r = 0
        cnt = 0
        for _ in range(self.M):
            x = np.random.multinomial(t, b)
            x = x + s
            tmp = x[r]
            x[r] = 0
            if tmp <= np.max(x):
                cnt += 1
        return float(cnt) / self.M

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

    def get_potential(self, y, n, s):
        '''Compute potential
        Args:
            r (int): True label index
            n (int): Number of weak learners until final decision
            s (list): Current state
        Returns:
            (float) potential function
        '''
        new_s = list(s)
        new_s[y] = -np.inf
        new_s.sort()
        new_s[0] = s[y]

        key = (n, tuple(new_s))
        if key not in self.potentials:
            value = self.mc_potential(n, self.biased_uniform, new_s)
            self.potentials[key] = value
        return self.potentials[key]


        # ret = np.zeros((k, k))
        #     for r in xrange(k):
        #         for l in xrange(k):
        #             e = np.zeros(k)
        #             e[l] = 1
        #             ret[r, l] = self.get_potential(r, self.num_wls-i, s+e)
        #     return ret

    def compute_cost(self, m, s, t, y):
        ''' Compute cost matrix
        Args:
            s (matrix): Current state
            t (int): iteration
        Return:
            (numpy.ndarray) Cost matrix
        '''
        k = self.num_classes
        T = len(self.weaklearners)
            
        ret = np.zeros((m, k))
        for i in range(m):
            for l in range(k):
                e = np.zeros(k)
                e[l] = 1
                tmp = self.get_potential(y[i], T-t-1, s[i]+e)
                ret[i, l] = tmp
        return ret  

    def grab_y_indices(self, y):
        y_indices = []
        for label in y:
            y_indices.append(self.find_Y_index(label))
        return y_indices

    def fit(self, X, y, T):
        self.gen_weaklearners(T)
        self.states = np.zeros((len(X), self.num_classes))  #one state for each example
        y_indices = self.grab_y_indices(y)
        m = len(X)

        for t in range(T):
            #get cost matrix for all examples
            cost_mat = self.compute_cost(m, self.states, t, y_indices)
            #compute weight based on correct labels for each example
            sample_weights = self.get_sample_weights(t, cost_mat, y_indices)
            # print(sample_weights)
            #train weak learner with sample weights
            self.weaklearners[t].fit(X, y_indices, sample_weights)
            #get wl_predictions
            wl_predictions = self.weaklearners[t].predict(X)
            #update state for next round
            self.states = self.update_state(wl_predictions, self.states)

            # const = self.weight_consts[i]
            # ret = self.cost_mat_diag[i,self.Y_index]/(self.num_classes-1)
            # ret = 0.1 * const * ret
            # return max(1e-10, ret)
    def get_sample_weights(self, t, cost_mat, y):
        sample_weights = []
        max_weight = -1
        for i in range(len(cost_mat)):
            const = self.weight_consts[t]
            cost_row = np.array(cost_mat[i, :])
            # ret = cost_row[y[i]]/(self.num_classes - 1)
            # ret = 0.1 * const * ret
            # sample_weights.append(max(1e-10, ret))

            cost_row -= cost_row[y[i]]
            cost_row = np.max(cost_row, 0)
            weight = np.sum(cost_row)
            max_weight = max(max_weight, weight)
            sample_weights.append(weight)

        sample_weights = np.array(sample_weights)

        if max_weight > 0:
            sample_weights /= max_weight
        return sample_weights


    def update_state(self, predictions, states):
        for i in range(len(predictions)):
            states[i][predictions[i]] += 1
        return states


    def predict(self, X):
        cum_votes = np.zeros((len(X), self.num_classes))
        for i in range(len(self.weaklearners)):
            predictions = self.weaklearners[i].predict(X)
            for j in range(len(predictions)):
                cum_votes[j][predictions[j]] += 1

        prediction_index = np.argmax(cum_votes, axis= 1)
        prediction_labels = []
        for prediction in prediction_index:
            label = self.find_Y(prediction)
            prediction_labels.append(label)

        return prediction_labels

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

        self.biased_uniform = \
                    np.ones(self.num_classes)*(1-self.gamma)/self.num_classes
        self.biased_uniform[0] += self.gamma

        self.dataset = dataset

    def gen_weaklearners(self, T, min_weight = 10, max_weight = 200):
        ''' Generate weak learners.
        Args:
            num_wls (int): Number of weak learners
            Other args (float): Range to randomly generate parameters
            seed (int): Random seed
        Returns:
            It does not return anything. Generated weak learners are stored in 
            internal variables. 
        '''
        self.weaklearners = [DecisionTreeClassifier(max_depth=1) for i in range(T)]
        self.weight_consts = [np.random.uniform(low=min_weight, high=max_weight)
                                for _ in range(T)]

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


