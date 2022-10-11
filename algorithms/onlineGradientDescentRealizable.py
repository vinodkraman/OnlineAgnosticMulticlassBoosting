import numpy as np
# import cvxpy as cvx
from numpy import linalg as LA

class OnlineGradientDescent:
    def __init__(self, k):
        '''
        The kwarg loss can take values of 'logistic', 'zero_one', or 'exp'. 
        'zero_one' option corresponds to OnlineMBBM. 

        The value gamma becomes meaningful only when the loss is 'zero_one'. 
        '''
        # Initializing computational elements of the algorithm
        self.k = k
        self.N = -1

        self.p = 0.50

        self.n = 1
        self.cum_loss = 0
        self.gradeint_sum = 0

    def compute_loss_gradient(self, WL_pred, y, gamma):
        return (2*np.dot(WL_pred, y) - 1)/gamma - 1

    def predict(self, y_index= None):
        return self.p

    def update(self, WL_pred, y, gamma):
        eta_t = 1/((1 + 1/gamma)*np.sqrt(100))
        grad_t = self.compute_loss_gradient(WL_pred, y, gamma)
        tmp = self.p - (eta_t * grad_t)
        # print(eta_t, grad_t)
        # print(eta_t*grad_t, "etAWDW")
        self.p = max(min(tmp, 1), 0) #if p is above 1, project onto 1, if p is smaller than 0, bring to 0. 
        self.n += 1

    def reset(self):
        self.p = 0.50
        self.n = 1
        self.cum_loss = 0
        self.gradeint_sum = 0
    
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w    


    


    
        

