import numpy as np
# import cvxpy as cvx
from numpy import linalg as LA
def proj(a, y):
        l = y/a
        idx = np.argsort(l)
        d = len(l)

        evalpL = lambda k: np.sum(a[idx[k:]]*(y[idx[k:]] - l[idx[k]]*a[idx[k:]]) ) -1
    
        def bisectsearch():
            idxL, idxH = 0, d-1
            L = evalpL(idxL)
            H = evalpL(idxH)
    
            if L<0:
                return idxL
    
            while (idxH-idxL)>1:
                iMid = int((idxL+idxH)/2)
                M = evalpL(iMid)
    
                if M>0:
                    idxL, L = iMid, M
                else:
                    idxH, H = iMid, M
    
            return idxH
    
        k = bisectsearch()
        lam = (np.sum(a[idx[k:]]*y[idx[k:]])-1)/np.sum(a[idx[k:]])
    
        x = np.maximum(0, y-lam*a)

        return x

dist = np.array([0, 0, 0.03, 0])
gamma = 1
dist /= gamma
print(dist)
print(proj(np.ones(len(dist)), dist))
