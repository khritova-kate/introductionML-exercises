import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def dist(x, y):
    d = 0
    for i in range(len(x)):
        d += (x[i] - y[i])**2
    return np.sqrt(d)

def gradient_step(w1, w2, k, C, X,y):
    sum1 = 0
    sum2 = 0
    l = len(data[0])
    for i in range(l):
        yi  = y[i]
        xi1 = X[i][0]
        xi2 = X[i][1]
        
        bracket = 1 - 1./(1 + np.exp( -yi*(w1*xi1 + w2*xi2) ))
        sum1 += yi*xi1*bracket
        sum2 += yi*xi2*bracket
     
    w1 = w1 + (k/l)*sum1 - k*C*w1
    w2 = w2 + (k/l)*sum2 - k*C*w2
    
    return w1, w2

def GradualDescent_fit(w1, w2, k, C, X,y, eps, n_iter):
    it = 0
    while True:
        if it > n_iter:
            print('too many iterations')
            break
        w1_old = w1
        w2_old = w2
        w1, w2 = gradient_step(w1, w2, k, C, X,y)
        if dist( np.array((w1_old, w2_old)), np.array((w1, w2)) ) <= eps:
            break
        it += 1
        
    return w1, w2

def GradualDescent_predict(w1, w2, X,y):
    res = list()
    for i in range( len(data[0]) ):
        xi1 = X[i][0]
        xi2 = X[i][1]
        ax = 1./(1 + math.exp(-w1*xi1 - w2*xi2))
        res.append( ax )
    return res

data = pd.read_csv("data-logistic.csv", header = None)
y = np.array(data.iloc[:,0 ])
X = np.array(data.iloc[:,1:])

w1, w2 = GradualDescent_fit(0,0, 0.01, 0, X,y, 1e-5, 10000)
ras1 = roc_auc_score(data[0], GradualDescent_predict(w1,w2,X,y))

w1, w2 = GradualDescent_fit(0,0, 0.01, 10, X,y, 1e-5, 10000)
ras2 = roc_auc_score(data[0], GradualDescent_predict(w1,w2,X,y))

print(round(ras1, 3), round(ras2, 3))
