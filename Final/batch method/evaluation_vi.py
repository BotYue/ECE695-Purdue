import sys
import numpy as np
from feature_vi import SelectedFeat
import cPickle as pickle

trueValue = pickle.load(open('random_value.pkl','r'))
estimatedValue = np.zeros_like(trueValue)

beta = pickle.load(open('beta.pkl','r'))
iternum = 2000
for state in range(20):
    value = 0.0
    for iter in range(iternum):
        #feat = featureFunc1(state)
        feat = SelectedFeat(state)
        value+=np.dot(beta,feat.T)
    value/=iternum
    estimatedValue[state] = value

mse = ((trueValue-estimatedValue)**2).mean()
print estimatedValue
print mse
