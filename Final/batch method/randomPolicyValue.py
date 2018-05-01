import sys
import numpy as np
import cPickle as pickle

statenum = 20

gamma = 0.9
rewardvec = np.zeros(statenum)
rewardvec[0] = 1
rewardvec[-1] = 1

value_vec = np.zeros(statenum)
Transition = np.zeros((statenum,statenum))

for idx in range(statenum):
    if idx==0:
        Transition[0,0] = 0.5
        Transition[0,1] = 0.5
    elif idx==statenum-1:
        Transition[idx,idx] = 0.5
        Transition[idx,idx-1] = 0.5
    else:
        Transition[idx,idx-1] = 0.5
        Transition[idx,idx+1] = 0.5

for idx in range(1000):
    value_vec = rewardvec + Transition.dot(value_vec)*gamma
    print value_vec

pickle.dump(value_vec,open('random_value.pkl','wb'))
