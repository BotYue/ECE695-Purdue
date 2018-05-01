import numpy as np
import random
import cPickle as pickle


class ChainWalkEnv(object):

    def __init__(self, statenum, successRate):
        random.seed()
        self.statenum = statenum
        self.succRate = successRate
        self.currentState = random.randint(0,statenum-1)
        self.history = []
        self.actionMap = {'left':0,'right':1}

    def Reset(self):
        self.currentState = random.randint(0,self.statenum-1)
        self.history = []

    def EstimateState(self):
        return self.currentState

    def StateMinus(self):
        if self.currentState>0: self.currentState-=1

    def StatePlus(self):
        if self.currentState<self.statenum-1: self.currentState+=1

    def ActionCorrupt(self,actidx):
        if(random.random()<(1-self.succRate)): actidx = int(not actidx)
        return actidx

    def TakeAction(self, action):
        prevState = self.currentState
        R = 0
        actidx = self.actionMap[action]
        actidx = self.ActionCorrupt(actidx)

        if actidx==0:
            self.StateMinus()
        elif actidx==1:
            self.StatePlus()

        R = self.InstantReward()

        self.history.append((prevState+1,action,R,self.currentState+1))
        return R

    def InstantReward(self):
        R = 0
        if self.currentState in [0,self.statenum-1]: R = 1
        return R

    def printHistory(self):
        print self.history
        


def RBF(x,m,v):
    return np.exp(-np.square((x-m)/v))

if __name__=='__main__':
    cwe = ChainWalkEnv(20,0.9)
    random.seed()
    history = []
    for idx in range(100):
        cwe.Reset()
        for jdx in range(10):
            #print cwe.EstimateState()
            action = 'left'
            if random.random()>=0.5: action= 'right'
            R = cwe.TakeAction(action)

        #cwe.printHistory()
        history.append(cwe.history)
        #print len(history)
    #print RBF(1,0,1)
    pickle.dump(history,open('history.pkl','wb'))
