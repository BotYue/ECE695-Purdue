import sys
import numpy as np
import cPickle as pickle

def RBF(x,m,v):
    return np.exp(-np.square((x-m)/v))

def ReleFeatureFunc(state):
    rel_feat = np.zeros((1,6))
    rel_feat[0,0] = state
    rel_feat[0,1] = RBF(state, 0, 1)
    rel_feat[0,2] = RBF(state, 5, 1)
    rel_feat[0,3] = RBF(state, 10, 1)
    rel_feat[0,4] = RBF(state, 15, 1)
    rel_feat[0,5] = RBF(state, 20, 1)
    return rel_feat

def featureFunc1(state):
    noise_feat_num = 500
    irre_feat = np.random.normal(size=(1,noise_feat_num))
    rel_feat = ReleFeatureFunc(state)
    feat = np.concatenate((rel_feat,irre_feat),axis = 1)

    return feat

def SelectedFeat(state):
    #return ReleFeatureFunc(state)
    return featureFunc1(state)

def main():
    history = pickle.load(open('history.pkl'))

    feat_total = None
    feat_total2 = None
    reward_total = None

    for idx in range(len(history)):
        samp_seq = history[idx]
        seq_size = len(samp_seq)
        if seq_size==0: continue
        rewards = np.zeros(seq_size)
        actions = np.zeros(seq_size)

        for jdx in range(seq_size):
            state = samp_seq[jdx][0]
            state2 = samp_seq[jdx][3]
            rewards[jdx] = samp_seq[jdx][2]

            #feat = featureFunc1(state)
            #feat2 = featureFunc1(state2)
            feat = SelectedFeat(state)
            feat2 = SelectedFeat(state2)

            if feat_total is None:
                feat_total = feat
                feat_total2 = feat2
            else:
                feat_total = np.concatenate((feat_total,feat),axis = 0)
                feat_total2 = np.concatenate((feat_total2,feat2),axis = 0)
        if reward_total is None: reward_total = rewards
        else: reward_total = np.concatenate((reward_total,rewards),axis = 0)

    pickle.dump([feat_total,reward_total,feat_total2],open('features.pkl','wb'))

if __name__=='__main__':
    main()

