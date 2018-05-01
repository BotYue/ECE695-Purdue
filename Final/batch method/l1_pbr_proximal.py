import sys
import numpy as np
import cPickle as pickle
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
def l1_proximal_map(X,t):
    X_map = X
    #np.copyto(X,X_map)
    dim = X_map.shape[0]
    for idx in range(dim):
        if X_map[idx]>t: X_map[idx]-=t
        elif X_map[idx]<-t: X_map[idx]+=t
        else: X_map[idx] = 0
    return X_map

def SpaceProjectionMat(Phi):
    PhiT = Phi.T
    PhiTPhi_inv = np.linalg.pinv(np.dot(PhiT,Phi))
    P = np.dot(Phi,np.dot(PhiTPhi_inv,PhiT))
    print 'heyyy'
    print P.shape
    return P

def regul_main(ld):
    iternum = 100
    [feat_mat1, rewards, feat_mat2] = pickle.load(open('features.pkl'))
    P = SpaceProjectionMat(feat_mat1)
    
    gamma = 0.9
    #ld = 1
# regulirization para, ld=10
    lr = 0.00001
    d = np.dot(P,rewards)
    C = np.dot(P,feat_mat2)*gamma-feat_mat1
    datanum,featnum = feat_mat1.shape
    #beta = np.random.normal(size=featnum)
    beta = np.zeros(featnum)
    iter_count=0
    for idx in range(iternum):
        
        beta_grad = np.dot(C.T,np.dot(C,beta)+d)
        beta_grad_update = beta - beta_grad*lr
        #beta = beta_grad_update
        beta = l1_proximal_map(beta_grad_update, ld*lr)
        obj = np.linalg.norm(np.dot(C,beta)+d) + ld*np.linalg.norm(beta,1)
        obj2 = np.linalg.norm(np.dot(C,beta)+d)
	#print np.count_nonzero(beta)
        iter_count+=1    
    print beta.shape
#shape=506
    #print beta
    #print np.count_nonzero(beta)
    #print obj 
    #print obj2
    print iter_count
    pickle.dump(beta,open('beta.pkl','wb'))
    return np.count_nonzero(beta),beta
def main():
    
    a=0
    lambda_set=[0.1,0.5,1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
    result_size=len(lambda_set)
    result=np.zeros(result_size)
    beta_total=np.zeros((result_size,506))
    for mn in lambda_set:
 
        result[a],beta_total[a,]=regul_main(mn)
        a=a+1
        print 'test', a
    print result    
    # log, ld=20,nnz=27;ld=10,nnz=151;ld=5,nnz=257;ld=2,nnz=381;ld=1,nnz=438
    
    #plot.scatter()
    plt.figure(figsize=(8,5))
    #plt.plot(lambda_set,result/506,'g*-')
    print beta_total.shape
    plt.plot(lambda_set,beta_total[:,:],linewidth=0.5)
    plt.xlabel("Lambda")
    plt.ylabel("Feature Weights")
    #plt.ylabel("Sparsity")
    plt.title("Sparsity of L1-PBR")
    plt.show()

if __name__=='__main__':
    main()
