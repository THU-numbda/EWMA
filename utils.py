import numpy as np
from scipy.sparse.linalg import LinearOperator
from sklearn.metrics.pairwise import rbf_kernel,euclidean_distances
from sklearn.datasets import load_svmlight_file
import pandas as pd

addr = "./data/"

def load_data(dataset="satimage"):
    if dataset == "satimage":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#satimage
        #file = "./data/satimage.scale"
        # gamma = 4.0 for RBF and 0.05 for csRBF
        data_file = addr+"satimage.scale"
        X, y = load_svmlight_file(data_file)
        U = X.toarray()
        return U
    elif dataset == "abalone":
        #  https://archive.ics.uci.edu/ml/datasets/Abalone
        # gamma = 500.0 for RBF and 2.5 for csRBF
        file = addr+"abalone.data"
        data = pd.read_csv(file, sep=',', engine='c', header=None)
        data.iloc[:, 0] = data.apply(lambda x: 1.0 if x[0] == 'M' else 0.0 if x[0] == 'F' else 0.5, axis=1)
        data = data.to_numpy(dtype=np.float32)
        X = data[:,:-1]
        y = data[:,-1]
        U = X
        return U
    elif dataset == "cpusmall_scale":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale
        # gamma = 50.0 for RBF and 1.0 for csRBF
        file = addr+"cpusmall_scale"
        # feature range[-1,0]
        X, y = load_svmlight_file(file)
        U = X.toarray()
        return U
    elif dataset == "usps":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps
        # gamma = 0.05 for RBF and 0.002 for csRBF
        train_file = addr+"usps"
        test_file = addr+"usps.t"
        train_X, train_y = load_svmlight_file(train_file)
        test_X, test_y = load_svmlight_file(test_file)
        train_X = train_X.toarray()
        test_X = test_X.toarray()
        U = np.concatenate((train_X,test_X),axis=0)
        return U
    elif dataset == "mushroom":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms
        # gamma = 0.5 for RBF and 0.01 for csRBF 
        # feature range{0,1}
        file = addr+"mushrooms"
        X, y = load_svmlight_file(file)
        U = X.toarray()
        return U
    elif dataset == "letter":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#letter
        # gamma = 5.0 for RBF and 0.2 for csRBF
        file = addr+"letter.scale"
        X, y = load_svmlight_file(file)
        U = X.toarray()
        return U
    elif dataset == "letter_large":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#letter
        # gamma = 5.0 for RBF and 0.2 for csRBF
        file = addr+"letter.scale"
        X, y = load_svmlight_file(file)
        X = X.toarray()
        file = addr+"letter.scale.t"
        Y, y = load_svmlight_file(file)
        Y = Y.toarray()
        U = np.concatenate((X,Y),axis=0)
        return U
    elif dataset == "a9a":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a
        # gamma = 0.4 for RBF and 0.01 for csRBF
        train_file = addr+"a9a"
        test_file = addr+"a9a.t"
        train_X, train_y = load_svmlight_file(train_file)
        test_X, test_y = load_svmlight_file(test_file)
        train_X = train_X.toarray()
        test_X = test_X.toarray()
        U = train_X
        return U

# compute the csRBF
def construct_csRBF(X,Y,gamma,theta=3.0,p=2.0):
    K = rbf_kernel(X,Y,gamma)
    n,d = X.shape
    para = theta*0.5/gamma
    compactly = np.maximum(0,np.power((1-np.sqrt(euclidean_distances(X, Y, squared=True))/para),int((d+1)/p)+1))
    FinalK = np.multiply(K,compactly)
    return FinalK

# norm(K_appro_K,2) is too slow for large matrix, use LinearOperator to compute the spectral norm.
def func(delta_K):
    n = delta_K.shape[0]
    def matvec(x):
        return delta_K.dot(x)
    def rmatvec(x):
        return delta_K.T.dot(x)
    return LinearOperator((n,n),matvec,rmatvec)
