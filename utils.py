import numpy as np
from scipy.sparse.linalg import LinearOperator
from sklearn.metrics.pairwise import rbf_kernel,euclidean_distances
from sklearn.datasets import load_svmlight_file

addr = "./data/"

def load_data(dataset="satimage"):
    if dataset == "satimage":
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#satimage
        #file = "./data/satimage.scale"
        data_file = addr+"satimage.scale"
        X, y = load_svmlight_file(data_file)
        U = X.toarray()
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
