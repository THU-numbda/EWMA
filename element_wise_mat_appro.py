import numpy as np
from scipy import linalg
import scipy.sparse as ss

from sklearn.metrics.pairwise import rbf_kernel,laplacian_kernel,euclidean_distances,polynomial_kernel,sigmoid_kernel
from sklearn.base import BaseEstimator
from scipy.stats import cauchy, laplace
from utils import *


class RFF(BaseEstimator):
    def __init__(self, gamma = 1, c = 50, metric = "rbf"):
        self.gamma = gamma
        self.c = c
        self.metric = metric
        
    def transform(self,U):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        d = U.shape[1]
        #Generate c iid samples from p(w) and we don't know the p(w) for csRBF.
        if self.metric == "rbf":
            self.w = np.sqrt(2*self.gamma)*np.random.normal(size=(self.c,d))
        elif self.metric == "laplacian":
            self.w = cauchy.rvs(scale = self.gamma, size=(self.c,d))
        elif self.metric == "cauchy":
            self.w = np.random.laplace(scale = self.gamma, size=(self.c,d))

        #Generate c iid samples from Uniform(0,2*pi) 
        self.b = 2*np.pi*np.random.rand(self.c)
        #Compute feature map Z(x):
        Z = np.sqrt(2/self.c)*np.cos((U.dot(self.w.T) + self.b[np.newaxis,:]))
        return Z


class Nystrom(BaseEstimator):
    def __init__(self, gamma = 1, theta = 3.0, p=2.0, c = 100, metric = "rbf", seed = 42):
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.c = c
        self.metric = metric
        self.rng = np.random.RandomState(seed)
    def transform(self, U):
        n,d = U.shape
        idx = self.rng.choice(n, self.c, replace=False)

        U_idx = U[idx, :]
        if self.metric == "rbf":
            W = rbf_kernel(U_idx, U_idx, gamma=self.gamma)
            C = rbf_kernel(U, U_idx, gamma=self.gamma)
        elif self.metric == "csrbf":
            W = construct_csRBF(U_idx, U_idx, gamma=self.gamma,theta=self.theta,p=self.p)
            C = construct_csRBF(U, U_idx, gamma=self.gamma,theta=self.theta,p=self.p)

        u, s, _ = linalg.svd(W, full_matrices=False)
        M = np.dot(u,np.diag(1/np.sqrt(s)))
        L = C.dot(M)
        return L


class FastSPSD(BaseEstimator):
    def __init__(self, gamma = 1, theta = 3.0, p = 2.0, c = 100, s = 1000, metric = "rbf", seed = 42):
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.c = c
        self.s = s
        self.metric = metric
        self.rng = np.random.RandomState(seed)

    # #implement the corresponding leverage_score in the paper
    # #https://www.jmlr.org/papers/volume17/15-190/15-190.pdf, but show bad performance.
    # def leverage_score(self,C,idx,s):
    #     n,c = C.shape
    #     p = np.linalg.matrix_rank(C)
    #     [u,_,_] = linalg.svd(C,full_matrices=False)  
    #     norml = [np.linalg.norm(u[i,:],2)**2 for i in range(0,n)] 
    #     probability = [s*l/p for l in norml]
    #     sum_pro = np.sum(probability)
    #     probability = probability/sum_pro
    #     new_idx = self.rng.choice(n,s,replace=False,p=probability)
    #     out_idx = np.unique(list(idx)+list(new_idx))
    #     s = len(out_idx)
    #     out = np.zeros(s)
    #     for i in range(0,s):
    #         out[i] = np.sqrt(c/s/norml[out_idx[i]])
    #     return out,out_idx

    # reference: https://arxiv.org/pdf/1505.07570.pdf and https://github.com/wangshusen/RandMatrixMatlab
    def leverage_score(self,C,idx,s):
        n,c = C.shape
        QC,_ = np.linalg.qr(C,mode='reduced')
        probability = [np.linalg.norm(QC[i,:],2)**2 for i in range(0,n)] 
        sum_pro = np.sum(probability)
        probability = probability/sum_pro
        new_idx = self.rng.choice(n,s,replace=False,p=probability)
        out_idx = np.unique(list(idx)+list(new_idx))
        s = len(out_idx)
        out = np.zeros(s)
        for i in range(0,s):
            out[i] = 1.0

        return out,out_idx
        
    def transform(self, U):
        n,d = U.shape
        idx = self.rng.choice(n, self.c, replace=False)

        U_idx = U[idx, :]
        if self.metric == "rbf":
            C = rbf_kernel(U, U_idx, gamma=self.gamma)
        elif self.metric == "csrbf":
            C = construct_csRBF(U, U_idx, gamma=self.gamma, theta = self.theta,p=self.p)
        #Q,_ = np.linalg.qr(C,mode='reduced')
        Q = C
        S,pos = self.leverage_score(Q,idx,self.s)
        if self.metric == "rbf":
            K_core = rbf_kernel(U[pos,:], U[pos,:], gamma=self.gamma).reshape(len(pos),len(pos))
        elif self.metric == "csrbf":
            K_core = construct_csRBF(U[pos,:], U[pos,:], gamma=self.gamma, theta = self.theta,p=self.p).reshape(len(pos),len(pos))

        T = np.einsum('i,ij,j->ij',S,K_core,S)
        left = linalg.pinv(np.einsum('i,ij->ij',S,Q[pos,:]))
        right = linalg.pinv(np.einsum('ji,i->ji',Q[pos,:].T,S))

        W = np.dot(np.dot(left,T),right)
        return Q,W



class ssrSVD(BaseEstimator):
    def __init__(self, gamma = 1, theta = 3.0, p=2.0, d = 100, c = 100, s = 1000, z = 4, metric = "rbf", shift=False, seed = 42):
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.d = d
        self.c = c
        self.s = s
        self.z = z
        self.metric = metric
        self.shift = shift
        self.rng = np.random.RandomState(seed)

    # produce normalized custom sparse sign random matrix
    def sparse_sign_mat_custom_gen(self,n,z,k):
        S = np.sign(self.rng.randn(z,k))/np.sqrt(z)
        pos = self.rng.choice(n,k*z,replace=False)
        return S.reshape(k,z),pos

    def transform(self,U):
        n,features = U.shape
        h = self.c
        l = self.s
        z = self.z
        S,pos = self.sparse_sign_mat_custom_gen(n,z,h)
        if (self.metric == "rbf"):
            fLR_pos = rbf_kernel(U, U[pos,:], gamma=self.gamma).reshape(n,h,z)
        elif self.metric == "csrbf":
            fLR_pos = construct_csRBF(U, U[pos,:], gamma=self.gamma,theta = self.theta, p=self.p).reshape(n,h,z)
        Y = np.einsum('nhz,hz->nh',fLR_pos,S)
        Q,_ = np.linalg.qr(Y,mode='reduced')

        S,pos = self.sparse_sign_mat_custom_gen(n,z,h)
        if (self.metric == "rbf"):
            fLR_pos = rbf_kernel(U, U[pos,:], gamma=self.gamma).reshape(n,h,z)
        elif self.metric == "csrbf":
            fLR_pos = construct_csRBF(U, U[pos,:], gamma=self.gamma,theta = self.theta, p=self.p).reshape(n,h,z)
        Y = np.einsum('nhz,hz->nh',fLR_pos,S)
        P,_ = np.linalg.qr(Y,mode='reduced')

        O,pos1 = self.sparse_sign_mat_custom_gen(n,z,l)
        H,pos2 = self.sparse_sign_mat_custom_gen(n,z,l)

        if (self.metric == "rbf"):
            fLR_core = rbf_kernel(U[pos2,:], U[pos1,:], gamma=self.gamma).reshape(l,z,l,z)
        elif self.metric == "csrbf":
            fLR_core = construct_csRBF(U[pos2,:], U[pos1,:], gamma=self.gamma,theta = self.theta, p=self.p).reshape(l,z,l,z)

        Z = np.einsum('ij,ijuv,uv->iu',H,fLR_core,O)
        #Z = O.T.dot((O.T.dot(fLR_core.T)).T)
        Q_sample = Q[pos2,:].reshape(l,z,h)
        P_sample = P[pos1,:].reshape(l,z,h)
        left = np.linalg.pinv(np.einsum('lz,lzh->lh',H,Q_sample))
        right = np.linalg.pinv(np.einsum('hzl,lz->hl',P_sample.T,O))

        W = left.dot(Z).dot(right)
        u,s,vt = np.linalg.svd(W, full_matrices=False)
        u = u[:,:self.d]
        s = s[:self.d]
        vt = vt[:self.d, :]
        return Q.dot(u),s,P.dot(vt.T)


class S3SPSD(BaseEstimator):
    def __init__(self, gamma = 1, theta = 3.0, p=2.0, c = 100, s = 1000, z = 4, metric = "rbf", shift=False, seed = 42):
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.c = c
        self.s = s
        self.z = z
        self.metric = metric
        self.shift = shift
        self.rng = np.random.RandomState(seed)

    # produce normalized sparse sign random matrix
    def sparse_sign_mat_custom_gen(self,n,z,k):
        S = np.sign(self.rng.randn(z,k))/np.sqrt(z)
        pos = self.rng.choice(n,k*z,replace=False)
        return S.reshape(k,z),pos

    def transform(self,U):
        n,d = U.shape
        # sketch process to get Q
        S,pos = self.sparse_sign_mat_custom_gen(n,self.z,self.c)
        data = S.reshape(self.z*self.c,)
        i_idx = pos
        j_idx = []
        for i in range(self.c):
            for j in range(self.z):
                j_idx.append(i)
        ss_mat = ss.coo_matrix((data,(i_idx,j_idx)),shape=(n,self.c))
        if self.metric == "rbf":
            K_pos = rbf_kernel(U, U[pos,:], gamma=self.gamma).reshape(n,self.c,self.z)
            K_core = rbf_kernel(U[pos,:], U[pos,:], gamma=self.gamma).reshape(self.c,self.z,self.c,self.z)
        elif self.metric == "csrbf":
            K_pos = construct_csRBF(U, U[pos,:], gamma=self.gamma, theta = self.theta,p=self.p).reshape(n,self.c,self.z)
            K_core = construct_csRBF(U[pos,:], U[pos,:], gamma=self.gamma, theta = self.theta,p=self.p).reshape(self.c,self.z,self.c,self.z)
        Q = np.einsum('ncz,cz->nc',K_pos,S)
        N = Q.T@Q
        T = np.einsum('ij,ijuv,uv->iu',S,K_core,S)

        alpha = 0
        old_alpha = 0
        if (self.shift):
            # generate the ss_mat
            data = S.reshape(self.z*self.c,)
            i_idx = pos
            j_idx = []
            for i in range(self.c):
                for j in range(self.z):
                    j_idx.append(i)
            ss_mat = ss.coo_matrix((data,(i_idx,j_idx)),shape=(n,self.c))
            # shift boost
            for i in range(100):
                eigenvalues,_ = np.linalg.eigh(N-2*alpha*T+(alpha**2)*np.identity(self.c))
                if eigenvalues[0]<=0:
                    break
                sv = np.sqrt(eigenvalues[0])
                if alpha>sv:
                    break
                old_alpha = alpha
                alpha = (alpha + sv)/2
                if (alpha-old_alpha)/alpha<1e-6:
                    break
            Q,_,_ = np.linalg.svd(np.asarray(Q-alpha*ss_mat),full_matrices=False)
            #Q = np.asarray(Q-alpha*ss_mat)
        else:
            Q,_ = np.linalg.qr(Q,mode='reduced')
            #Q = Q
        S,pos = self.sparse_sign_mat_custom_gen(n,self.z,self.s)
        if self.metric == "rbf":
            K_core = rbf_kernel(U[pos,:], U[pos,:], gamma=self.gamma).reshape(self.s,self.z,self.s,self.z)
        elif self.metric == "csrbf":
            K_core = construct_csRBF(U[pos,:], U[pos,:], gamma=self.gamma, theta = self.theta, p=self.p).reshape(self.s,self.z,self.s,self.z)
        if self.shift==False:
            alpha = 0
        Z = np.einsum('ij,ijuv,uv->iu',S,K_core,S) - alpha*np.identity(self.s)
        Q_sample = Q[pos,:].reshape(self.s,self.z,self.c)
        left = linalg.pinv(np.einsum('sz,szc->sc',S,Q_sample))
        right = linalg.pinv(np.einsum('czs,sz->cs',Q_sample.T,S))

        W = left.dot(Z).dot(right)
        return Q,W,alpha




