from element_wise_mat_appro import *
from utils import *
import time

U = load_data("satimage")
print(U.shape)

dim = 50
seed = 1

gamma = 4.0
metric = "rbf"
K = rbf_kernel(U,U,gamma) 

#gamma = 0.05
#metric = "csrbf"
#K = construct_csRBF(U,U,gamma)

Anorm2 = ss.linalg.svds(K,k=1,which="LM",return_singular_vectors=False)
#Anorm2 = 1.0

if (metric != "csrbf"):
    # RFF
    t0 = time.time()
    rff = RFF(gamma,c=dim,metric=metric)
    Z = rff.transform(U)
    print("done RFF kernel in %0.3fs" % (time.time() - t0))
    appro_K = Z.dot(Z.T)
    LO = func(K-appro_K)
    spectral_error = ss.linalg.svds(LO,k=1,which='LM',return_singular_vectors=False)/Anorm2
    print("relative approximation error of RFF=%0.10f" % spectral_error)

# Nystrom
t0 = time.time()
nystrom = Nystrom(gamma,c=dim,metric=metric,seed=seed)
L = nystrom.transform(U)
print("done Nystrom in %0.3fs" % (time.time() - t0))
appro_K = L.dot(L.T)
LO = func(K-appro_K)
spectral_error = ss.linalg.svds(LO,k=1,which='LM',return_singular_vectors=False)/Anorm2
print("relative approximation error of Nystrom=%0.10f" % spectral_error)


# FastSPSD
t0 = time.time()
fastspsd = FastSPSD(gamma,c=dim,s=dim*5,metric=metric,seed=seed)
Q,W = fastspsd.transform(U)
print("done FastSPSD kernel in %0.3fs" % (time.time() - t0))
appro_K = Q.dot(W).dot(Q.T)
LO = func(K-appro_K)
spectral_error = ss.linalg.svds(LO,k=1,which='LM',return_singular_vectors=False)/Anorm2
print("relative approximation error of FastSPSD=%0.10f" % spectral_error)

# ssrSVD
t0 = time.time()
ssrsvd = ssrSVD(gamma,d=dim,c=dim,s=dim*5,z=4,metric=metric,shift=False,seed=seed)
Q,s,V = ssrsvd.transform(U)
print("done ssrSVD kernel in %0.3fs" % (time.time() - t0))
appro_K = Q.dot(np.diag(s)).dot(V.T)
LO = func(K-appro_K)
spectral_error = ss.linalg.svds(LO,k=1,which='LM',return_singular_vectors=False)/Anorm2
print("relative approximation error of ssrSVD =%0.10f" % spectral_error)

# shift S3SPSD
t0 = time.time()
s3spsd = S3SPSD(gamma,c=dim,s=dim*5,z=4,metric=metric,shift=True,seed=seed)
Q,X,alpha = s3spsd.transform(U)
print("done S3SPSD kernel in %0.3fs" % (time.time() - t0))
appro_K = Q.dot(X).dot(Q.T)+alpha*np.identity(U.shape[0])
LO = func(K-appro_K)
spectral_error = ss.linalg.svds(LO,k=1,which='LM',return_singular_vectors=False)/Anorm2
print("relative approximation error of S3SPSD=%0.10f" % spectral_error)





