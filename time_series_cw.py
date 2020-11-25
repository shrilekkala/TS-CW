"""
"""
import numpy as np
import matplotlib.pyplot as plt


"""
Question 1
"""
"a"

def S_AR(f, phis, sigma2):
    
    # set value of p from phis
    p = len(phis)
    
    # # generate the vector (1, -e^(-i*2*pi*f), ..., -e^(-i*2*pi*f*p))
    # v1 = np.ones(p+1, dtype = 'complex') * np.exp(--1j * 2 * np.pi)
    
    # create input vector required to construct the vandermonde matrix
    vec = np.exp(--1j * 2 * np.pi) ** f
    
    # create the Vandermonde matrix  V and the vector x
    V = np.vander(vec, p+1)
    
    x = np.ones(p+1)
    x[1:] = - phis
    
    # evaluate S using the parametric form of the sdf
    S = sigma2 / np.abs(V @ x)
    
    return S

"b"

def AR2_sim(phis, sigma2, N):
    # create a vector to store Xt
    X = np.ones(100+N)
    
    # set the intial values
    X[0] = 0
    X[1] = 0
    
    # iteratively find the values of the time series
    for i in range(2, 100+N):
        X[i] = phis[0] * X[i-1] + phis[1] * X[i-2] + np.random.normal(0, np.sqrt(sigma2))
    
    # return the vector X C with the 1st 100 values discarded
    return X[100:]

# plotting an example sim for AR(2)
plt.plot(AR2_sim((0.1, 0.1), 1, 50))
plt.xlabel('t')
plt.ylabel(r'$X_t$', rotation = 0)

"c"

def acvs_hat(X, tau):
    N = len(X)
    
    # initialise s_hat
    s_hat = np.zeros(len(tau))
    
    for i in range(len(tau)):
        # define k = the absolute value of the element of tau
        k = np.abs(tau[i])
        
        # compute the dot product between the first N - k elements of X
        # and the first N - k elements of X shifted by k, and divide by N
        s_hat[i] = (X[: N - k].T @ X[k :]) / N
    
    return s_hat