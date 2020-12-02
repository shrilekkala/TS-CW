"""
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
Question 1
"""
"a"


def S_AR(f, phis, sigma2):
    
    # set value of p from phis
    p = len(phis)
    
    # initialise S
    S = f.copy()

    
    # create vectors phis1 = ([1, -phi_1p, -phi_2p, .., -phi_pp])
    phis1 = np.ones(p + 1)
    phis1[1:] = - phis
    
    for i in range(len(f)):
        # generate the vector (1, e^(-i*2*pi*f), ..., -^(-i*2*pi*f*p))
        v1 = np.ones(p+1, dtype = 'complex') * np.exp(-1j * 2 * np.pi * f[i] * np.arange(p+1))
        
        S[i] = sigma2 / (np.abs(phis1.T @ v1) ** 2)
    
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
plt.plot(AR2_sim((0.8, 0), 1, 50))
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

# check part c
AR1 = AR2_sim((0.8, 0), 1, 500)
tot = np.zeros(3)
for i in range(200):
    tot += acvs_hat(AR2_sim((0.8, 0), 1, 500), np.array([0,1,2]))
avg = tot / 1000
print(avg)
#print(acvs_hat(AR2_sim((0.8, 0), 1, 5000), np.array([0,1,2])))


"""
Question 2
"""
"a"

def periodogram(X):
    N = len(X)
    
    # compute the periodogram Sp using the fft algorithm
    Sp = (np.abs(scipy.fft.fft(X)) ** 2) / N
    
    return Sp

def direct(X):
    N = len(X)
    
    # construct the taper h_t
    h = np.arange(1, N+1)
    h = 0.5 * np.sqrt(8 / (3 * (N + 1))) * (1 - np.cos(h * 2 * np.pi/(N+1)))
    
    # compute the direct spectral estimate Sd using the fft algorithm
    Sd = np.abs(scipy.fft.fft(h * X)) ** 2
    
    return Sd
    

"b (A) "
# store the parameters for the AR2
phis1 = np.array([np.sqrt(2) * 0.95, - 0.95 ** 2])

# create matrices to store the periodogram / direct spectral estimates
p_mat = np.ones((10000, 3))
d_mat = np.ones((10000, 3))

for i in range(10000):
    X = AR2_sim(phis1, 1, 16)
    # extract the values for frequencies 1/8, 2/8 and 3/8 and store in matrices
    p_mat[i,:] = periodogram(X)[[2, 4, 6]]
    d_mat[i,:] = direct(X)[[2, 4, 6]]
    
    
    
