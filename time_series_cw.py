"""
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

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
    
    # create vector of epsilons via Multivariate Normal Distribution
    eps = np.random.normal(0, np.sqrt(sigma2), 100+N)
    
    # iteratively find the values of the time series
    for i in range(2, 100+N):
        X[i] = phis[0] * X[i-1] + phis[1] * X[i-2] + eps[i]
    
    # return the vector X C with the 1st 100 values discarded
    return X[100:]

# plotting an example sim for AR(2)
plt.plot(AR2_sim((0.8, 0), 1, 50))
plt.xlabel('t')
plt.ylabel(r'$X_t$', rotation = 0)
plt.show()


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

    
"b (B) "
# compute the true sdf at frequencies 1/8, 2/8 and 3/8
freq = np.array([1/8, 2/8, 3/8])
true_s = 1 / ((1 - 2*0.95*np.cos(2*np.pi*(1/8 + freq)) + 0.95**2)*(1 - 2*0.95*np.cos(2*np.pi*(1/8 - freq)) + 0.95**2))

# compute the sample mean for the periodogram and the direct spectral estimator
p_sample_mean = np.mean(p_mat, axis=0)
d_sample_mean = np.mean(d_mat, axis=0)

# compute the absolute value of the sample bias
p_empirical_bias = np.abs(p_sample_mean - true_s)
d_empirical_bias = np.abs(d_sample_mean - true_s)

"b (C)"
# store the parameters for the AR2
phis1 = np.array([np.sqrt(2) * 0.95, - 0.95 ** 2])

# create a function that goes through steps A) and B) for a specific N
def spectral_estimators(N):
    # create matrices to store the periodogram / direct spectral estimates
    p_mat = np.ones((10000, 3))
    d_mat = np.ones((10000, 3))
    
    for i in range(10000):
        X = AR2_sim(phis1, 1, N)
        
        # the indices for 1/8, 2/8 and 3/8 from the fourier frequencies
        indicies = np.arange(1,4) * (N / 8)
        indicies = indicies.astype(int)
        
        # extract the values for frequencies 1/8, 2/8 and 3/8 and store in matrices
        p_mat[i,:] = periodogram(X)[indicies]
        d_mat[i,:] = direct(X)[indicies]
    
    # compute the absolute value of the sample bias
    p_empirical_bias = np.abs(np.mean(p_mat, axis=0) - true_s)
    d_empirical_bias = np.abs(np.mean(d_mat, axis=0) - true_s)
    
    return p_empirical_bias, d_empirical_bias

# generate N values of powers of 2 from 16 to 4096
N_vals = 2 ** np.arange(4,13)

# create a 9 x 6 matrix to store the empirical bias for different frequencies and different estimators
bias_matrix = np.zeros((9,6))

start = time.time()

for i in range(len(N_vals)):
    ### just for output ###
    print(i)
    p_bias, d_bias = spectral_estimators(N_vals[i])
    bias_matrix[i, 0:3] = p_bias
    bias_matrix[i, 3:6] = d_bias

end = time.time()
print("time taken: ", end - start)

"b (D)"
# plot comparison for f = 1/8
plt.title('f = 1/8')
plt.plot(N_vals, bias_matrix[:,0], marker = '.')
plt.plot(N_vals, bias_matrix[:,3], marker = '.', color = 'red')
plt.xscale('log', basex = 2)
plt.show()

# plot comparison for f = 2/8
plt.title('f = 2/8')
plt.plot(N_vals, bias_matrix[:,1], marker = '.')
plt.plot(N_vals, bias_matrix[:,4], marker = '.', color = 'red')
plt.xscale('log', basex = 2)
plt.show()

# plot comparison for f = 3/8
plt.title('f = 3/8')
plt.plot(N_vals, bias_matrix[:,2], marker = '.')
plt.plot(N_vals, bias_matrix[:,5], marker = '.', color = 'red')
plt.xscale('log', basex = 2)
plt.show()

# plot true S (S(f) against f)
frequencies = np.linspace(0, 1/2, 101)
sdf = S_AR(frequencies, phis1, 1)
plt.plot(frequencies, sdf)
plt.axvline(x=1/8, color = 'r', ls = 'dotted', lw=1, label = 'f = 1/8')
plt.show()