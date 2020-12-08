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
        indices = np.arange(1,4) * (N / 8)
        indices = indices.astype(int)
        
        # extract the values for frequencies 1/8, 2/8 and 3/8 and store in matrices
        p_mat[i,:] = periodogram(X)[indices]
        d_mat[i,:] = direct(X)[indices]
    
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
plt.plot(N_vals, bias_matrix[:,0], marker = '.', label = 'periodogram')
plt.plot(N_vals, bias_matrix[:,3], marker = '.', color = 'red', label = 'direct')
plt.xscale('log', basex = 2)
plt.legend()
plt.show()

# plot comparison for f = 2/8
plt.title('f = 2/8')
plt.plot(N_vals, bias_matrix[:,1], marker = '.', label = 'periodogram')
plt.plot(N_vals, bias_matrix[:,4], marker = '.', color = 'red', label = 'direct')
plt.xscale('log', basex = 2)
plt.legend()
plt.show()

# plot comparison for f = 3/8
plt.title('f = 3/8')
plt.plot(N_vals, bias_matrix[:,2], marker = '.', label = 'periodogram')
plt.plot(N_vals, bias_matrix[:,5], marker = '.', color = 'red', label = 'direct')
plt.xscale('log', basex = 2)
plt.legend()
plt.show()

# plot true S (S(f) against f)
frequencies = np.linspace(0, 1/2, 101)
sdf = S_AR(frequencies, phis1, 1)
plt.plot(frequencies, sdf)
plt.axvline(x=1/8, color = 'r', ls = 'dotted', lw=1, label = 'f = 1/8')
plt.show()

"""
Question 3
"""
time_series = np.array([0.17434,1.2875,1.6276,1.7517,2.479,1.8537,0.53564,-0.50514,-1.6363,-1.218,-2.6449,-1.013,-0.72697,0.16774,2.7097,1.7318,-0.11525,0.45508,-1.4272,-1.6926,-1.6151,-1.5088,-0.67334,0.068332,-0.74788,0.7717,0.060807,-0.92956,-0.62906,0.47694,-0.31672,1.306,0.47398,0.17882,1.7122,-0.20193,0.1694,0.20111,-1.2939,0.12961,-0.72722,-1.7166,0.635,-0.20768,-0.28336,1.8252,-0.91008,-0.14618,1.286,0.32403,0.55636,0.44725,-0.20653,0.71298,-0.22987,0.17478,-0.77277,-1.4834,-0.26414,-0.63661,0.52803,0.99871,-1.2287,-1.6065,-1.0727,-0.17734,0.34751,1.4752,1.9244,0.89304,1.6845,-0.41764,-0.22658,-0.15097,-0.89085,0.65978,0.84356,2.6663,1.9601,2.4622,1.1335,-0.27514,-0.6371,-2.4988,-2.3709,1.1849,1.0343,1.1595,2.0649,-0.75627,-1.4214,-1.3401,-2.1773,-0.88608,0.25399,0.6641,0.90835,0.51589,-0.64756,-0.72083,-1.9588,-1.6901,-1.4143,-1.2355,-0.80299,0.36638,1.8794,1.8922,0.82107,0.49564,-0.87657,-1.4168,0.086388,-1.2344,-0.031149,-0.76558,-1.0273,0.62231,0.99486,-1.6726,0.099965,1.0288,0.73741,1.951,1.0893,-0.15028,0.34172,-2.8197])

"""
"a"
"""
N = len(time_series)
# obtain the estimates
periodogram_ts = scipy.fft.fftshift(periodogram(time_series))
direct_ts = scipy.fft.fftshift(direct(time_series))
frequencies = np.linspace(-1/2, 1/2, N, endpoint = False)

# plot the estimates
plt.plot(frequencies, periodogram_ts)
plt.title("Periodogram estimate of my time series")
plt.xlabel("f")
plt.show()

plt.plot(frequencies, direct_ts)
plt.title("Direct Spectral estimate of my time series")
plt.xlabel("f")
plt.show()

"""
"b"
"""
def Yule_Walker(X, p):
    """
    Function that fits an AR(p) model  by the Yule-Walker method
    given data X and number of parameters p
    and returns the estimates for the coefficients and sigma epsilon squared
    """
    N = len(X)
    # create a vector estimate for the autocovariance sequence
    s_hat = np.zeros(p+1)
    for i in range(p):
        s_hat[i] = np.dot(X[0: N-i], X[i: N])
    s_hat = s_hat / N
    
    # create the gamma vector
    gamma_vec = s_hat[1:]
    
    # construct the GAMMA matrix by diagonals
    GAMMA_mat = np.zeros((p,p))
    for j in range(p):
        diag_index = np.arange(p-j)
        GAMMA_mat[diag_index, diag_index + j] = s_hat[j] * np.ones(p-j)
        GAMMA_mat[diag_index + j, diag_index] = s_hat[j] * np.ones(p-j)
        
    # find the vector estimate of the phis
    phis_v = np.linalg.inv(GAMMA_mat) @ gamma_vec
    
    # find the estimate of sigma_epsilon squared
    sigma_eps = s_hat[0] - np.dot(phis_v, s_hat[1:])
        
    return phis_v, sigma_eps

def Least_Squares(X, p):
    """
    Function that fits an AR(p) model by the Least Squares method
    given data X and number of parameters p
    and returns the estimates for the coefficients and sigma epsilon squared
    """
    N = len(X)
    # create vector X_v
    X_v = X[p:]
    
    # construct matrix F (column by column)
    F = np.zeros((N - p, p))
    for i in range(p):
        F[:, i] = X[(p-1-i): (N-1-i)]
        
    # find the vector estimate of the phis
    phis_v = np.linalg.inv(F.T @ F) @ F.T @ X_v
    
    # find the estimate of sigma_epsilon squared
    sigma_eps = (X_v - F @ phis_v).T @ (X_v - F @ phis_v) / (N - 2*p)

    return phis_v, sigma_eps

def Maximum_Likelihood(X, p):
    """
    Function that fits an AR(p) model by the approximate Maximum Likelihood method
    given data X and number of parameters p
    and returns the estimates for the coefficients and sigma epsilon squared
    """
    N = len(X)
    
    # obtain the least squares estimates for the vectors of phis and sigma epsilon squared
    phis_v, LS_sigma_eps = Least_Squares(X, p)
    
    # create the maximum likelihood estimator of sigma_eps from the least squares one
    sigma_eps = LS_sigma_eps * (N - 2*p) / (N - p)
    
    return phis_v, sigma_eps
    
"""
"c"
"""
# create a 20 x 3 matrix that stores sigma_eps for each p and each method
sigma_mat = np.zeros((20, 3))
for p in range(20):
    sigma_mat[p, 0] = Yule_Walker(time_series, p+1)[1]
    sigma_mat[p, 1] = Least_Squares(time_series, p+1)[1]
    sigma_mat[p, 2] = Maximum_Likelihood(time_series, p+1)[1]

# construct a 20 x 3 matrix of AIC values for each p and each method
p_mat = np.array([np.arange(1,21)]*3).T
AIC = 2 * p_mat + N * np.log(sigma_mat)

"""
"d"
"""
# Find the p with the lowest AIC for each method
p_min = np.argmin(AIC, axis = 0) + 1

# Find the parameter values for the chosen p for each method
YW_phis, YW_sigma_eps = Yule_Walker(time_series, 5)
YW_phis
YW_sigma_eps

LS_phis, LS_sigma_eps = Least_Squares(time_series, x)
LS_phis
LS_sigma_eps

ML_phis, ML_sigma_eps = Maximum_Likelihood(time_series, x)
ML_phis
ML_sigma_eps

"""
e)
"""
f = np.linspace(-1/2, 1/2, 101)
# Compute the 3 sdfs using the S_AR function
S_YW = S_AR(f, YW_phis, YW_sigma_eps)
S_LS = S_AR(f, LS_phis, LS_sigma_eps)
S_ML = S_AR(f, ML_phis, ML_sigma_eps)

# plot the sdfs
plt.plot(f, S_YW, label = "Yule-Walker")
plt.plot(f, S_LS, label = "Least Squares")
plt.plot(f, S_ML, label = "Approximate Maximum Likelihood")
plt.xlabel("f")
plt.ylabel("S(f)")
plt.legend(loc = "upper left", fontsize = 5)
plt.show()